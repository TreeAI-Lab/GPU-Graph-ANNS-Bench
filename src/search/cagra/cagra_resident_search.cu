// CAGRA search (cuVS C++ API) for a graph-only index.
// Attaches the base vectors once (mmap of base.fbin, copied to the GPU by update_dataset), then times each query
// batch as H2D, cagra::search and top-k D2H with CUDA events.
// With --it-file: only the first batch is searched; per max_iterations value one untimed warm-up run, then three
// timed runs (kTimedRuns), with itopk_size = max(32, it rounded up to a multiple of 32). sweep.tsv holds the means
// over the timed runs (qps = nq / mean wall time) and the sample standard deviations; it_batches.tsv holds every run.

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_mdspan.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuvs/neighbors/cagra.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

// Timed runs per setting of a sweep, after one untimed warm-up run
constexpr int kTimedRuns = 3;

struct Mapped {
  int fd = -1;
  void* map = MAP_FAILED;
  size_t bytes = 0;
  const char* base = nullptr;
  int64_t n = 0;
  int64_t dim = 0;

  Mapped() = default;
  Mapped(const Mapped&) = delete;
  Mapped& operator=(const Mapped&) = delete;
  Mapped(Mapped&& o) noexcept { *this = std::move(o); }
  Mapped& operator=(Mapped&& o) noexcept {
    if (this == &o) return *this;
    close_all();
    fd = o.fd;
    map = o.map;
    bytes = o.bytes;
    base = o.base;
    n = o.n;
    dim = o.dim;
    o.fd = -1;
    o.map = MAP_FAILED;
    o.bytes = 0;
    o.base = nullptr;
    o.n = 0;
    o.dim = 0;
    return *this;
  }
  ~Mapped() { close_all(); }

  void close_all() {
    if (map != MAP_FAILED) munmap(map, bytes);
    if (fd >= 0) close(fd);
    map = MAP_FAILED;
    fd = -1;
  }
};

Mapped map_file(const std::string& path) {
  Mapped m;
  m.fd = open(path.c_str(), O_RDONLY);
  if (m.fd < 0) throw std::runtime_error("open failed: " + path);
  struct stat st;
  if (fstat(m.fd, &st) != 0) throw std::runtime_error("fstat failed: " + path);
  m.bytes = static_cast<size_t>(st.st_size);
  m.map = mmap(nullptr, m.bytes, PROT_READ, MAP_PRIVATE, m.fd, 0);
  if (m.map == MAP_FAILED) throw std::runtime_error("mmap failed: " + path);
  madvise(m.map, m.bytes, MADV_SEQUENTIAL);
  m.base = static_cast<const char*>(m.map);
  return m;
}

void read_header(const Mapped& m, const std::string& path) {
  if (m.bytes < 8) throw std::runtime_error("file too small: " + path);
}

Mapped map_fbin(const std::string& path) {
  Mapped m = map_file(path);
  read_header(m, path);
  int32_t header[2];
  std::memcpy(header, m.base, sizeof(header));
  m.n = header[0];
  m.dim = header[1];
  const size_t need = 8ull + static_cast<size_t>(m.n) * static_cast<size_t>(m.dim) * sizeof(float);
  if (m.n <= 0 || m.dim <= 0 || need != m.bytes) {
    throw std::runtime_error("fbin header does not match file size: " + path);
  }
  return m;
}

Mapped map_ibin(const std::string& path) {
  Mapped m = map_file(path);
  read_header(m, path);
  int32_t header[2];
  std::memcpy(header, m.base, sizeof(header));
  m.n = header[0];
  m.dim = header[1];
  const size_t need = 8ull + static_cast<size_t>(m.n) * static_cast<size_t>(m.dim) * sizeof(int32_t);
  if (m.n <= 0 || m.dim <= 0 || need != m.bytes) {
    throw std::runtime_error("ibin header does not match file size: " + path);
  }
  return m;
}

struct Args {
  std::string index, base, query, gt, out, it_file;
  int batch = 10000;
  int k = 10;
  int max_iterations = 55;
  int itopk = 64;
  int search_width = 1;
  int max_batches = 0;
  int gpu = 0;
};

void usage() {
  std::cerr
    << "Usage: cagra_resident_search --index IDX --base BASE.fbin --query QUERY.fbin --out DIR\n"
    << "  [--gt GT.ibin] [--batch-size 10000] [--k 10] [--max-iterations 55]\n"
    << "  [--itopk-size 64] [--search-width 1] [--max-batches 0] [--gpu 0]\n"
    << "  [--it-file schedule.txt]  first-batch sweep, 1 warm-up and 3 timed runs per it, itopk=ceil32(it)\n";
}

Args parse_args(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; i++) {
    std::string k = argv[i];
    auto need = [&](const char* name) -> std::string {
      if (i + 1 >= argc) throw std::runtime_error(std::string("missing value for ") + name);
      return argv[++i];
    };
    if (k == "--index") a.index = need("--index");
    else if (k == "--base") a.base = need("--base");
    else if (k == "--query") a.query = need("--query");
    else if (k == "--gt") a.gt = need("--gt");
    else if (k == "--out") a.out = need("--out");
    else if (k == "--batch-size") a.batch = std::stoi(need("--batch-size"));
    else if (k == "--k") a.k = std::stoi(need("--k"));
    else if (k == "--max-iterations") a.max_iterations = std::stoi(need("--max-iterations"));
    else if (k == "--itopk-size") a.itopk = std::stoi(need("--itopk-size"));
    else if (k == "--search-width") a.search_width = std::stoi(need("--search-width"));
    else if (k == "--max-batches") a.max_batches = std::stoi(need("--max-batches"));
    else if (k == "--gpu") a.gpu = std::stoi(need("--gpu"));
    else if (k == "--it-file") a.it_file = need("--it-file");
    else if (k == "-h" || k == "--help") {
      usage();
      std::exit(0);
    } else {
      throw std::runtime_error("unknown arg " + k);
    }
  }
  if (a.index.empty() || a.base.empty() || a.query.empty() || a.out.empty()) {
    usage();
    throw std::runtime_error("missing required args");
  }
  if (a.batch <= 0 || a.k <= 0 || a.max_iterations <= 0) {
    throw std::runtime_error("batch/k/max-iterations must be positive");
  }
  return a;
}

#define CUDA_CHECK(call)                                                                 \
  do {                                                                                   \
    cudaError_t err_ = (call);                                                           \
    if (err_ != cudaSuccess) {                                                           \
      throw std::runtime_error(std::string("CUDA error: ") + cudaGetErrorString(err_)); \
    }                                                                                    \
  } while (0)

double host_rss_mib() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) == 0) return std::atof(line.c_str() + 6) / 1024.0;
  }
  return 0;
}

void append_mem(const std::string& path, const std::string& stage) {
  size_t free_b = 0, tot_b = 0;
  CUDA_CHECK(cudaMemGetInfo(&free_b, &tot_b));
  double used = (tot_b - free_b) / 1048576.0;
  double free_m = free_b / 1048576.0;
  double tot_m = tot_b / 1048576.0;
  bool neu = access(path.c_str(), F_OK) != 0;
  std::ofstream f(path, std::ios::app);
  if (neu) f << "stage\tgpu_used_mib\tgpu_free_mib\tgpu_total_mib\thost_rss_mib\n";
  f << stage << "\t" << used << "\t" << free_m << "\t" << tot_m << "\t" << host_rss_mib() << "\n";
}

float elapsed_ms(cudaEvent_t a, cudaEvent_t b) {
  float ms = 0;
  CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
  return ms;
}

double mean_of(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return v.empty() ? 0.0 : s / v.size();
}

// Sample standard deviation (n - 1), 0 for a single value
double sd_of(const std::vector<double>& v) {
  if (v.size() < 2) return 0.0;
  const double m = mean_of(v);
  double s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / (v.size() - 1));
}

std::vector<int> load_its(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("open it-file: " + path);
  std::vector<int> v;
  std::string line;
  while (std::getline(in, line)) {
    auto p = line.find('#');
    if (p != std::string::npos) line = line.substr(0, p);
    size_t a = line.find_first_not_of(" \t\r");
    if (a == std::string::npos) continue;
    v.push_back(std::atoi(line.c_str() + a));
  }
  if (v.empty()) throw std::runtime_error("empty it-file");
  return v;
}

int itopk_for_it(int it, int k) {
  int need = std::max(it, std::max(k, 32));
  return ((need + 31) / 32) * 32;
}

int intersect_k(const uint32_t* pred, const int32_t* gt, int k) {
  int32_t pset[64];
  int32_t gset[64];
  if (k > 64) throw std::runtime_error("k>64 not supported in recall");
  int np = 0, ng = 0;
  for (int a = 0; a < k; a++) {
    int32_t id = static_cast<int32_t>(pred[a]);
    bool seen = false;
    for (int j = 0; j < np; j++)
      if (pset[j] == id) seen = true;
    if (!seen) pset[np++] = id;
  }
  for (int b = 0; b < k; b++) {
    int32_t id = gt[b];
    bool seen = false;
    for (int j = 0; j < ng; j++)
      if (gset[j] == id) seen = true;
    if (!seen) gset[ng++] = id;
  }
  int hits = 0;
  for (int a = 0; a < np; a++) {
    for (int b = 0; b < ng; b++) {
      if (pset[a] == gset[b]) {
        hits++;
        break;
      }
    }
  }
  return hits;
}

}  // namespace

int main(int argc, char** argv) {
  std::cout.setf(std::ios::unitbuf);
  std::cerr.setf(std::ios::unitbuf);
  try {
    Args args = parse_args(argc, argv);
    CUDA_CHECK(cudaSetDevice(args.gpu));

    raft::device_resources dev_resources;
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
      rmm::mr::get_current_device_resource(), 3ull * 1024 * 1024 * 1024);
    rmm::mr::set_current_device_resource(&pool_mr);

    if (mkdir(args.out.c_str(), 0755) != 0 && errno != EEXIST) {
      throw std::runtime_error("mkdir failed: " + args.out);
    }

    auto t_load0 = std::chrono::steady_clock::now();
    std::cout << "[cagra] deserializing " << args.index << "\n";
    cuvs::neighbors::cagra::index<float, uint32_t> index(dev_resources);
    cuvs::neighbors::cagra::deserialize(dev_resources, args.index, &index);
    raft::resource::sync_stream(dev_resources);
    std::cout << "[cagra] graph n=" << index.size() << " degree=" << index.graph_degree()
              << " graph=[" << index.graph().extent(0) << "," << index.graph().extent(1) << "]\n";

    std::cout << "[cagra] mapping base " << args.base << "\n";
    Mapped base = map_fbin(args.base);
    if (static_cast<int64_t>(index.size()) != base.n) {
      throw std::runtime_error("base rows " + std::to_string(base.n) + " != index size " +
                               std::to_string(index.size()));
    }
    std::cout << "[cagra] attaching dataset n=" << base.n << " dim=" << base.dim << "\n";
    const float* base_ptr = reinterpret_cast<const float*>(base.base + 8);
    auto host_ds = raft::make_host_matrix_view<const float, int64_t>(base_ptr, base.n, base.dim);
    index.update_dataset(dev_resources, host_ds);
    raft::resource::sync_stream(dev_resources);
    std::cout << "[cagra] dataset attached rows=" << index.dataset().extent(0)
              << " dim=" << index.dataset().extent(1) << "\n";

    Mapped query = map_fbin(args.query);
    if (query.dim != base.dim) {
      throw std::runtime_error("query dim != base dim");
    }
    const float* q_ptr = reinterpret_cast<const float*>(query.base + 8);

    const int32_t* gt_ptr = nullptr;
    Mapped gt;
    if (!args.gt.empty()) {
      gt = map_ibin(args.gt);
      if (gt.dim < args.k) throw std::runtime_error("gt k < search k");
      gt_ptr = reinterpret_cast<const int32_t*>(gt.base + 8);
      std::cout << "[cagra] GT loaded (" << gt.n << ", " << gt.dim << ")\n";
    } else {
      std::cout << "[cagra] no GT; recall column will be empty\n";
    }

    int64_t n_batches = (query.n + args.batch - 1) / args.batch;
    if (args.max_batches > 0) n_batches = std::min(n_batches, static_cast<int64_t>(args.max_batches));
    int64_t n_search = 0;
    for (int64_t bi = 0; bi < n_batches; bi++) {
      int64_t s = bi * args.batch;
      int64_t e = std::min(s + args.batch, query.n);
      n_search += e - s;
    }
    if (gt_ptr && gt.n < n_search) {
      throw std::runtime_error("gt nq " + std::to_string(gt.n) + " < queries searched " +
                               std::to_string(n_search));
    }

    using namespace cuvs::neighbors;
    cagra::search_params sp;
    sp.max_iterations = static_cast<size_t>(args.max_iterations);
    sp.itopk_size = static_cast<size_t>(args.itopk);
    sp.search_width = static_cast<size_t>(args.search_width);

    const int bs = args.batch;
    const int64_t dim = query.dim;
    auto d_q = raft::make_device_matrix<float, int64_t>(dev_resources, bs, dim);
    auto d_n = raft::make_device_matrix<uint32_t, int64_t>(dev_resources, bs, args.k);
    auto d_d = raft::make_device_matrix<float, int64_t>(dev_resources, bs, args.k);

    float* h_q = nullptr;
    uint32_t* h_n = nullptr;
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_q), static_cast<size_t>(bs) * dim * sizeof(float)));
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&h_n), static_cast<size_t>(bs) * args.k * sizeof(uint32_t)));

    cudaStream_t stream = raft::resource::get_cuda_stream(dev_resources);
    cudaEvent_t ev_h0, ev_h1, ev_k0, ev_k1, ev_d0, ev_d1;
    CUDA_CHECK(cudaEventCreate(&ev_h0));
    CUDA_CHECK(cudaEventCreate(&ev_h1));
    CUDA_CHECK(cudaEventCreate(&ev_k0));
    CUDA_CHECK(cudaEventCreate(&ev_k1));
    CUDA_CHECK(cudaEventCreate(&ev_d0));
    CUDA_CHECK(cudaEventCreate(&ev_d1));

    double load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();
    {
      std::ofstream f(args.out + "/load_timing.tsv");
      f << "stage\tms\n";
      f << "load_index_and_init\t" << load_ms << "\n";
    }
    append_mem(args.out + "/gpu_mem_cuda.tsv", "after_load");

    {
      std::ofstream f(args.out + "/params.txt");
      f << "algo\tCAGRA\n";
      f << "index\t" << args.index << "\n";
      f << "base\t" << args.base << "\n";
      f << "query\t" << args.query << "\n";
      f << "gt\t" << args.gt << "\n";
      f << "batch_size\t" << bs << "\n";
      f << "k\t" << args.k << "\n";
      f << "max_iterations\t" << args.max_iterations << "\n";
      f << "itopk_size\t" << args.itopk << "\n";
      f << "search_width\t" << args.search_width << "\n";
      f << "n_query_total\t" << query.n << "\n";
      f << "dim\t" << dim << "\n";
      f << "gpu\t" << args.gpu << "\n";
      f << "note\tgraph-only index; dataset attached in-process from base fbin\n";
      f << "h2d_src\tcudaMallocHost pinned (mmap copy is outside H2D event)\n";
    }

    std::cout << "[cagra] " << query.n << " queries, batch=" << bs << ", batches=" << n_batches << "\n";

    if (!args.it_file.empty()) {
      const int nq = static_cast<int>(std::min<int64_t>(bs, query.n));
      if (gt_ptr && gt.n < nq) throw std::runtime_error("gt nq < sweep nq");
      std::memcpy(h_q, q_ptr, static_cast<size_t>(nq) * dim * sizeof(float));
      auto its = load_its(args.it_file);
      {
        std::ofstream f(args.out + "/params.txt", std::ios::app);
        f << "protocol\tfirst_batch_only_warmup1_timed" << kTimedRuns << "_mean\n";
        f << "it_file\t" << args.it_file << "\n";
        f << "n_it\t" << its.size() << "\n";
        f << "itopk_policy\tlinear_round32: itopk=max(32, ceil32(it))\n";
      }
      std::ofstream sf(args.out + "/sweep.tsv");
      std::ofstream df(args.out + "/it_batches.tsv");
      sf << "it\titopk\tnq\twarmup_kernel_ms\th2d_query_ms\tkernel_ms\td2h_topk_ms\t"
            "wall_ms\tqps\trecall@10\tgpu_used_mib\tkernel_ms_std\twall_ms_std\tn_timed\n";
      df << "it\titopk\tphase\tnq\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10\n";
      auto qview = raft::make_device_matrix_view<const float, int64_t>(d_q.data_handle(), nq, dim);
      auto nview = raft::make_device_matrix_view<uint32_t, int64_t>(d_n.data_handle(), nq, args.k);
      auto dview = raft::make_device_matrix_view<float, int64_t>(d_d.data_handle(), nq, args.k);
      const int gtk = gt_ptr ? static_cast<int>(gt.dim) : 0;
      auto run_once = [&](int it, int itopk, const char* phase, float* oh, float* ok, float* od,
                          double* ow, double* oq, std::string* orec) {
        cagra::search_params sp;
        sp.max_iterations = static_cast<size_t>(it);
        sp.itopk_size = static_cast<size_t>(itopk);
        sp.search_width = static_cast<size_t>(args.search_width);
        auto wall0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaEventRecord(ev_h0, stream));
        CUDA_CHECK(cudaMemcpyAsync(d_q.data_handle(), h_q,
                                   static_cast<size_t>(nq) * dim * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaEventRecord(ev_h1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_h1));
        float h2d_ms = elapsed_ms(ev_h0, ev_h1);
        CUDA_CHECK(cudaEventRecord(ev_k0, stream));
        cagra::search(dev_resources, sp, index, qview, nview, dview);
        CUDA_CHECK(cudaEventRecord(ev_k1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_k1));
        float kernel_ms = elapsed_ms(ev_k0, ev_k1);
        CUDA_CHECK(cudaEventRecord(ev_d0, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_n, d_n.data_handle(),
                                   static_cast<size_t>(nq) * args.k * sizeof(uint32_t),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaEventRecord(ev_d1, stream));
        CUDA_CHECK(cudaEventSynchronize(ev_d1));
        float d2h_ms = elapsed_ms(ev_d0, ev_d1);
        double wall_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
        double qps = wall_ms > 0 ? nq / (wall_ms / 1000.0) : 0.0;
        std::string rec;
        if (gt_ptr) {
          long hits = 0;
          for (int i = 0; i < nq; i++) {
            hits += intersect_k(h_n + static_cast<size_t>(i) * args.k,
                                gt_ptr + static_cast<size_t>(i) * gtk, args.k);
          }
          char buf[32];
          std::snprintf(buf, sizeof(buf), "%.6f",
                        static_cast<double>(hits) / (static_cast<double>(nq) * args.k));
          rec = buf;
        }
        df << it << "\t" << itopk << "\t" << phase << "\t" << nq << "\t" << h2d_ms << "\t" << kernel_ms
           << "\t" << d2h_ms << "\t" << wall_ms << "\t" << qps << "\t" << rec << "\n";
        df.flush();
        *oh = h2d_ms;
        *ok = kernel_ms;
        *od = d2h_ms;
        *ow = wall_ms;
        *oq = qps;
        *orec = rec;
      };
      for (size_t i = 0; i < its.size(); i++) {
        int it = its[i];
        int itopk = itopk_for_it(it, args.k);
        float wh, wk, wd;
        double ww, wq;
        std::string wrec;
        run_once(it, itopk, "warmup", &wh, &wk, &wd, &ww, &wq, &wrec);
        std::vector<double> h2d_v, kern_v, d2h_v, wall_v, rec_v;
        for (int r = 0; r < kTimedRuns; ++r) {
          float th, tk, td;
          double tw, tq;
          std::string trec;
          const std::string phase = "timed" + std::to_string(r + 1);
          run_once(it, itopk, phase.c_str(), &th, &tk, &td, &tw, &tq, &trec);
          h2d_v.push_back(th);
          kern_v.push_back(tk);
          d2h_v.push_back(td);
          wall_v.push_back(tw);
          if (!trec.empty()) rec_v.push_back(std::atof(trec.c_str()));
        }
        const double h2d = mean_of(h2d_v), kern = mean_of(kern_v), d2h = mean_of(d2h_v);
        const double wall = mean_of(wall_v);
        const double qps = wall > 0 ? nq / (wall / 1000.0) : 0.0;
        std::string rec;
        if (!rec_v.empty()) {
          char buf[32];
          std::snprintf(buf, sizeof(buf), "%.6f", mean_of(rec_v));
          rec = buf;
        }
        size_t free_b = 0, tot_b = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_b, &tot_b));
        double used = (tot_b - free_b) / 1048576.0;
        if (i == 0) append_mem(args.out + "/gpu_mem_cuda.tsv", "after_first_it");
        sf << it << "\t" << itopk << "\t" << nq << "\t" << wk << "\t" << h2d << "\t" << kern << "\t"
           << d2h << "\t" << wall << "\t" << qps << "\t" << rec << "\t" << used << "\t" << sd_of(kern_v)
           << "\t" << sd_of(wall_v) << "\t" << kTimedRuns << "\n";
        sf.flush();
        std::printf("[cagra] it=%d/%d itopk=%d nq=%d h2d=%.2f kern=%.2f d2h=%.2f qps=%.0f recall=%s (mean of %d)\n",
                    it, its.back(), itopk, nq, h2d, kern, d2h, qps, rec.c_str(), kTimedRuns);
        std::fflush(stdout);
      }
      append_mem(args.out + "/gpu_mem_cuda.tsv", "after_last_it");
      CUDA_CHECK(cudaFreeHost(h_q));
      CUDA_CHECK(cudaFreeHost(h_n));
      std::cout << "[cagra] sweep done -> " << args.out << "\n";
      return 0;
    }

    std::ofstream bf(args.out + "/batch_timing.tsv");
    std::ofstream rf(args.out + "/recall.tsv");
    bf << "batch_id\tnq\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10\n";
    rf << "batch_id\trecall@10\n";
    bf.flush();
    rf.flush();

    for (int64_t bi = 0; bi < n_batches; bi++) {
      int64_t s = bi * static_cast<int64_t>(bs);
      int64_t e = std::min(s + bs, query.n);
      int nq = static_cast<int>(e - s);
      std::memcpy(h_q, q_ptr + s * dim, static_cast<size_t>(nq) * dim * sizeof(float));

      auto qview = raft::make_device_matrix_view<const float, int64_t>(d_q.data_handle(), nq, dim);
      auto nview = raft::make_device_matrix_view<uint32_t, int64_t>(d_n.data_handle(), nq, args.k);
      auto dview = raft::make_device_matrix_view<float, int64_t>(d_d.data_handle(), nq, args.k);

      auto wall0 = std::chrono::steady_clock::now();

      CUDA_CHECK(cudaEventRecord(ev_h0, stream));
      CUDA_CHECK(cudaMemcpyAsync(d_q.data_handle(), h_q,
                                 static_cast<size_t>(nq) * dim * sizeof(float),
                                 cudaMemcpyHostToDevice, stream));
      CUDA_CHECK(cudaEventRecord(ev_h1, stream));
      CUDA_CHECK(cudaEventSynchronize(ev_h1));
      float h2d_ms = elapsed_ms(ev_h0, ev_h1);

      CUDA_CHECK(cudaEventRecord(ev_k0, stream));
      cagra::search(dev_resources, sp, index, qview, nview, dview);
      CUDA_CHECK(cudaEventRecord(ev_k1, stream));
      CUDA_CHECK(cudaEventSynchronize(ev_k1));
      float kernel_ms = elapsed_ms(ev_k0, ev_k1);

      CUDA_CHECK(cudaEventRecord(ev_d0, stream));
      CUDA_CHECK(cudaMemcpyAsync(h_n, d_n.data_handle(),
                                 static_cast<size_t>(nq) * args.k * sizeof(uint32_t),
                                 cudaMemcpyDeviceToHost, stream));
      CUDA_CHECK(cudaEventRecord(ev_d1, stream));
      CUDA_CHECK(cudaEventSynchronize(ev_d1));
      float d2h_ms = elapsed_ms(ev_d0, ev_d1);

      double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
      double qps = wall_ms > 0 ? nq / (wall_ms / 1000.0) : 0.0;

      char rec[32];
      rec[0] = 0;
      if (gt_ptr) {
        long hits = 0;
        const int gtk = static_cast<int>(gt.dim);
        for (int i = 0; i < nq; i++) {
          hits += intersect_k(h_n + static_cast<size_t>(i) * args.k,
                              gt_ptr + (s + i) * gtk, args.k);
        }
        double r = static_cast<double>(hits) / (static_cast<double>(nq) * args.k);
        std::snprintf(rec, sizeof(rec), "%.6f", r);
        rf << bi << "\t" << rec << "\n";
        rf.flush();
      }

      char line[256];
      std::snprintf(line, sizeof(line),
                    "%ld\t%d\t%.4f\t%.4f\t%.4f\t%.4f\t%.2f\t%s\n",
                    static_cast<long>(bi), nq, h2d_ms, kernel_ms, d2h_ms, wall_ms, qps, rec);
      bf << line;
      bf.flush();

      if (bi == 0) append_mem(args.out + "/gpu_mem_cuda.tsv", "after_first_batch");
      if (bi == n_batches - 1) append_mem(args.out + "/gpu_mem_cuda.tsv", "after_last_batch");
      if (bi % 10 == 0 || bi == n_batches - 1) {
        std::printf("[cagra] batch %ld/%ld nq=%d h2d=%.2f kern=%.2f d2h=%.2f wall=%.2f qps=%.0f recall=%s\n",
                    static_cast<long>(bi), static_cast<long>(n_batches - 1), nq, h2d_ms, kernel_ms,
                    d2h_ms, wall_ms, qps, rec);
        std::fflush(stdout);
      }
    }

    CUDA_CHECK(cudaFreeHost(h_q));
    CUDA_CHECK(cudaFreeHost(h_n));
    std::cout << "[cagra] done -> " << args.out << "\n";
    return 0;
  } catch (const std::exception& e) {
    std::cerr << "ERROR: " << e.what() << "\n";
    return 1;
  }
}
