// GGNN search with base vectors and graph resident in GPU memory (GGNN library API).
// Loads the base (fvecs, or fbin through mmap) and the graph (part_<i>.ggnn) once; per batch: query H2D,
// GGNN query and top-k D2H, each timed with CUDA events.
// With --it-file: only the first batch is searched; per iteration count one untimed warm-up run, then three timed
// runs (kTimedRuns). sweep.tsv holds the means over the timed runs (qps = nq / mean wall time) and the sample
// standard deviations; it_batches.tsv holds every run.
//
// usage:
//   ggnn_resident_search \
//     --base base.fvecs --graph-dir <dir with part_0.ggnn> --query query.fbin \
//     --out OUT --k-build 32 --k-query 10 --tau-query 0.5 --max-iterations 55 \
//     --batch-size 10000 --gpu 0 [--gt gt.k10.ibin] [--it-file it_schedule.txt] [--shard-size N]
#include <ggnn/base/ggnn.cuh>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace fs = std::filesystem;

// Timed runs per setting of a sweep, after one untimed warm-up run
constexpr int kTimedRuns = 3;

static void die(const char* msg) {
  std::fprintf(stderr, "ERROR: %s\n", msg);
  std::exit(1);
}

static void check_cuda(cudaError_t e, const char* what) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA %s: %s\n", what, cudaGetErrorString(e));
    std::exit(1);
  }
}

static double host_rss_mib() {
  std::ifstream in("/proc/self/status");
  std::string line;
  while (std::getline(in, line)) {
    if (line.rfind("VmRSS:", 0) == 0) {
      return std::atol(line.c_str() + 6) / 1024.0;
    }
  }
  return 0;
}

static std::vector<int> load_its(const std::string& path, int fallback) {
  std::vector<int> v;
  if (path.empty()) {
    v.push_back(fallback);
    return v;
  }
  std::ifstream in(path);
  if (!in) die("open it-file");
  std::string line;
  while (std::getline(in, line)) {
    auto p = line.find('#');
    if (p != std::string::npos) line = line.substr(0, p);
    size_t a = line.find_first_not_of(" \t\r");
    if (a == std::string::npos) continue;
    v.push_back(std::atoi(line.c_str() + a));
  }
  if (v.empty()) die("empty it-file");
  return v;
}

static void append_mem(const std::string& path, const char* stage) {
  size_t free_b = 0, tot_b = 0;
  cudaMemGetInfo(&free_b, &tot_b);
  const bool fresh = !fs::exists(path);
  std::ofstream f(path, std::ios::app);
  if (fresh)
    f << "stage\tgpu_used_mib\tgpu_free_mib\tgpu_total_mib\thost_rss_mib\n";
  f << stage << "\t" << (tot_b - free_b) / 1048576.0 << "\t" << free_b / 1048576.0 << "\t"
    << tot_b / 1048576.0 << "\t" << host_rss_mib() << "\n";
}

struct Args {
  std::string base, graph_dir, query, out, gt;
  uint32_t k_build = 32;
  uint32_t k_query = 10;
  float tau_query = 0.5f;
  uint32_t max_iterations = 55;
  uint32_t batch_size = 10000;
  int gpu = 0;
  int max_batches = 0;
  std::string it_file;
  uint32_t shard_size = 0;
};

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    auto eat = [&](const char* key) -> const char* {
      if (std::strcmp(argv[i], key) != 0) return nullptr;
      if (i + 1 >= argc) die("missing value");
      return argv[++i];
    };
    if (const char* v = eat("--base")) a.base = v;
    else if (const char* v = eat("--graph-dir")) a.graph_dir = v;
    else if (const char* v = eat("--query")) a.query = v;
    else if (const char* v = eat("--out")) a.out = v;
    else if (const char* v = eat("--gt")) a.gt = v;
    else if (const char* v = eat("--k-build")) a.k_build = std::atoi(v);
    else if (const char* v = eat("--k-query")) a.k_query = std::atoi(v);
    else if (const char* v = eat("--tau-query")) a.tau_query = std::atof(v);
    else if (const char* v = eat("--max-iterations")) a.max_iterations = std::atoi(v);
    else if (const char* v = eat("--batch-size")) a.batch_size = std::atoi(v);
    else if (const char* v = eat("--gpu")) a.gpu = std::atoi(v);
    else if (const char* v = eat("--max-batches")) a.max_batches = std::atoi(v);
    else if (const char* v = eat("--it-file")) a.it_file = v;
    else if (const char* v = eat("--shard-size")) a.shard_size = std::atoi(v);
    else {
      std::fprintf(stderr, "unknown arg: %s\n", argv[i]);
      std::exit(1);
    }
  }
  if (a.base.empty() || a.graph_dir.empty() || a.query.empty() || a.out.empty())
    die("required: --base --graph-dir --query --out");
  return a;
}

struct MmapFbin {
  int fd = -1;
  void* map = MAP_FAILED;
  size_t bytes = 0;
  int64_t n = 0;
  int64_t d = 0;
  float* data = nullptr;
  MmapFbin() = default;
  MmapFbin(const MmapFbin&) = delete;
  MmapFbin& operator=(const MmapFbin&) = delete;
  MmapFbin(MmapFbin&& o) noexcept { *this = std::move(o); }
  MmapFbin& operator=(MmapFbin&& o) noexcept {
    if (this == &o) return *this;
    if (map != MAP_FAILED) munmap(map, bytes);
    if (fd >= 0) close(fd);
    fd = o.fd;
    map = o.map;
    bytes = o.bytes;
    n = o.n;
    d = o.d;
    data = o.data;
    o.fd = -1;
    o.map = MAP_FAILED;
    o.bytes = 0;
    o.data = nullptr;
    return *this;
  }
  ~MmapFbin() {
    if (map != MAP_FAILED) munmap(map, bytes);
    if (fd >= 0) close(fd);
  }
};

static bool ends_with(const std::string& s, const char* suf) {
  const size_t n = std::strlen(suf);
  return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

static MmapFbin map_fbin(const std::string& path) {
  MmapFbin m;
  m.fd = open(path.c_str(), O_RDONLY);
  if (m.fd < 0) die("cannot open base fbin");
  struct stat st;
  if (fstat(m.fd, &st) != 0) die("fstat base fbin");
  m.bytes = static_cast<size_t>(st.st_size);
  m.map = mmap(nullptr, m.bytes, PROT_READ, MAP_PRIVATE, m.fd, 0);
  if (m.map == MAP_FAILED) die("mmap base fbin");
  if (m.bytes < 8) die("fbin too small");
  int32_t header[2];
  std::memcpy(header, m.map, sizeof(header));
  m.n = header[0];
  m.d = header[1];
  const size_t need = 8ull + static_cast<size_t>(m.n) * static_cast<size_t>(m.d) * sizeof(float);
  if (m.n <= 0 || m.d <= 0 || need != m.bytes) die("fbin header mismatch");
  m.data = reinterpret_cast<float*>(static_cast<char*>(m.map) + 8);
  return m;
}

// fbin: int32 n, int32 dim, float32[n*dim]
struct FBinMeta {
  int32_t n = 0, d = 0;
};

static FBinMeta read_fbin_meta(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("cannot open query fbin");
  FBinMeta m;
  in.read(reinterpret_cast<char*>(&m.n), 4);
  in.read(reinterpret_cast<char*>(&m.d), 4);
  return m;
}

// ibin GT: int32 n, int32 k, int32[n*k]
static std::vector<int32_t> load_ibin(const std::string& path, int32_t& n, int32_t& k) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("cannot open gt ibin");
  in.read(reinterpret_cast<char*>(&n), 4);
  in.read(reinterpret_cast<char*>(&k), 4);
  std::vector<int32_t> v((size_t)n * k);
  in.read(reinterpret_cast<char*>(v.data()), (size_t)n * k * 4);
  return v;
}

// Multi-shard GGNN stores K * num_shards columns per query, sorted so the
// nearest K sit at the front of each row. Single-shard results are packed.
static void copy_topk_ids(int32_t* dst, const int32_t* src, uint32_t src_cols, uint32_t nq,
                          uint32_t k) {
  if (src_cols < k) die("result width < k");
  if (src_cols == k) {
    check_cuda(cudaMemcpy(dst, src, (size_t)nq * k * sizeof(int32_t), cudaMemcpyDeviceToHost),
               "D2H");
  } else {
    check_cuda(cudaMemcpy2D(dst, k * sizeof(int32_t), src, (size_t)src_cols * sizeof(int32_t),
                            k * sizeof(int32_t), nq, cudaMemcpyDeviceToHost),
               "D2H");
  }
}

static float recall_at_k(const int32_t* pred, const int32_t* gt, int nq, int k) {
  double hits = 0;
  for (int i = 0; i < nq; ++i) {
    std::unordered_set<int32_t> gset(gt + i * k, gt + (i + 1) * k);
    for (int j = 0; j < k; ++j) hits += gset.count(pred[i * k + j]) ? 1.0 : 0.0;
  }
  return static_cast<float>(hits / (double(nq) * k));
}

static double mean_of(const std::vector<double>& v) {
  double s = 0;
  for (double x : v) s += x;
  return v.empty() ? 0.0 : s / v.size();
}

// Sample standard deviation (n - 1), 0 for a single value
static double sd_of(const std::vector<double>& v) {
  if (v.size() < 2) return 0.0;
  const double m = mean_of(v);
  double s = 0;
  for (double x : v) s += (x - m) * (x - m);
  return std::sqrt(s / (v.size() - 1));
}

int main(int argc, char** argv) {
  Args args = parse(argc, argv);
  fs::create_directories(args.out);
  check_cuda(cudaSetDevice(args.gpu), "cudaSetDevice");

  using KeyT = int32_t;
  using ValueT = float;
  using GGNN = ggnn::GGNN<KeyT, ValueT>;
  using DatasetF = ggnn::Dataset<float>;
  using DatasetK = ggnn::Dataset<KeyT>;

  const auto t_load0 = std::chrono::steady_clock::now();

  // Keep base alive for the whole run (setBaseReference).
  MmapFbin mapped_base;
  ggnn::GenericDataset base_owned;
  ggnn::Dataset<float> base_view;
  const ggnn::GenericDataset* base_ref = nullptr;
  if (ends_with(args.base, ".fbin")) {
    mapped_base = map_fbin(args.base);
    std::printf("[ggnn] mapped fbin n=%ld dim=%ld\n", static_cast<long>(mapped_base.n),
                static_cast<long>(mapped_base.d));
    base_view = ggnn::Dataset<float>::referenceCPUData(mapped_base.data, mapped_base.n, mapped_base.d);
    base_ref = &base_view;
  } else {
    base_owned = ggnn::GenericDataset::load(args.base, 0, std::numeric_limits<uint32_t>::max(), true);
    base_ref = &base_owned;
  }

  GGNN ggnn{};
  ggnn.setWorkingDirectory(args.graph_dir);
  ggnn.setGPUs(std::vector<int>{args.gpu});
  ggnn.setBaseReference(*base_ref);
  if (args.shard_size > 0) {
    std::printf("[ggnn] setShardSize %u\n", args.shard_size);
    ggnn.setShardSize(args.shard_size);
  }
  // Keep results on GPU so we can time D2H ourselves.
  ggnn.setReturnResultsOnGPU(true);
  ggnn.load(args.k_build);

  FBinMeta qm = read_fbin_meta(args.query);
  if (static_cast<uint32_t>(qm.d) != base_ref->D) die("query dim != base dim");

  std::vector<int32_t> gt;
  int32_t gt_n = 0, gt_k = 0;
  bool have_gt = false;
  if (!args.gt.empty() && fs::is_regular_file(args.gt)) {
    gt = load_ibin(args.gt, gt_n, gt_k);
    if (gt_n != qm.n) die("gt nq mismatch");
    have_gt = true;
    if (gt_k > static_cast<int32_t>(args.k_query)) {
      std::vector<int32_t> slim(static_cast<size_t>(gt_n) * args.k_query);
      for (int32_t i = 0; i < gt_n; ++i)
        for (uint32_t j = 0; j < args.k_query; ++j)
          slim[static_cast<size_t>(i) * args.k_query + j] = gt[static_cast<size_t>(i) * gt_k + j];
      gt.swap(slim);
      gt_k = static_cast<int32_t>(args.k_query);
    }
    std::printf("[ggnn] GT loaded %d x %d\n", gt_n, gt_k);
  }

  // The query file must hold exactly n x dim floats after the header
  std::ifstream qin(args.query, std::ios::binary);
  qin.seekg(0, std::ios::end);
  const size_t qbytes = static_cast<size_t>(qin.tellg()) - 8;
  qin.close();
  if (qbytes != (size_t)qm.n * qm.d * sizeof(float)) die("query size mismatch");

  // Host buffer for one batch + pinned for faster H2D
  const uint32_t bs = args.batch_size;
  float* h_q = nullptr;
  KeyT* h_ids = nullptr;
  check_cuda(cudaMallocHost(&h_q, (size_t)bs * qm.d * sizeof(float)), "mallocHost q");
  check_cuda(cudaMallocHost(&h_ids, (size_t)bs * args.k_query * sizeof(KeyT)), "mallocHost ids");

  // Device query buffer (reused)
  DatasetF d_query = DatasetF::emptyOnGPU(bs, qm.d, args.gpu);

  cudaEvent_t e_h2d0, e_h2d1, e_k0, e_k1, e_d2h0, e_d2h1;
  check_cuda(cudaEventCreate(&e_h2d0), "ev");
  check_cuda(cudaEventCreate(&e_h2d1), "ev");
  check_cuda(cudaEventCreate(&e_k0), "ev");
  check_cuda(cudaEventCreate(&e_k1), "ev");
  check_cuda(cudaEventCreate(&e_d2h0), "ev");
  check_cuda(cudaEventCreate(&e_d2h1), "ev");

  const double load_ms =
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0).count();

  {
    std::ofstream f(args.out + "/load_timing.tsv");
    f << "stage\tms\nload_base_graph_init\t" << load_ms << "\n";
  }
  append_mem(args.out + "/gpu_mem_cuda.tsv", "after_load");
  {
    std::ofstream f(args.out + "/params.txt");
    f << "algo\tGGNN\n"
      << "base\t" << args.base << "\n"
      << "graph_dir\t" << args.graph_dir << "\n"
      << "query\t" << args.query << "\n"
      << "gt\t" << args.gt << "\n"
      << "k_build\t" << args.k_build << "\n"
      << "k_query\t" << args.k_query << "\n"
      << "tau_query\t" << args.tau_query << "\n"
      << "max_iterations\t" << args.max_iterations << "\n"
      << "it_file\t" << args.it_file << "\n"
      << "batch_size\t" << bs << "\n"
      << "n_query_total\t" << qm.n << "\n"
      << "dim\t" << qm.d << "\n"
      << "gpu\t" << args.gpu << "\n"
      << "return_results_on_gpu\t1\n";
  }

  auto its = load_its(args.it_file, (int)args.max_iterations);
  const bool sweep = !args.it_file.empty();
  int n_batches = sweep ? 1 : (qm.n + (int)bs - 1) / (int)bs;
  if (!sweep && args.max_batches > 0) n_batches = std::min(n_batches, args.max_batches);
  std::printf("[ggnn] %d queries, batch=%u, batches=%d sweep=%d n_it=%zu\n", qm.n, bs, n_batches,
              (int)sweep, its.size());

  std::ifstream qfile(args.query, std::ios::binary);
  qfile.seekg(8);

  if (sweep) {
    const uint32_t nq = std::min(bs, (uint32_t)qm.n);
    qfile.read(reinterpret_cast<char*>(h_q), (size_t)nq * qm.d * sizeof(float));
    if (!qfile) die("query read failed");
    DatasetF d_q = DatasetF::referenceGPUData(d_query.data(), nq, qm.d, args.gpu);
    std::ofstream sf(args.out + "/sweep.tsv");
    std::ofstream df(args.out + "/it_batches.tsv");
    sf << "it\tnq\twarmup_kernel_ms\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10"
          "\tkernel_ms_std\twall_ms_std\tn_timed\n";
    df << "it\tphase\tnq\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10\n";
    auto run_once = [&](uint32_t it, const char* phase, float* out_h2d, float* out_k, float* out_d2h,
                        double* out_wall, double* out_qps, std::string* out_rec) {
      const auto wall0 = std::chrono::steady_clock::now();
      check_cuda(cudaEventRecord(e_h2d0), "rec");
      check_cuda(cudaMemcpy(d_query.data(), h_q, (size_t)nq * qm.d * sizeof(float),
                            cudaMemcpyHostToDevice),
                 "H2D");
      check_cuda(cudaEventRecord(e_h2d1), "rec");
      check_cuda(cudaEventSynchronize(e_h2d1), "sync");
      float h2d_ms = 0;
      check_cuda(cudaEventElapsedTime(&h2d_ms, e_h2d0, e_h2d1), "elapsed");
      check_cuda(cudaEventRecord(e_k0), "rec");
      auto results = ggnn.query(d_q, args.k_query, args.tau_query, it,
                                ggnn::DistanceMeasure::Euclidean);
      check_cuda(cudaEventRecord(e_k1), "rec");
      check_cuda(cudaEventSynchronize(e_k1), "sync");
      float kernel_ms = 0;
      check_cuda(cudaEventElapsedTime(&kernel_ms, e_k0, e_k1), "elapsed");
      check_cuda(cudaEventRecord(e_d2h0), "rec");
      copy_topk_ids(h_ids, results.ids.data(), results.ids.D, nq, args.k_query);
      check_cuda(cudaEventRecord(e_d2h1), "rec");
      check_cuda(cudaEventSynchronize(e_d2h1), "sync");
      float d2h_ms = 0;
      check_cuda(cudaEventElapsedTime(&d2h_ms, e_d2h0, e_d2h1), "elapsed");
      const double wall_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
      const double qps = nq / (wall_ms / 1000.0);
      std::string rec;
      if (have_gt) {
        const int kk = std::min((int)args.k_query, gt_k);
        std::vector<int32_t> pred((size_t)nq * kk);
        for (uint32_t i = 0; i < nq; ++i)
          for (int j = 0; j < kk; ++j) pred[i * kk + j] = h_ids[i * args.k_query + j];
        float r = recall_at_k(pred.data(), gt.data(), (int)nq, kk);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6f", r);
        rec = buf;
      }
      df << it << "\t" << phase << "\t" << nq << "\t" << h2d_ms << "\t" << kernel_ms << "\t" << d2h_ms
         << "\t" << wall_ms << "\t" << qps << "\t" << rec << "\n";
      df.flush();
      *out_h2d = h2d_ms;
      *out_k = kernel_ms;
      *out_d2h = d2h_ms;
      *out_wall = wall_ms;
      *out_qps = qps;
      *out_rec = rec;
    };
    for (int it : its) {
      float wh2d, wk, wd2h;
      double ww, wq;
      std::string wrec;
      run_once((uint32_t)it, "warmup", &wh2d, &wk, &wd2h, &ww, &wq, &wrec);
      std::vector<double> h2d_v, kern_v, d2h_v, wall_v, rec_v;
      for (int r = 0; r < kTimedRuns; ++r) {
        float th2d, tk, td2h;
        double tw, tq;
        std::string trec;
        const std::string phase = "timed" + std::to_string(r + 1);
        run_once((uint32_t)it, phase.c_str(), &th2d, &tk, &td2h, &tw, &tq, &trec);
        h2d_v.push_back(th2d);
        kern_v.push_back(tk);
        d2h_v.push_back(td2h);
        wall_v.push_back(tw);
        if (!trec.empty()) rec_v.push_back(std::atof(trec.c_str()));
      }
      const double h2d = mean_of(h2d_v), kern = mean_of(kern_v), d2h = mean_of(d2h_v), wall = mean_of(wall_v);
      const double qps = nq / (wall / 1000.0);
      std::string rec;
      if (!rec_v.empty()) {
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6f", mean_of(rec_v));
        rec = buf;
      }
      sf << it << "\t" << nq << "\t" << wk << "\t" << h2d << "\t" << kern << "\t" << d2h << "\t"
         << wall << "\t" << qps << "\t" << rec << "\t" << sd_of(kern_v) << "\t" << sd_of(wall_v) << "\t"
         << kTimedRuns << "\n";
      sf.flush();
      std::printf("[ggnn] it=%d nq=%u h2d=%.2f kern=%.2f d2h=%.2f qps=%.0f recall=%s (mean of %d)\n", it, nq,
                  h2d, kern, d2h, qps, rec.c_str(), kTimedRuns);
    }
    append_mem(args.out + "/gpu_mem_cuda.tsv", "after_last_it");
    cudaFreeHost(h_q);
    cudaFreeHost(h_ids);
    std::printf("[ggnn] sweep done -> %s\n", args.out.c_str());
    return 0;
  }

  std::ofstream bf(args.out + "/batch_timing.tsv");
  std::ofstream rf(args.out + "/recall.tsv");
  bf << "batch_id\tnq\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10\n";
  rf << "batch_id\trecall@10\n";

  for (int bi = 0; bi < n_batches; ++bi) {
    const int32_t s = bi * (int)bs;
    const int32_t e = std::min(s + (int)bs, qm.n);
    const uint32_t nq = static_cast<uint32_t>(e - s);

    qfile.read(reinterpret_cast<char*>(h_q), (size_t)nq * qm.d * sizeof(float));
    if (!qfile) die("query read failed");

    // View first nq rows of the resident GPU query buffer
    DatasetF d_q = DatasetF::referenceGPUData(d_query.data(), nq, qm.d, args.gpu);

    const auto wall0 = std::chrono::steady_clock::now();

    check_cuda(cudaEventRecord(e_h2d0), "rec");
    check_cuda(cudaMemcpy(d_query.data(), h_q, (size_t)nq * qm.d * sizeof(float),
                          cudaMemcpyHostToDevice),
               "H2D");
    check_cuda(cudaEventRecord(e_h2d1), "rec");
    check_cuda(cudaEventSynchronize(e_h2d1), "sync");
    float h2d_ms = 0;
    check_cuda(cudaEventElapsedTime(&h2d_ms, e_h2d0, e_h2d1), "elapsed");

    check_cuda(cudaEventRecord(e_k0), "rec");
    auto results = ggnn.query(d_q, args.k_query, args.tau_query, args.max_iterations,
                              ggnn::DistanceMeasure::Euclidean);
    check_cuda(cudaEventRecord(e_k1), "rec");
    check_cuda(cudaEventSynchronize(e_k1), "sync");
    float kernel_ms = 0;
    check_cuda(cudaEventElapsedTime(&kernel_ms, e_k0, e_k1), "elapsed");

    check_cuda(cudaEventRecord(e_d2h0), "rec");
    copy_topk_ids(h_ids, results.ids.data(), results.ids.D, nq, args.k_query);
    check_cuda(cudaEventRecord(e_d2h1), "rec");
    check_cuda(cudaEventSynchronize(e_d2h1), "sync");
    float d2h_ms = 0;
    check_cuda(cudaEventElapsedTime(&d2h_ms, e_d2h0, e_d2h1), "elapsed");

    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
    const double qps = nq / (wall_ms / 1000.0);

    std::string rec;
    if (have_gt) {
      const int kk = std::min((int)args.k_query, gt_k);
      // pred may be int32 KeyT
      std::vector<int32_t> pred((size_t)nq * kk);
      for (uint32_t i = 0; i < nq; ++i)
        for (int j = 0; j < kk; ++j) pred[i * kk + j] = h_ids[i * args.k_query + j];
      float r = recall_at_k(pred.data(), gt.data() + (size_t)s * gt_k, (int)nq, kk);
      char buf[64];
      std::snprintf(buf, sizeof(buf), "%.6f", r);
      rec = buf;
      rf << bi << "\t" << rec << "\n";
      rf.flush();
    }

    bf << bi << "\t" << nq << "\t" << h2d_ms << "\t" << kernel_ms << "\t" << d2h_ms << "\t"
       << wall_ms << "\t" << qps << "\t" << rec << "\n";
    bf.flush();
    if (bi == 0) append_mem(args.out + "/gpu_mem_cuda.tsv", "after_first_batch");
    if (bi == n_batches - 1) append_mem(args.out + "/gpu_mem_cuda.tsv", "after_last_batch");

    if (bi % 10 == 0 || bi == n_batches - 1) {
      std::printf("[ggnn] batch %d/%d nq=%u h2d=%.2f kern=%.2f d2h=%.2f wall=%.2f qps=%.0f recall=%s\n",
                  bi, n_batches - 1, nq, h2d_ms, kernel_ms, d2h_ms, wall_ms, qps, rec.c_str());
    }
  }

  cudaFreeHost(h_q);
  cudaFreeHost(h_ids);
  std::printf("[ggnn] done -> %s\n", args.out.c_str());
  return 0;
}
