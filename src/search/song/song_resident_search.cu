// SONG search with base vectors and graph resident in GPU memory.
// Uses the SONG kernel with an iteration limit (song/warp_astar_accelerator.h). The graph is a SONG bfsg.graph
// with 2^shift slots per vertex (FIXED_DEGREE, FIXED_DEGREE_SHIFT and DIM are compile-time constants).
// Per batch: query H2D, kernel and result D2H, each timed with CUDA events. The result list has
// max_iterations - 1 entries; Recall@10 uses the first 10.
// With --it-file: only the first batch is searched; per iteration count one untimed warm-up run, then three timed
// runs (kTimedRuns). sweep.tsv holds the means over the timed runs (qps = nq / mean wall time) and the sample
// standard deviations; it_batches.tsv holds every run.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <unordered_set>

#include <cuda_runtime.h>

#include "astar_accelerator.h"
#include "warp_astar_accelerator.h"

// Timed runs per setting of a sweep, after one untimed warm-up run
constexpr int kTimedRuns = 3;

static void die(const char* m) { std::fprintf(stderr, "ERROR: %s\n", m); std::exit(1); }
static void check_cuda(cudaError_t e, const char* w) {
  if (e != cudaSuccess) {
    std::fprintf(stderr, "CUDA %s: %s\n", w, cudaGetErrorString(e));
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
  const bool fresh = !std::ifstream(path).good();
  std::ofstream f(path, std::ios::app);
  if (fresh)
    f << "stage\tgpu_used_mib\tgpu_free_mib\tgpu_total_mib\thost_rss_mib\n";
  f << stage << "\t" << (tot_b - free_b) / 1048576.0 << "\t" << free_b / 1048576.0 << "\t"
    << tot_b / 1048576.0 << "\t" << host_rss_mib() << "\n";
}

struct Args {
  std::string base, graph, query, out, gt;
  int batch_size = 10000, gpu = 0, max_batches = 0;
  int n_base = 1000000, dim = 96, shift = 6;
  int kernel_chunk = 0; // accepted but ignored: each batch is one kernel launch
  int max_iterations = 55;
  int recall_k = 10;
  std::string it_file;
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
    else if (const char* v = eat("--graph")) a.graph = v;
    else if (const char* v = eat("--query")) a.query = v;
    else if (const char* v = eat("--out")) a.out = v;
    else if (const char* v = eat("--gt")) a.gt = v;
    else if (const char* v = eat("--batch-size")) a.batch_size = std::atoi(v);
    else if (const char* v = eat("--gpu")) a.gpu = std::atoi(v);
    else if (const char* v = eat("--max-batches")) a.max_batches = std::atoi(v);
    else if (const char* v = eat("--n-base")) a.n_base = std::atoi(v);
    else if (const char* v = eat("--dim")) a.dim = std::atoi(v);
    else if (const char* v = eat("--shift")) a.shift = std::atoi(v);
    else if (const char* v = eat("--kernel-chunk")) a.kernel_chunk = std::atoi(v);
    else if (const char* v = eat("--max-iterations")) a.max_iterations = std::atoi(v);
    else if (const char* v = eat("--recall-k")) a.recall_k = std::atoi(v);
    else if (const char* v = eat("--it-file")) a.it_file = v;
    else { std::fprintf(stderr, "unknown: %s\n", argv[i]); std::exit(1); }
  }
  if (a.base.empty() || a.graph.empty() || a.query.empty() || a.out.empty())
    die("required: --base --graph --query --out");
  if (a.max_iterations < 2) die("max_iterations >= 2");
  return a;
}

struct FBinMeta { int32_t n = 0, d = 0; };
static FBinMeta read_fbin_meta(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("open fbin");
  FBinMeta m; in.read((char*)&m.n, 4); in.read((char*)&m.d, 4); return m;
}

static std::vector<value_t> load_base(const std::string& path, int n, int dim) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("open base");
  in.seekg(0, std::ios::end);
  const size_t sz = (size_t)in.tellg();
  in.seekg(0);
  const size_t expect_raw = (size_t)n * dim * sizeof(value_t);
  std::vector<value_t> v((size_t)n * dim);
  if (sz == expect_raw + 8) {
    int32_t hn=0, hd=0; in.read((char*)&hn,4); in.read((char*)&hd,4);
    if (hn != n || hd != dim) die("fbin header mismatch");
    in.read((char*)v.data(), expect_raw);
  } else if (sz == expect_raw) {
    in.read((char*)v.data(), expect_raw);
  } else {
    die("bad base size");
  }
  return v;
}

static std::vector<idx_t> load_graph(const std::string& path, int n, int shift) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("open graph");
  const size_t nslots = (size_t)n << shift;
  std::vector<idx_t> g(nslots);
  in.read((char*)g.data(), nslots * sizeof(idx_t));
  if (!in) die("graph read incomplete");
  return g;
}

static std::vector<int32_t> load_ibin(const std::string& path, int32_t& n, int32_t& k) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("open gt");
  in.read((char*)&n, 4); in.read((char*)&k, 4);
  std::vector<int32_t> v((size_t)n * k);
  in.read((char*)v.data(), (size_t)n * k * 4);
  return v;
}

// pred layout: nq * result_width; use first recall_k columns
static float recall_at_k(const idx_t* pred, int result_width, const int32_t* gt,
                         int nq, int recall_k, int gt_k) {
  const int kk = std::min(recall_k, gt_k);
  double hits = 0;
  for (int i = 0; i < nq; ++i) {
    std::unordered_set<int32_t> gset(gt + i * gt_k, gt + i * gt_k + kk);
    for (int j = 0; j < kk; ++j)
      hits += gset.count((int32_t)pred[i * result_width + j]) ? 1.0 : 0.0;
  }
  return float(hits / (double(nq) * kk));
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
  check_cuda(cudaSetDevice(args.gpu), "setDevice");
  // One launch per batch: every query allocates its search structures on the device heap
  // (about 1 GB for 100,000 queries at it=240), so the heap limit is raised to 16 GB.
  check_cuda(cudaDeviceSetLimit(cudaLimitMallocHeapSize, (size_t)16 * 1024 * 1024 * 1024), "heap");
  std::system(("mkdir -p \"" + args.out + "\"").c_str());

  std::printf("[song] loading base+graph...\n"); fflush(stdout);
  const auto t_load0 = std::chrono::steady_clock::now();
  auto h_data = load_base(args.base, args.n_base, args.dim);
  auto h_graph = load_graph(args.graph, args.n_base, args.shift);

  FBinMeta qm = read_fbin_meta(args.query);
  if (qm.d != args.dim) die("query dim mismatch");

  std::vector<int32_t> gt;
  int32_t gt_n = 0, gt_k = 0;
  bool have_gt = false;
  if (!args.gt.empty()) {
    std::ifstream t(args.gt);
    if (t.good()) {
      t.close();
      gt = load_ibin(args.gt, gt_n, gt_k);
      if (gt_n != qm.n) die("gt nq mismatch");
      have_gt = true;
    }
  }

  const int bs = args.batch_size;
  const bool sweep = !args.it_file.empty();
  auto its = load_its(args.it_file, args.max_iterations);
  const int alloc_it = sweep ? 500 : args.max_iterations;
  const int result_width_alloc = alloc_it - 1;
  SongResidentSearch search;
  search.init(h_data.data(), h_graph.data(), args.shift, args.n_base, args.dim, bs, alloc_it);
  std::printf("[song] resident FD=%d SHIFT=%d alloc_it=%d sweep=%d n_it=%zu\n", FIXED_DEGREE,
              args.shift, alloc_it, (int)sweep, its.size());

  value_t* h_q = nullptr;
  idx_t* h_res = nullptr;
  check_cuda(cudaMallocHost(&h_q, (size_t)bs * args.dim * sizeof(value_t)), "pin q");
  check_cuda(cudaMallocHost(&h_res, (size_t)bs * result_width_alloc * sizeof(idx_t)), "pin res");

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
    f << "stage\tms\nload_base_graph_h2d\t" << load_ms << "\n";
  }
  append_mem(args.out + "/gpu_mem_cuda.tsv", "after_load");
  {
    std::ofstream f(args.out + "/params.txt");
    f << "algo\tSONG\n"
      << "source\tSONG warp A* kernel with max_iterations\n"
      << "FIXED_DEGREE\t" << FIXED_DEGREE << "\n"
      << "FIXED_DEGREE_SHIFT\t" << FIXED_DEGREE_SHIFT << "\n"
      << "degree\t" << (FIXED_DEGREE / 2) << "\n"
      << "max_iterations\t" << args.max_iterations << "\n"
      << "it_file\t" << args.it_file << "\n"
      << "alloc_it\t" << alloc_it << "\n"
      << "result_width\t" << result_width_alloc << "\n"
      << "recall_k\t" << args.recall_k << "\n"
      << "base\t" << args.base << "\n"
      << "graph\t" << args.graph << "\n"
      << "query\t" << args.query << "\n"
      << "batch_size\t" << bs << "\n"
      << "kernel_chunk\t" << args.kernel_chunk << "\n"
      << "gpu\t" << args.gpu << "\n";
  }

  int n_batches = sweep ? 1 : (qm.n + bs - 1) / bs;
  if (!sweep && args.max_batches > 0) n_batches = std::min(n_batches, args.max_batches);
  std::printf("[song] %d queries, batch=%d, batches=%d\n", qm.n, bs, n_batches);

  std::ifstream qfile(args.query, std::ios::binary);
  qfile.seekg(8);

  if (sweep) {
    const int nq = std::min(bs, (int)qm.n);
    if (nq % N_MULTIQUERY != 0) die("nq not divisible by N_MULTIQUERY");
    qfile.read((char*)h_q, (size_t)nq * args.dim * sizeof(value_t));
    if (!qfile) die("query read");
    std::ofstream sf(args.out + "/sweep.tsv");
    std::ofstream df(args.out + "/it_batches.tsv");
    sf << "it\tnq\twarmup_kernel_ms\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10"
          "\tkernel_ms_std\twall_ms_std\tn_timed\n";
    df << "it\tphase\tnq\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10\n";
    auto run_once = [&](int it, const char* phase, float* oh, float* ok, float* od, double* ow,
                        double* oq, std::string* orec) {
      search.set_iterations(it);
      const auto wall0 = std::chrono::steady_clock::now();
      float h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
      search.search_batch(h_q, nq, h_res, e_h2d0, e_h2d1, e_k0, e_k1, e_d2h0, e_d2h1, &h2d_ms,
                          &kernel_ms, &d2h_ms, args.kernel_chunk);
      const double wall_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
      const double qps = nq / (wall_ms / 1000.0);
      std::string rec;
      if (have_gt) {
        float r = recall_at_k(h_res, it - 1, gt.data(), nq, args.recall_k, gt_k);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6f", r);
        rec = buf;
      }
      df << it << "\t" << phase << "\t" << nq << "\t" << h2d_ms << "\t" << kernel_ms << "\t" << d2h_ms
         << "\t" << wall_ms << "\t" << qps << "\t" << rec << "\n";
      df.flush();
      *oh = h2d_ms;
      *ok = kernel_ms;
      *od = d2h_ms;
      *ow = wall_ms;
      *oq = qps;
      *orec = rec;
    };
    for (int it : its) {
      float wh, wk, wd;
      double ww, wq;
      std::string wrec;
      run_once(it, "warmup", &wh, &wk, &wd, &ww, &wq, &wrec);
      std::vector<double> h2d_v, kern_v, d2h_v, wall_v, rec_v;
      for (int r = 0; r < kTimedRuns; ++r) {
        float th, tk, td;
        double tw, tq;
        std::string trec;
        const std::string phase = "timed" + std::to_string(r + 1);
        run_once(it, phase.c_str(), &th, &tk, &td, &tw, &tq, &trec);
        h2d_v.push_back(th);
        kern_v.push_back(tk);
        d2h_v.push_back(td);
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
      std::printf("[song] it=%d nq=%d h2d=%.2f kern=%.2f d2h=%.2f qps=%.0f recall=%s (mean of %d)\n", it, nq,
                  h2d, kern, d2h, qps, rec.c_str(), kTimedRuns);
    }
    append_mem(args.out + "/gpu_mem_cuda.tsv", "after_last_it");
    search.cleanup();
    cudaFreeHost(h_q);
    cudaFreeHost(h_res);
    std::printf("[song] sweep done -> %s\n", args.out.c_str());
    return 0;
  }

  std::ofstream bf(args.out + "/batch_timing.tsv");
  std::ofstream rf(args.out + "/recall.tsv");
  bf << "batch_id\tnq\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10\n";
  rf << "batch_id\trecall@10\n";

  for (int bi = 0; bi < n_batches; ++bi) {
    const int s = bi * bs;
    const int e = std::min(s + bs, (int)qm.n);
    const int nq = e - s;
    if (nq % N_MULTIQUERY != 0) die("nq not divisible by N_MULTIQUERY");

    qfile.read((char*)h_q, (size_t)nq * args.dim * sizeof(value_t));
    if (!qfile) die("query read");

    const auto wall0 = std::chrono::steady_clock::now();
    float h2d_ms = 0, kernel_ms = 0, d2h_ms = 0;
    search.search_batch(h_q, nq, h_res, e_h2d0, e_h2d1, e_k0, e_k1, e_d2h0, e_d2h1, &h2d_ms,
                        &kernel_ms, &d2h_ms, args.kernel_chunk);
    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
    const double qps = nq / (wall_ms / 1000.0);

    std::string rec;
    if (have_gt) {
      float r = recall_at_k(h_res, search.result_width, gt.data() + (size_t)s * gt_k, nq, args.recall_k, gt_k);
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
      std::printf(
          "[song] batch %d/%d nq=%d h2d=%.2f kern=%.2f d2h=%.2f wall=%.2f qps=%.0f recall=%s\n", bi,
          n_batches - 1, nq, h2d_ms, kernel_ms, d2h_ms, wall_ms, qps, rec.c_str());
    }
  }

  search.cleanup();
  cudaFreeHost(h_q);
  cudaFreeHost(h_res);
  std::printf("[song] done -> %s\n", args.out.c_str());
  return 0;
}
