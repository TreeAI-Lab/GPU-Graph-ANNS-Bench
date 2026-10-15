// GPU-resident GANNS (NSW) batch search timing.
// Loads base.fvecs + graph.nsw once onto the GPU; per batch only H2D queries + SearchK + D2H.
// With GANNS_FP16 the base and query vectors are stored as __half on the GPU (FP32 accumulation).
//
// Built by src/fp16/build.sh (PLACE_HOLDER_DIM = dataset dimension, USE_DIST_L2_).
//
// usage:
//   ganns_<dataset>_<fp32|fp16> \
//     --base base.fvecs --graph graph.nsw --query query.fbin --out OUT \
//     --k 10 --M 64 --n-e 55 --batch-size 10000 --gpu 0 [--gt gt_k10.ibin] [--it-file its.txt]
//   The ground truth must have exactly k columns. With --it-file only the first batch is searched; per iteration
//   count one untimed warm-up run, then three timed runs (kTimedRuns). OUT/sweep.tsv holds the means over the
//   timed runs (qps = nq / mean wall time) and the sample standard deviations; OUT/it_batches.tsv holds every run.

#include <iostream>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unordered_set>
#include <vector>

#include <cuda_runtime.h>

#include "data.h"
#include "kernels.h"
#include "type.h"
#include <thread>

// Timed runs per setting of a sweep, after one untimed warm-up run
constexpr int kTimedRuns = 3;

// IT_SLEEP_MS (milliseconds): sleep before each iteration count, so that every measurement starts from an
// idle GPU and a sustained load does not run into the board power limit. Unset or 0: no sleep.
static void it_sleep() {
  static const char* s = std::getenv("IT_SLEEP_MS");
  if (s && std::atoi(s) > 0) std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(s)));
}

static void die(const char* m) {
  std::fprintf(stderr, "ERROR: %s\n", m);
  std::exit(1);
}
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
  int k = 10, M = 64, n_e = 55, batch_size = 10000, gpu = 0, max_batches = 0;
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
    else if (const char* v = eat("--k")) a.k = std::atoi(v);
    else if (const char* v = eat("--M")) a.M = std::atoi(v);
    else if (const char* v = eat("--n-e")) a.n_e = std::atoi(v);
    else if (const char* v = eat("--batch-size")) a.batch_size = std::atoi(v);
    else if (const char* v = eat("--gpu")) a.gpu = std::atoi(v);
    else if (const char* v = eat("--max-batches")) a.max_batches = std::atoi(v);
    else if (const char* v = eat("--it-file")) a.it_file = v;
    else {
      std::fprintf(stderr, "unknown: %s\n", argv[i]);
      std::exit(1);
    }
  }
  if (a.base.empty() || a.graph.empty() || a.query.empty() || a.out.empty())
    die("required: --base --graph --query --out");
  return a;
}

struct FBinMeta {
  int32_t n = 0, d = 0;
};
static FBinMeta read_fbin_meta(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("open query");
  FBinMeta m;
  in.read((char*)&m.n, 4);
  in.read((char*)&m.d, 4);
  return m;
}

static std::vector<int32_t> load_ibin(const std::string& path, int32_t& n, int32_t& k) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("open gt");
  in.read((char*)&n, 4);
  in.read((char*)&k, 4);
  std::vector<int32_t> v((size_t)n * k);
  in.read((char*)v.data(), (size_t)n * k * 4);
  return v;
}

static float recall_at_k(const int* pred, const int32_t* gt, int nq, int k) {
  double hits = 0;
  for (int i = 0; i < nq; ++i) {
    std::unordered_set<int32_t> gset(gt + i * k, gt + (i + 1) * k);
    for (int j = 0; j < k; ++j) hits += gset.count(pred[i * k + j]) ? 1.0 : 0.0;
  }
  return float(hits / (double(nq) * k));
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

// Load NSW graph: int n, int s_bits, int graph[n << s_bits]
struct NswGraph {
  int n = 0, s_bits = 0;
  int* graph = nullptr;  // pinned host
};

static NswGraph load_nsw(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  if (!in) die("open graph.nsw");
  NswGraph g;
  in.read((char*)&g.n, 4);
  in.read((char*)&g.s_bits, 4);
  const size_t nedge = (size_t)g.n << g.s_bits;
  check_cuda(cudaMallocHost(&g.graph, nedge * sizeof(int)), "pin graph");
  in.read((char*)g.graph, nedge * sizeof(int));
  return g;
}

int main(int argc, char** argv) {
  Args args = parse(argc, argv);
  check_cuda(cudaSetDevice(args.gpu), "setDevice");
  {
    std::string cmd = "mkdir -p \"" + args.out + "\"";
    std::system(cmd.c_str());
  }

  int M = (int)std::pow(2.0, std::ceil(std::log((double)args.M) / std::log(2.0)));
  if (args.it_file.empty() && args.n_e > M) die("n_e must be <= M (after power-of-2 round)");

  const auto t_load0 = std::chrono::steady_clock::now();

  Data data(args.base);
  NswGraph nsw = load_nsw(args.graph);
  if (nsw.n != data.num()) die("graph n != base n");

  FBinMeta qm = read_fbin_meta(args.query);
  if (qm.d != data.dim()) die("query dim != base dim");

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

  // Upload base + graph once
#ifdef GANNS_FP16
  std::vector<vec_t> base_vec((size_t)data.num() * data.dim());
  {
    const float* src = data.get_vector(0);
    for (size_t i = 0; i < base_vec.size(); ++i) base_vec[i] = __float2half_rn(src[i]);
  }
  vec_t* d_base = copy_to_device<vec_t>(base_vec.data(), base_vec.size());
#else
  float* d_base = copy_to_device<float>(data.get_vector(0), (size_t)data.num() * data.dim());
#endif
  int* d_graph = copy_to_device<int>(nsw.graph, (size_t)nsw.n << nsw.s_bits);

  const int bs = args.batch_size;
  float* h_q = nullptr;
  int* h_ann = nullptr;
  check_cuda(cudaMallocHost(&h_q, (size_t)bs * qm.d * sizeof(float)), "pin q");
  check_cuda(cudaMallocHost(&h_ann, (size_t)bs * args.k * sizeof(int)), "pin ann");

  vec_t* d_q = nullptr;
  int* d_ann = nullptr;
  check_cuda(cudaMalloc(&d_q, (size_t)bs * qm.d * sizeof(vec_t)), "d_q");
#ifdef GANNS_FP16
  vec_t* h_qv = nullptr;  // converted queries; the conversion runs right after reading, outside the timed region
  check_cuda(cudaMallocHost(&h_qv, (size_t)bs * qm.d * sizeof(vec_t)), "pin qv");
  auto convert_q = [&](int nq) {
    for (size_t i = 0; i < (size_t)nq * qm.d; ++i) h_qv[i] = __float2half_rn(h_q[i]);
  };
#define H_QUP h_qv
#else
  auto convert_q = [&](int) {};
#define H_QUP h_q
#endif
  check_cuda(cudaMalloc(&d_ann, (size_t)bs * args.k * sizeof(int)), "d_ann");

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
    f << "algo\tGANNS\n"
#ifdef GANNS_FP16
      << "vector_precision\tfp16 (base and query stored as __half, fp32 accumulate)\n"
#else
      << "vector_precision\tfp32\n"
#endif
      << "base\t" << args.base << "\n"
      << "graph\t" << args.graph << "\n"
      << "query\t" << args.query << "\n"
      << "gt\t" << args.gt << "\n"
      << "k\t" << args.k << "\n"
      << "M\t" << M << "\n"
      << "n_e\t" << args.n_e << "\n"
      << "batch_size\t" << bs << "\n"
      << "s_bits\t" << nsw.s_bits << "\n"
      << "n_base\t" << nsw.n << "\n"
      << "dim\t" << qm.d << "\n"
      << "gpu\t" << args.gpu << "\n"
      << "note\teffort=n_e (exploration limit); M=candidate list (power of 2)\n";
  }

  auto its = load_its(args.it_file, args.n_e);
  const bool sweep = !args.it_file.empty();
  int n_batches = sweep ? 1 : (qm.n + bs - 1) / bs;
  if (!sweep && args.max_batches > 0) n_batches = std::min(n_batches, args.max_batches);
  std::printf("[ganns] %d queries, batch=%d, batches=%d sweep=%d n_it=%zu\n", qm.n, bs, n_batches,
              (int)sweep, its.size());

  std::ifstream qfile(args.query, std::ios::binary);
  qfile.seekg(8);

  if (sweep) {
    const int nq = std::min(bs, (int)qm.n);
    qfile.read((char*)h_q, (size_t)nq * qm.d * sizeof(float));
    if (!qfile) die("query read");
    convert_q(nq);
    std::ofstream sf(args.out + "/sweep.tsv");
    std::ofstream df(args.out + "/it_batches.tsv");
    sf << "it\tM\tnq\twarmup_kernel_ms\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10"
          "\tkernel_ms_std\twall_ms_std\tn_timed\n";
    df << "it\tphase\tM\tnq\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10\n";
    const int block_dim = 256;
    const int grid_dim = (nq + block_dim / WS - 1) / (block_dim / WS);
    check_cuda(cudaFuncSetAttribute(SearchK, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                    96 * 1024),
               "set_dyn_shmem");
    auto run_once = [&](int n_e, int Mm, const char* phase, float* oh, float* ok, float* od,
                        double* ow, double* oq, std::string* orec) {
      const int shmem =
          ((1 << nsw.s_bits) + Mm) * (256 / WS) * (int)(sizeof(NBR) + sizeof(int));
      const auto wall0 = std::chrono::steady_clock::now();
      check_cuda(cudaEventRecord(e_h2d0), "rec");
      check_cuda(cudaMemcpy(d_q, H_QUP, (size_t)nq * qm.d * sizeof(vec_t), cudaMemcpyHostToDevice),
                 "H2D");
      check_cuda(cudaEventRecord(e_h2d1), "rec");
      check_cuda(cudaEventSynchronize(e_h2d1), "sync");
      float h2d_ms = 0;
      check_cuda(cudaEventElapsedTime(&h2d_ms, e_h2d0, e_h2d1), "el");
      check_cuda(cudaEventRecord(e_k0), "rec");
      SearchK<<<grid_dim, block_dim, shmem>>>(d_base, d_q, d_ann, d_graph, nsw.n, nq, nsw.s_bits, Mm,
                                              args.k, n_e);
      check_cuda(cudaGetLastError(), "SearchK launch");
      check_cuda(cudaEventRecord(e_k1), "rec");
      check_cuda(cudaEventSynchronize(e_k1), "sync");
      float kernel_ms = 0;
      check_cuda(cudaEventElapsedTime(&kernel_ms, e_k0, e_k1), "el");
      check_cuda(cudaEventRecord(e_d2h0), "rec");
      check_cuda(cudaMemcpy(h_ann, d_ann, (size_t)nq * args.k * sizeof(int), cudaMemcpyDeviceToHost),
                 "D2H");
      check_cuda(cudaEventRecord(e_d2h1), "rec");
      check_cuda(cudaEventSynchronize(e_d2h1), "sync");
      float d2h_ms = 0;
      check_cuda(cudaEventElapsedTime(&d2h_ms, e_d2h0, e_d2h1), "el");
      const double wall_ms =
          std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
      const double qps = nq / (wall_ms / 1000.0);
      std::string rec;
      if (have_gt) {
        const int kk = std::min(args.k, gt_k);
        float r = recall_at_k(h_ann, gt.data(), nq, kk);
        char buf[64];
        std::snprintf(buf, sizeof(buf), "%.6f", r);
        rec = buf;
      }
      df << n_e << "\t" << phase << "\t" << Mm << "\t" << nq << "\t" << h2d_ms << "\t" << kernel_ms
         << "\t" << d2h_ms << "\t" << wall_ms << "\t" << qps << "\t" << rec << "\n";
      df.flush();
      *oh = h2d_ms;
      *ok = kernel_ms;
      *od = d2h_ms;
      *ow = wall_ms;
      *oq = qps;
      *orec = rec;
    };
    // Candidate list must cover the adjacency slots (1<<s_bits). A shorter list
    // walks off the shared buffer when distances tie (GIST k=64 -> 128 slots).
    const int graph_slots = 1 << nsw.s_bits;
    for (int it : its) {
      it_sleep();
      int Mm = 64;
      while (Mm < it || Mm < graph_slots) Mm *= 2;
      if (Mm > M) die("M cap smaller than graph slots or it");
      float wh, wk, wd;
      double ww, wq;
      std::string wrec;
      run_once(it, Mm, "warmup", &wh, &wk, &wd, &ww, &wq, &wrec);
      std::vector<double> h2d_v, kern_v, d2h_v, wall_v, rec_v;
      for (int r = 0; r < kTimedRuns; ++r) {
        float th, tk, td;
        double tw, tq;
        std::string trec;
        const std::string phase = "timed" + std::to_string(r + 1);
        run_once(it, Mm, phase.c_str(), &th, &tk, &td, &tw, &tq, &trec);
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
      sf << it << "\t" << Mm << "\t" << nq << "\t" << wk << "\t" << h2d << "\t" << kern << "\t" << d2h
         << "\t" << wall << "\t" << qps << "\t" << rec << "\t" << sd_of(kern_v) << "\t" << sd_of(wall_v)
         << "\t" << kTimedRuns << "\n";
      sf.flush();
      std::printf("[ganns] it=%d M=%d nq=%d h2d=%.2f kern=%.2f d2h=%.2f qps=%.0f recall=%s (mean of %d)\n", it,
                  Mm, nq, h2d, kern, d2h, qps, rec.c_str(), kTimedRuns);
    }
    append_mem(args.out + "/gpu_mem_cuda.tsv", "after_last_it");
    cudaFree(d_base);
    cudaFree(d_graph);
    cudaFree(d_q);
    cudaFree(d_ann);
    cudaFreeHost(h_q);
    cudaFreeHost(h_ann);
    cudaFreeHost(nsw.graph);
    std::printf("[ganns] sweep done -> %s\n", args.out.c_str());
    return 0;
  }

  std::ofstream bf(args.out + "/batch_timing.tsv");
  std::ofstream rf(args.out + "/recall.tsv");
  bf << "batch_id\tnq\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10\n";
  rf << "batch_id\trecall@10\n";

  const int shmem = ((1 << nsw.s_bits) + M) * (256 / WS) * (int)(sizeof(NBR) + sizeof(int));

  for (int bi = 0; bi < n_batches; ++bi) {
    const int s = bi * bs;
    const int e = std::min(s + bs, (int)qm.n);
    const int nq = e - s;
    qfile.read((char*)h_q, (size_t)nq * qm.d * sizeof(float));
    if (!qfile) die("query read");
    convert_q(nq);

    const auto wall0 = std::chrono::steady_clock::now();

    check_cuda(cudaEventRecord(e_h2d0), "rec");
    check_cuda(cudaMemcpy(d_q, H_QUP, (size_t)nq * qm.d * sizeof(vec_t), cudaMemcpyHostToDevice),
               "H2D");
    check_cuda(cudaEventRecord(e_h2d1), "rec");
    check_cuda(cudaEventSynchronize(e_h2d1), "sync");
    float h2d_ms = 0;
    check_cuda(cudaEventElapsedTime(&h2d_ms, e_h2d0, e_h2d1), "el");

    const int block_dim = 256;
    const int grid_dim = (nq + block_dim / WS - 1) / (block_dim / WS);

    check_cuda(cudaEventRecord(e_k0), "rec");
    SearchK<<<grid_dim, block_dim, shmem>>>(d_base, d_q, d_ann, d_graph, nsw.n, nq, nsw.s_bits, M,
                                            args.k, args.n_e);
    check_cuda(cudaGetLastError(), "SearchK launch");
    check_cuda(cudaEventRecord(e_k1), "rec");
    check_cuda(cudaEventSynchronize(e_k1), "sync");
    float kernel_ms = 0;
    check_cuda(cudaEventElapsedTime(&kernel_ms, e_k0, e_k1), "el");

    check_cuda(cudaEventRecord(e_d2h0), "rec");
    check_cuda(cudaMemcpy(h_ann, d_ann, (size_t)nq * args.k * sizeof(int), cudaMemcpyDeviceToHost),
               "D2H");
    check_cuda(cudaEventRecord(e_d2h1), "rec");
    check_cuda(cudaEventSynchronize(e_d2h1), "sync");
    float d2h_ms = 0;
    check_cuda(cudaEventElapsedTime(&d2h_ms, e_d2h0, e_d2h1), "el");

    const double wall_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - wall0).count();
    const double qps = nq / (wall_ms / 1000.0);

    std::string rec;
    if (have_gt) {
      const int kk = std::min(args.k, gt_k);
      float r = recall_at_k(h_ann, gt.data() + (size_t)s * gt_k, nq, kk);
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
          "[ganns] batch %d/%d nq=%d h2d=%.2f kern=%.2f d2h=%.2f wall=%.2f qps=%.0f recall=%s\n", bi,
          n_batches - 1, nq, h2d_ms, kernel_ms, d2h_ms, wall_ms, qps, rec.c_str());
    }
  }

  cudaFree(d_base);
  cudaFree(d_graph);
  cudaFree(d_q);
  cudaFree(d_ann);
  cudaFreeHost(h_q);
  cudaFreeHost(h_ann);
  cudaFreeHost(nsw.graph);
  std::printf("[ganns] done -> %s\n", args.out.c_str());
  return 0;
}
