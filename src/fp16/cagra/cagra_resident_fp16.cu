// CAGRA resident search with float or half vectors on the same graph.
// The Python API of cuVS 25.02 cannot load or save a half index, so this driver uses the C++ API:
//   1. Deserialize the float index that the CAGRA QPS sweep searches (the cached index with its dataset,
//      index_with_dataset.cagra).
//   2. fp16: convert the float dataset of that index to half on the GPU, create an index<half>, attach the
//      half data with update_dataset, and let update_graph reference the same graph as the float index
//      (no copy, no rebuild).
//   3. Sweep protocol as in the Python CAGRA sweep: only the first query batch; per iteration count one untimed
//      warm-up search, then three timed searches (kTimedRuns); itopk = max(32, ceil32(it)), search_width 1, other
//      search parameters at their defaults. sweep.tsv has the columns of the Python sweep without gpu_used_mib:
//      the means over the timed searches (qps = nq / mean wall time), then the sample standard deviations and the
//      number of timed searches; it_batches.tsv holds every search.
// usage: cagra_fp16 --index <float index> --query <fbin> --gt <ibin> --out <dir>
//        --batch-size 10000 --it-file <file> [--k 10] [--precision fp32|fp16]
#include <cuvs/neighbors/cagra.hpp>
#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
#include <thread>

// Timed runs per setting of a sweep, after one untimed warm-up run
constexpr int kTimedRuns = 3;

// IT_SLEEP_MS (milliseconds): sleep before each iteration count, so that every measurement starts from an
// idle GPU and a sustained load does not run into the board power limit. Unset or 0: no sleep.
static void it_sleep() {
  static const char* s = std::getenv("IT_SLEEP_MS");
  if (s && std::atoi(s) > 0) std::this_thread::sleep_for(std::chrono::milliseconds(std::atoi(s)));
}

namespace cagra = cuvs::neighbors::cagra;

static void die(const std::string& m) { std::fprintf(stderr, "ERROR: %s\n", m.c_str()); std::exit(1); }
static void ck(cudaError_t e, const char* w) { if (e != cudaSuccess) die(std::string(w) + ": " + cudaGetErrorString(e)); }

struct Args {
  std::string index, query, gt, out, it_file, precision = "fp16";
  int batch_size = 10000, k = 10, search_width = 1;
};

static Args parse(int argc, char** argv) {
  Args a;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    if (i + 1 >= argc) die("missing value for " + key);
    std::string v = argv[++i];
    if (key == "--index") a.index = v;
    else if (key == "--query") a.query = v;
    else if (key == "--gt") a.gt = v;
    else if (key == "--out") a.out = v;
    else if (key == "--it-file") a.it_file = v;
    else if (key == "--precision") a.precision = v;
    else if (key == "--batch-size") a.batch_size = std::atoi(v.c_str());
    else if (key == "--k") a.k = std::atoi(v.c_str());
    else if (key == "--search-width") a.search_width = std::atoi(v.c_str());
    else die("unknown argument " + key);
  }
  if (a.index.empty() || a.query.empty() || a.out.empty() || a.it_file.empty()) die("required: --index --query --out --it-file");
  if (a.precision != "fp32" && a.precision != "fp16") die("--precision fp32|fp16");
  return a;
}

static std::vector<int> load_its(const std::string& path) {
  std::ifstream in(path);
  if (!in) die("open it-file");
  std::vector<int> v;
  std::string line;
  while (std::getline(in, line)) {
    line = line.substr(0, line.find('#'));
    if (line.find_first_not_of(" \t\r") != std::string::npos) v.push_back(std::atoi(line.c_str()));
  }
  if (v.empty()) die("empty it-file");
  return v;
}

static int itopk_for_it(int it, int k) { int need = std::max({it, k, 32}); return (need + 31) / 32 * 32; }

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

__global__ void float_to_half(const float* src, int64_t stride, half* dst, int64_t n, int64_t dim) {
  int64_t i = blockIdx.x * (int64_t)blockDim.x + threadIdx.x;
  if (i < n * dim) dst[i] = __float2half_rn(src[(i / dim) * stride + i % dim]);
}

template <typename T>
static void run(const Args& a, raft::device_resources& res, const cagra::index<T, uint32_t>& idx,
                const std::vector<float>& q32, int nq, int dim, const std::vector<int32_t>& gt, int gt_k) {
  cudaStream_t s = raft::resource::get_cuda_stream(res);
  std::vector<T> qh((size_t)nq * dim);
  for (size_t i = 0; i < qh.size(); ++i) {
    if constexpr (std::is_same_v<T, half>) qh[i] = __float2half_rn(q32[i]);
    else qh[i] = q32[i];
  }
  T* h_q = nullptr;
  uint32_t* h_n = nullptr;
  ck(cudaMallocHost(&h_q, qh.size() * sizeof(T)), "pin q");
  ck(cudaMallocHost(&h_n, (size_t)nq * a.k * sizeof(uint32_t)), "pin n");
  std::memcpy(h_q, qh.data(), qh.size() * sizeof(T));
  auto d_q = raft::make_device_matrix<T, int64_t>(res, nq, dim);
  auto d_n = raft::make_device_matrix<uint32_t, int64_t>(res, nq, a.k);
  auto d_d = raft::make_device_matrix<float, int64_t>(res, nq, a.k);
  cudaEvent_t e[6];
  for (auto& x : e) ck(cudaEventCreate(&x), "event");

  std::ofstream sf(a.out + "/sweep.tsv"), df(a.out + "/it_batches.tsv");
  sf << "it\titopk\tnq\twarmup_kernel_ms\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10"
        "\tkernel_ms_std\twall_ms_std\tn_timed\n";
  df << "it\titopk\tphase\tnq\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10\n";
  struct Run {
    float h2d = 0, kern = 0, d2h = 0;
    double wall = 0;
    std::string rec;
  };
  auto run_once = [&](int it, int itopk, const char* phase) -> Run {
    cagra::search_params sp;
    sp.max_iterations = it;
    sp.itopk_size = itopk;
    sp.search_width = a.search_width;
    auto w0 = std::chrono::steady_clock::now();
    ck(cudaEventRecord(e[0], s), "rec");
    ck(cudaMemcpyAsync(d_q.data_handle(), h_q, (size_t)nq * dim * sizeof(T), cudaMemcpyHostToDevice, s), "H2D");
    ck(cudaEventRecord(e[1], s), "rec");
    ck(cudaEventSynchronize(e[1]), "sync");
    ck(cudaEventRecord(e[2], s), "rec");
    cagra::search(res, sp, idx, raft::make_const_mdspan(d_q.view()), d_n.view(), d_d.view());
    ck(cudaEventRecord(e[3], s), "rec");
    ck(cudaEventSynchronize(e[3]), "sync");
    ck(cudaEventRecord(e[4], s), "rec");
    ck(cudaMemcpyAsync(h_n, d_n.data_handle(), (size_t)nq * a.k * sizeof(uint32_t), cudaMemcpyDeviceToHost, s), "D2H");
    ck(cudaEventRecord(e[5], s), "rec");
    ck(cudaEventSynchronize(e[5]), "sync");
    float h2d = 0, kern = 0, d2h = 0;
    cudaEventElapsedTime(&h2d, e[0], e[1]);
    cudaEventElapsedTime(&kern, e[2], e[3]);
    cudaEventElapsedTime(&d2h, e[4], e[5]);
    double wall = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - w0).count();
    char rec[32] = "";
    if (!gt.empty()) {
      const int kk = std::min(a.k, gt_k);
      double hits = 0;
      for (int i = 0; i < nq; ++i) {
        std::unordered_set<int32_t> g(gt.begin() + (size_t)i * gt_k, gt.begin() + (size_t)i * gt_k + kk);
        for (int j = 0; j < kk; ++j) hits += g.count((int32_t)h_n[(size_t)i * a.k + j]);
      }
      std::snprintf(rec, sizeof(rec), "%.6f", hits / ((double)nq * kk));
    }
    char row[256];
    std::snprintf(row, sizeof(row), "%.4f\t%.4f\t%.4f\t%.4f\t%.2f\t%s", h2d, kern, d2h, wall, nq / (wall / 1000.0), rec);
    df << it << "\t" << itopk << "\t" << phase << "\t" << nq << "\t" << row << "\n";
    Run r;
    r.h2d = h2d;
    r.kern = kern;
    r.d2h = d2h;
    r.wall = wall;
    r.rec = rec;
    return r;
  };
  for (int it : load_its(a.it_file)) {
    it_sleep();
    const int itopk = itopk_for_it(it, a.k);
    const Run w = run_once(it, itopk, "warmup");
    std::vector<double> h2d_v, kern_v, d2h_v, wall_v, rec_v;
    for (int r = 0; r < kTimedRuns; ++r) {
      const std::string phase = "timed" + std::to_string(r + 1);
      const Run t = run_once(it, itopk, phase.c_str());
      h2d_v.push_back(t.h2d);
      kern_v.push_back(t.kern);
      d2h_v.push_back(t.d2h);
      wall_v.push_back(t.wall);
      if (!t.rec.empty()) rec_v.push_back(std::atof(t.rec.c_str()));
    }
    const double wall = mean_of(wall_v);
    char rec[32] = "";
    if (!rec_v.empty()) std::snprintf(rec, sizeof(rec), "%.6f", mean_of(rec_v));
    char row[320];
    std::snprintf(row, sizeof(row), "%.4f\t%.4f\t%.4f\t%.4f\t%.2f\t%s\t%.4f\t%.4f\t%d", mean_of(h2d_v),
                  mean_of(kern_v), mean_of(d2h_v), wall, nq / (wall / 1000.0), rec, sd_of(kern_v), sd_of(wall_v),
                  kTimedRuns);
    sf << it << "\t" << itopk << "\t" << nq << "\t" << w.kern << "\t" << row << "\n";
    std::printf("[cagra-%s] it=%d itopk=%d %s (mean of %d)\n", a.precision.c_str(), it, itopk, row, kTimedRuns);
  }
  cudaFreeHost(h_q);
  cudaFreeHost(h_n);
}

int main(int argc, char** argv) {
  Args a = parse(argc, argv);
  std::system(("mkdir -p \"" + a.out + "\"").c_str());
  raft::device_resources res;

  cagra::index<float, uint32_t> idx_f(res);
  cagra::deserialize(res, a.index, &idx_f);
  const int64_t n = idx_f.size(), dim = idx_f.dim();
  std::printf("[cagra] index n=%ld dim=%ld degree=%u\n", (long)n, (long)dim, idx_f.graph_degree());

  // Queries and ground truth
  std::ifstream qf(a.query, std::ios::binary);
  if (!qf) die("open query");
  int32_t qn = 0, qd = 0;
  qf.read((char*)&qn, 4);
  qf.read((char*)&qd, 4);
  if (qd != dim) die("query dim mismatch");
  const int nq = std::min(a.batch_size, (int)qn);
  std::vector<float> q32((size_t)nq * dim);
  qf.read((char*)q32.data(), q32.size() * sizeof(float));
  std::vector<int32_t> gt;
  int32_t gn = 0, gk = 0;
  if (!a.gt.empty()) {
    std::ifstream gf(a.gt, std::ios::binary);
    if (gf) {
      gf.read((char*)&gn, 4);
      gf.read((char*)&gk, 4);
      gt.resize((size_t)gn * gk);
      gf.read((char*)gt.data(), gt.size() * 4);
    }
  }
  {
    std::ofstream f(a.out + "/params.txt");
    f << "algo\tCAGRA\nvector_precision\t" << (a.precision == "fp16" ? "fp16 (dataset and query half, graph from the float index)" : "fp32")
      << "\nindex\t" << a.index << "\nquery\t" << a.query << "\nbatch_size\t" << a.batch_size << "\nn_query_this_run\t" << nq
      << "\nit_file\t" << a.it_file << "\nitopk_policy\tmax(32, ceil32(it))\nsearch_width\t" << a.search_width
      << "\nprotocol\tfirst_batch_only_warmup1_timed" << kTimedRuns << "_mean\ndriver\tcagra_resident_fp16.cu (cuVS C++ API)\n";
  }

  if (a.precision == "fp32") {
    run<float>(a, res, idx_f, q32, nq, (int)dim, gt, gk);
    return 0;
  }
  auto* sd = dynamic_cast<const cuvs::neighbors::strided_dataset<float, int64_t>*>(&idx_f.data());
  if (sd == nullptr) die("float index has no strided dataset attached");
  auto v = sd->view();
  auto d_h = raft::make_device_matrix<half, int64_t>(res, n, dim);
  const int64_t total = n * dim;
  float_to_half<<<(unsigned)((total + 255) / 256), 256, 0, raft::resource::get_cuda_stream(res)>>>(
      v.data_handle(), v.stride(0), d_h.data_handle(), n, dim);
  ck(cudaStreamSynchronize(raft::resource::get_cuda_stream(res)), "convert");
  cagra::index<half, uint32_t> idx_h(res, idx_f.metric());
  idx_h.update_dataset(res, raft::make_const_mdspan(d_h.view()));
  idx_h.update_graph(res, idx_f.graph());  // references the graph of the float index
  if (idx_h.graph().data_handle() != idx_f.graph().data_handle()) die("graph was copied unexpectedly");
  run<half>(a, res, idx_h, q32, nq, (int)dim, gt, gk);
  return 0;
}
