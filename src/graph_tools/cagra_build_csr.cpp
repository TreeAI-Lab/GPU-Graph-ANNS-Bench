/*
 * Build a CAGRA graph with NN-descent and cagra::optimize and write it as CSR.
 * With the patched libcuvs, CUVS_CAGRA_NO_REVERSE=1 makes optimize skip the reverse edges.
 *
 *   ./cagra_build_csr <base.fbin|data_dir> <out_dir> [gd=32] [igd=64] [niter=20]
 *   ./cagra_build_csr --ab <base.fbin|data_dir> <out_dir> [gd=32] [igd=64] [niter=20]
 *
 * --ab: one nn-descent, then optimize twice (with / without reverse) into
 *       <out_dir>/with_reverse and <out_dir>/no_reverse.
 */

#include <algorithm>
#include <chrono>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

#include <cuda_runtime.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdarray.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <cuvs/distance/distance.hpp>
#include <cuvs/neighbors/cagra.hpp>
#include <cuvs/neighbors/nn_descent.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "dataset_io.hpp"

// cagra::optimize on host graphs, compiled into libcuvs
namespace cuvs::neighbors::cagra {
void optimize(raft::resources const& handle,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> knn_graph,
              raft::host_matrix_view<uint32_t, int64_t, raft::row_major> new_graph);
}

namespace {

void usage(const char* argv0) {
  fprintf(stderr,
          "Usage:\n"
          "  %s <base.fbin|data_dir> <out_dir> [gd=32] [igd=64] [niter=20]\n"
          "  %s --ab <base.fbin|data_dir> <out_dir> [gd=32] [igd=64] [niter=20]\n"
          "Env: CUVS_CAGRA_NO_REVERSE=1 skips reverse merge (single-run mode).\n",
          argv0,
          argv0);
}

bool is_dir(const char* p) {
  struct stat st;
  return ::stat(p, &st) == 0 && S_ISDIR(st.st_mode);
}

void ensure_dir(const std::string& path) {
  if (mkdir(path.c_str(), 0755) != 0 && errno != EEXIST) {
    fprintf(stderr, "ERROR: mkdir %s failed\n", path.c_str());
    abort();
  }
}

void write_csr(const std::string& path, int64_t n, int degree, const uint32_t* adj) {
  const int64_t nnz = n * static_cast<int64_t>(degree);
  std::vector<int64_t> offsets(static_cast<size_t>(n) + 1);
  for (int64_t i = 0; i <= n; ++i) offsets[static_cast<size_t>(i)] = i * degree;
  std::vector<int32_t> nbrs(static_cast<size_t>(nnz));
  for (int64_t i = 0; i < nnz; ++i) nbrs[static_cast<size_t>(i)] = static_cast<int32_t>(adj[i]);

  FILE* f = fopen(path.c_str(), "wb");
  if (!f) {
    fprintf(stderr, "ERROR: cannot write %s\n", path.c_str());
    abort();
  }
  fwrite(&n, sizeof(int64_t), 1, f);
  fwrite(&nnz, sizeof(int64_t), 1, f);
  fwrite(offsets.data(), sizeof(int64_t), offsets.size(), f);
  fwrite(nbrs.data(), sizeof(int32_t), nbrs.size(), f);
  fclose(f);
}

void write_outputs(const std::string& out_dir,
                   int64_t n,
                   int gd,
                   int igd,
                   int niter,
                   const std::string& base_path,
                   size_t d,
                   const uint32_t* adj,
                   double sec,
                   const char* norev_label) {
  ensure_dir(out_dir);
  write_csr(out_dir + "/graph.csr", n, gd, adj);
  {
    FILE* f = fopen((out_dir + "/graph.abin").c_str(), "wb");
    if (!f) abort();
    int32_t hn = static_cast<int32_t>(n), hd = gd;
    fwrite(&hn, 4, 1, f);
    fwrite(&hd, 4, 1, f);
    fwrite(adj, sizeof(uint32_t), static_cast<size_t>(n) * gd, f);
    fclose(f);
  }
  {
    FILE* f = fopen((out_dir + "/params.txt").c_str(), "w");
    fprintf(f, "base=%s\n", base_path.c_str());
    fprintf(f, "n=%lld dim=%zu\n", (long long)n, d);
    fprintf(f, "GD=%d\nIGD=%d\nNITER=%d\n", gd, igd, niter);
    fprintf(f, "path=nn_descent+optimize\n");
    fprintf(f, "CUVS_CAGRA_NO_REVERSE=%s\n", norev_label);
    fprintf(f, "CAGRA_BUILD_SECONDS=%.3f\n", sec);
    fclose(f);
  }
}

struct EdgeDiffStats {
  int64_t n = 0;
  int gd = 0;
  int64_t edges_total = 0;
  int64_t edges_same_as_set = 0;    // neighbor present in both lists (order ignored)
  int64_t edges_identical_pos = 0;  // same id in the same slot
  double mean_jaccard = 0;
};

EdgeDiffStats compare_adj(const uint32_t* a, const uint32_t* b, int64_t n, int gd) {
  EdgeDiffStats s;
  s.n = n;
  s.gd = gd;
  s.edges_total = n * gd;
  double jac_sum = 0;
  std::vector<uint32_t> sa(gd), sb(gd);
  for (int64_t i = 0; i < n; ++i) {
    const uint32_t* ra = a + i * gd;
    const uint32_t* rb = b + i * gd;
    for (int k = 0; k < gd; ++k) {
      if (ra[k] == rb[k]) {
        ++s.edges_identical_pos;
      }
    }
    sa.assign(ra, ra + gd);
    sb.assign(rb, rb + gd);
    std::sort(sa.begin(), sa.end());
    std::sort(sb.begin(), sb.end());
    int inter = 0, u = 0;
    int ia = 0, ib = 0;
    while (ia < gd && ib < gd) {
      if (sa[ia] == sb[ib]) {
        ++inter;
        ++ia;
        ++ib;
      } else if (sa[ia] < sb[ib]) {
        ++ia;
      } else {
        ++ib;
      }
    }
    u = gd + gd - inter;
    s.edges_same_as_set += inter;
    jac_sum += u > 0 ? static_cast<double>(inter) / u : 1.0;
  }
  s.mean_jaccard = jac_sum / static_cast<double>(n);
  return s;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) {
    usage(argv[0]);
    return 1;
  }
  bool ab_mode = false;
  int argi = 1;
  if (std::strcmp(argv[1], "--ab") == 0) {
    ab_mode = true;
    argi = 2;
    if (argc < 4) {
      usage(argv[0]);
      return 1;
    }
  }
  const char* in_path = argv[argi];
  const std::string out_dir = argv[argi + 1];
  const int gd = argc > argi + 2 ? atoi(argv[argi + 2]) : 32;
  const int igd = argc > argi + 3 ? atoi(argv[argi + 3]) : 64;
  const int niter = argc > argi + 4 ? atoi(argv[argi + 4]) : 20;

  ensure_dir(out_dir);

  std::string base_path = in_path;
  if (is_dir(in_path)) {
    auto paths = dsio::resolve_or_die(in_path, /*need_gt=*/false, /*need_learn=*/false);
    base_path = paths.base;
  }

  size_t n = 0, d = 0;
  float* host = dsio::floats_read(base_path, &d, &n);
  fprintf(stderr, "loaded %s: n=%zu d=%zu gd=%d igd=%d niter=%d ab=%d\n",
          base_path.c_str(), n, d, gd, igd, niter, (int)ab_mode);

  rmm::mr::device_memory_resource* default_mr = rmm::mr::get_current_device_resource();
  rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
    default_mr, 2ull * 1024ull * 1024ull * 1024ull);
  rmm::mr::set_current_device_resource(&pool_mr);

  raft::device_resources res;
  auto dataset =
    raft::make_device_matrix<float, int64_t>(res, static_cast<int64_t>(n), static_cast<int64_t>(d));
  raft::copy(dataset.data_handle(), host, n * d, raft::resource::get_cuda_stream(res));
  raft::resource::sync_stream(res);
  delete[] host;

  auto dataset_const = raft::make_const_mdspan(dataset.view());

  cuvs::neighbors::nn_descent::index_params nn_params(static_cast<size_t>(igd));
  nn_params.metric = cuvs::distance::DistanceType::L2Expanded;
  nn_params.max_iterations = static_cast<size_t>(niter);
  nn_params.return_distances = false;

  auto t_all0 = std::chrono::steady_clock::now();
  fprintf(stderr, "nn_descent build...\n");
  auto knn_index = cuvs::neighbors::nn_descent::build(res, nn_params, dataset_const);
  raft::resource::sync_stream(res);
  fprintf(stderr, "nn_descent done\n");

  // every optimize call gets its own copy of the kNN graph
  auto knn_copy = [&]() {
    auto g = knn_index.graph();
    auto m = raft::make_host_matrix<uint32_t, int64_t>(g.extent(0), g.extent(1));
    std::memcpy(m.data_handle(), g.data_handle(),
                sizeof(uint32_t) * static_cast<size_t>(g.extent(0) * g.extent(1)));
    return m;
  };

  auto run_opt = [&](bool skip_reverse, raft::host_matrix<uint32_t, int64_t>& knn,
                     raft::host_matrix<uint32_t, int64_t>& out) {
    if (skip_reverse) {
      setenv("CUVS_CAGRA_NO_REVERSE", "1", 1);
    } else {
      unsetenv("CUVS_CAGRA_NO_REVERSE");
    }
    auto t0 = std::chrono::steady_clock::now();
    cuvs::neighbors::cagra::optimize(res, knn.view(), out.view());
    raft::resource::sync_stream(res);
    return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  };

  if (!ab_mode) {
    const char* norev = std::getenv("CUVS_CAGRA_NO_REVERSE");
    fprintf(stderr, "CUVS_CAGRA_NO_REVERSE=%s\n", norev ? norev : "(unset)");
    auto knn = knn_copy();
    auto cagra_graph =
      raft::make_host_matrix<uint32_t, int64_t>(static_cast<int64_t>(n), static_cast<int64_t>(gd));
    cuvs::neighbors::cagra::optimize(res, knn.view(), cagra_graph.view());
    raft::resource::sync_stream(res);
    // NN-descent plus optimize
    double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t_all0).count();
    printf("CAGRA_BUILD_SECONDS %.3f\n", sec);
    write_outputs(out_dir, static_cast<int64_t>(n), gd, igd, niter, base_path, d,
                  cagra_graph.data_handle(), sec, norev ? norev : "");
    fprintf(stderr, "wrote %s/graph.csr\n", out_dir.c_str());
  } else {
    auto knn_a = knn_copy();
    auto knn_b = knn_copy();
    auto with_rev =
      raft::make_host_matrix<uint32_t, int64_t>(static_cast<int64_t>(n), static_cast<int64_t>(gd));
    auto no_rev =
      raft::make_host_matrix<uint32_t, int64_t>(static_cast<int64_t>(n), static_cast<int64_t>(gd));

    fprintf(stderr, "optimize WITH reverse...\n");
    double sec_with = run_opt(false, knn_a, with_rev);
    fprintf(stderr, "optimize NO reverse...\n");
    double sec_no = run_opt(true, knn_b, no_rev);

    const std::string dir_with = out_dir + "/with_reverse";
    const std::string dir_no = out_dir + "/no_reverse";
    write_outputs(dir_with, static_cast<int64_t>(n), gd, igd, niter, base_path, d,
                  with_rev.data_handle(), sec_with, "");
    write_outputs(dir_no, static_cast<int64_t>(n), gd, igd, niter, base_path, d,
                  no_rev.data_handle(), sec_no, "1");

    auto st = compare_adj(with_rev.data_handle(), no_rev.data_handle(),
                          static_cast<int64_t>(n), gd);
    FILE* f = fopen((out_dir + "/compare.txt").c_str(), "w");
    fprintf(f, "shared_nn_descent=1\n");
    fprintf(f, "n=%lld gd=%d\n", (long long)st.n, st.gd);
    fprintf(f, "optimize_seconds_with_reverse=%.3f\n", sec_with);
    fprintf(f, "optimize_seconds_no_reverse=%.3f\n", sec_no);
    fprintf(f, "edges_total=%lld\n", (long long)st.edges_total);
    fprintf(f, "edges_identical_position=%lld (%.4f)\n",
            (long long)st.edges_identical_pos,
            (double)st.edges_identical_pos / st.edges_total);
    fprintf(f, "edges_same_as_set=%lld (%.4f of with_reverse edges)\n",
            (long long)st.edges_same_as_set,
            (double)st.edges_same_as_set / st.edges_total);
    fprintf(f, "mean_neighbor_jaccard=%.6f\n", st.mean_jaccard);
    fprintf(f, "edges_changed_set=%lld (%.4f)\n",
            (long long)(st.edges_total - st.edges_same_as_set),
            (double)(st.edges_total - st.edges_same_as_set) / st.edges_total);
    fclose(f);

    printf("AB_OPTIMIZE_WITH_REVERSE_SECONDS %.3f\n", sec_with);
    printf("AB_OPTIMIZE_NO_REVERSE_SECONDS %.3f\n", sec_no);
    printf("AB_EDGES_CHANGED_FRAC %.6f\n",
           (double)(st.edges_total - st.edges_same_as_set) / st.edges_total);
    printf("AB_MEAN_JACCARD %.6f\n", st.mean_jaccard);
    printf("AB_IDENTICAL_POS_FRAC %.6f\n",
           (double)st.edges_identical_pos / st.edges_total);
    fprintf(stderr, "wrote %s/{with_reverse,no_reverse}/graph.csr and compare.txt\n",
            out_dir.c_str());
  }

  rmm::mr::set_current_device_resource(default_mr);
  return 0;
}
