/*
 * Sample S base points (one sample shared by all graphs) and compute their exact top-K
 * neighbors via cuVS GPU brute-force. Self is excluded from the neighbor list.
 *
 *   ./sample_top32 <data_dir> <out_dir> [S=10000] [K=32] [seed=42] [bf_index]
 *
 * If bf_index is given, deserialize that cuVS BF index; otherwise build from base.
 *
 * Writes (every graph is later measured against these files):
 *   sample_ids.ibin     int32 [S, 1]   sampled base ids
 *   gt_topK.ibin        int32 [S, K]   exact neighbors (no self)
 *   gt_topK_dist.fbin   float [S, K]   L2 distances to those neighbors
 *   meta.txt
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <cuvs/neighbors/brute_force.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "dataset_io.hpp"
#include "ibin_io.hpp"

namespace {

cudaStream_t raft_stream(raft::device_resources& handle) {
    return static_cast<cudaStream_t>(raft::resource::get_cuda_stream(handle).value());
}

void cuda_check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(e));
        abort();
    }
}

double elapsed_ms(std::chrono::steady_clock::time_point t0,
                  std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

void usage(const char* argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s <data_dir> <out_dir> [S=10000] [K=32] [seed=42] [bf_index]\n"
            "\n"
            "Sample S base ids (seeded RNG, shared file) and write exact top-K\n"
            "via cuVS GPU brute-force (self excluded).\n",
            argv0);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }
    const char* data_dir = argv[1];
    const char* out_dir = argv[2];
    const int S = argc > 3 ? atoi(argv[3]) : 10000;
    const int K = argc > 4 ? atoi(argv[4]) : 32;
    const unsigned seed = argc > 5 ? static_cast<unsigned>(atoi(argv[5])) : 42u;
    const char* bf_index = argc > 6 ? argv[6] : nullptr;

    if (S <= 0 || K <= 0) {
        fprintf(stderr, "S and K must be > 0\n");
        return 1;
    }

    // ensure out_dir exists
    {
        std::string cmd = std::string("mkdir -p '") + out_dir + "'";
        if (system(cmd.c_str()) != 0) {
            fprintf(stderr, "mkdir failed: %s\n", out_dir);
            return 1;
        }
    }

    auto paths = dsio::resolve_or_die(data_dir, /*need_gt=*/false, /*need_learn=*/false);
    size_t d = 0, nb = 0;
    auto t0 = std::chrono::steady_clock::now();
    float* xb = dsio::floats_read(paths.base, &d, &nb);
    auto t1 = std::chrono::steady_clock::now();
    printf("loaded base nb=%zu d=%zu in %.2f ms from %s\n",
           nb,
           d,
           elapsed_ms(t0, t1),
           paths.base.c_str());

    if (static_cast<size_t>(S) > nb) {
        fprintf(stderr, "S=%d > nb=%zu\n", S, nb);
        delete[] xb;
        return 1;
    }
    if (static_cast<size_t>(K + 1) > nb) {
        fprintf(stderr, "K+1=%d > nb=%zu\n", K + 1, nb);
        delete[] xb;
        return 1;
    }

    // ---- shared sample (shuffle then take prefix) ----
    std::vector<int> all(nb);
    std::iota(all.begin(), all.end(), 0);
    std::mt19937 rng(seed);
    std::shuffle(all.begin(), all.end(), rng);
    std::vector<int> sample_ids(all.begin(), all.begin() + S);
    std::sort(sample_ids.begin(), sample_ids.end());  // stable order for later joins

    const std::string ids_path = dsio::join_path(out_dir, "sample_ids.ibin");
    binio::write_ibin(ids_path, sample_ids.data(), static_cast<size_t>(S), 1);
    printf("wrote %s (S=%d seed=%u, sorted)\n", ids_path.c_str(), S, seed);

    // pack query matrix
    std::vector<float> xq(static_cast<size_t>(S) * d);
    for (int i = 0; i < S; i++) {
        std::memcpy(xq.data() + static_cast<size_t>(i) * d,
                    xb + static_cast<size_t>(sample_ids[i]) * d,
                    d * sizeof(float));
    }

    // ---- cuVS BF ----
    cuda_check(cudaSetDevice(0), "cudaSetDevice");
    raft::device_resources handle;
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
            rmm::mr::get_current_device_resource(), 2ull * 1024ull * 1024ull * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);
    cudaStream_t stream = raft_stream(handle);

    using namespace cuvs::neighbors;
    brute_force::index<float, float> index(handle);

    t0 = std::chrono::steady_clock::now();
    if (bf_index && *bf_index) {
        printf("deserialize BF index: %s\n", bf_index);
        brute_force::deserialize(handle, std::string(bf_index), &index);
    } else {
        printf("build BF index from base (nb=%zu d=%zu)\n", nb, d);
        auto d_base = raft::make_device_matrix<float, int64_t>(
                handle, static_cast<int64_t>(nb), static_cast<int64_t>(d));
        cuda_check(cudaMemcpyAsync(d_base.data_handle(),
                                   xb,
                                   nb * d * sizeof(float),
                                   cudaMemcpyHostToDevice,
                                   stream),
                   "H2D base");
        cuda_check(cudaStreamSynchronize(stream), "H2D base sync");
        brute_force::index_params ip;
        ip.metric = cuvs::distance::DistanceType::L2Expanded;
        index = brute_force::build(handle, ip, raft::make_const_mdspan(d_base.view()));
    }
    t1 = std::chrono::steady_clock::now();
    printf("BF index ready in %.2f ms | size=%zu dim=%zu\n",
           elapsed_ms(t0, t1),
           index.size(),
           index.dim());

    // search K+1 then drop self
    const int k_search = K + 1;
    auto d_queries = raft::make_device_matrix<float, int64_t>(
            handle, static_cast<int64_t>(S), static_cast<int64_t>(d));
    auto d_neighbors = raft::make_device_matrix<int64_t, int64_t>(
            handle, static_cast<int64_t>(S), static_cast<int64_t>(k_search));
    auto d_distances = raft::make_device_matrix<float, int64_t>(
            handle, static_cast<int64_t>(S), static_cast<int64_t>(k_search));

    cuda_check(cudaMemcpyAsync(d_queries.data_handle(),
                               xq.data(),
                               static_cast<size_t>(S) * d * sizeof(float),
                               cudaMemcpyHostToDevice,
                               stream),
               "H2D queries");
    cuda_check(cudaStreamSynchronize(stream), "H2D q sync");

    brute_force::search_params sp;
    // warmup
    {
        int64_t nw = std::min<int64_t>(S, 16);
        auto q_warm = raft::make_device_matrix_view<const float, int64_t>(
                d_queries.data_handle(), nw, static_cast<int64_t>(d));
        auto n_warm = raft::make_device_matrix_view<int64_t, int64_t>(
                d_neighbors.data_handle(), nw, k_search);
        auto d_warm = raft::make_device_matrix_view<float, int64_t>(
                d_distances.data_handle(), nw, k_search);
        brute_force::search(handle, sp, index, q_warm, n_warm, d_warm);
        cuda_check(cudaStreamSynchronize(stream), "warmup sync");
    }

    t0 = std::chrono::steady_clock::now();
    brute_force::search(handle,
                        sp,
                        index,
                        raft::make_const_mdspan(d_queries.view()),
                        d_neighbors.view(),
                        d_distances.view());
    cuda_check(cudaStreamSynchronize(stream), "search sync");
    t1 = std::chrono::steady_clock::now();
    printf("BF search S=%d k=%d done in %.2f ms\n", S, k_search, elapsed_ms(t0, t1));

    std::vector<int64_t> I(static_cast<size_t>(S) * k_search);
    std::vector<float> D(static_cast<size_t>(S) * k_search);
    cuda_check(cudaMemcpy(I.data(),
                          d_neighbors.data_handle(),
                          I.size() * sizeof(int64_t),
                          cudaMemcpyDeviceToHost),
               "D2H I");
    cuda_check(cudaMemcpy(D.data(),
                          d_distances.data_handle(),
                          D.size() * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "D2H D");

    std::vector<int> gt(static_cast<size_t>(S) * K);
    std::vector<float> gt_dist(static_cast<size_t>(S) * K);
    size_t missing_self = 0;
    for (int i = 0; i < S; i++) {
        const int uid = sample_ids[i];
        int filled = 0;
        bool saw_self = false;
        for (int j = 0; j < k_search && filled < K; j++) {
            int64_t v = I[static_cast<size_t>(i) * k_search + j];
            float dist = D[static_cast<size_t>(i) * k_search + j];
            if (v == uid) {
                saw_self = true;
                continue;
            }
            if (v < 0 || static_cast<size_t>(v) >= nb) {
                continue;
            }
            gt[static_cast<size_t>(i) * K + filled] = static_cast<int>(v);
            gt_dist[static_cast<size_t>(i) * K + filled] = dist;
            filled++;
        }
        if (!saw_self) {
            missing_self++;
        }
        if (filled < K) {
            fprintf(stderr,
                    "ERROR: sample %d (id=%d) only got %d/%d neighbors after dropping self\n",
                    i,
                    uid,
                    filled,
                    K);
            delete[] xb;
            return 1;
        }
    }
    if (missing_self) {
        printf("WARN: %zu / %d queries did not return self in top-%d (still filled K)\n",
               missing_self,
               S,
               k_search);
    }

    const std::string gt_path =
            dsio::join_path(out_dir, ("gt_top" + std::to_string(K) + ".ibin").c_str());
    const std::string dist_path =
            dsio::join_path(out_dir, ("gt_top" + std::to_string(K) + "_dist.fbin").c_str());
    binio::write_ibin(gt_path, gt.data(), static_cast<size_t>(S), static_cast<size_t>(K));
    binio::write_fbin(dist_path, gt_dist.data(), static_cast<size_t>(S), static_cast<size_t>(K));
    printf("wrote %s\n", gt_path.c_str());
    printf("wrote %s\n", dist_path.c_str());

    const std::string meta_path = dsio::join_path(out_dir, "meta.txt");
    FILE* mf = fopen(meta_path.c_str(), "w");
    if (mf) {
        fprintf(mf, "data_dir\t%s\n", data_dir);
        fprintf(mf, "base\t%s\n", paths.base.c_str());
        fprintf(mf, "nb\t%zu\n", nb);
        fprintf(mf, "d\t%zu\n", d);
        fprintf(mf, "S\t%d\n", S);
        fprintf(mf, "K\t%d\n", K);
        fprintf(mf, "seed\t%u\n", seed);
        fprintf(mf, "sample_order\tsorted_ascending\n");
        fprintf(mf, "metric\tL2Expanded\n");
        fprintf(mf, "bf_index\t%s\n", bf_index && *bf_index ? bf_index : "(built_from_base)");
        fprintf(mf, "sample_ids\t%s\n", ids_path.c_str());
        fprintf(mf, "gt\t%s\n", gt_path.c_str());
        fprintf(mf, "gt_dist\t%s\n", dist_path.c_str());
        fprintf(mf,
                "note\tall algorithms must reuse sample_ids.ibin + gt_topK.ibin for fair comparison\n");
        fclose(mf);
        printf("wrote %s\n", meta_path.c_str());
    }

    delete[] xb;
    printf("DONE sample_top%d S=%d\n", K, S);
    return 0;
}
