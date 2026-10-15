/*
 * IVF-PQ on the GPU with cuVS (cuvs::neighbors::ivf_pq), with per-phase timing and memory metrics.
 *
 * Build and search are separate modes; the index is serialized to a file:
 *
 *   cuvs_ivf_pq build  <data_dir> [nlist] [m] [index_path] [json_out]
 *   cuvs_ivf_pq search <data_dir> [index_path] [nprobe] [k] [json_out]
 *   cuvs_ivf_pq all    <data_dir> [nlist] [m] [nprobe] [k] [json_out]
 *                      (build and search in one process, no index file, no refinement)
 *
 * Defaults (same as faiss_ivf_pq):
 *   nlist=4096 m=16 nprobe=64 k=100 (pq_bits=8, device 0)
 *   index_path=cuvs_ivf_pq.cuvsindex
 *   json_out=survey_metrics_{build|search|all}.json
 *
 * Build: k-means and PQ codebooks are trained on learn.fbin if the dataset has one, else on
 * the first LEARN_RATIO (default 0.1) of the base vectors; then the base vectors are added.
 *
 * Search mode: lookup tables and internal distances in float32 (cuVS default; CUVS_FP16_LUT=1 for
 * float16). With REFINE_RATIO r > 1 (default 2) the search returns ceil(r * k) candidates that
 * are re-ranked with exact distances on the base vectors (cuvs::neighbors::refine).
 * One untimed warm-up of the whole search (ANN plus refine) on all queries, then the phases
 * search_ann, refine and search_e2e (ANN plus refine) each run SEARCH_REPEATS timed searches
 * (default 3) and report the mean wall and CUDA-event time per search, their sample standard
 * deviations and the number of timed searches; search_qps = nq / mean wall time of search_e2e.
 * Recall is computed from the results of the last timed search_e2e run.
 * Recall: R@x is the fraction of queries whose nearest ground-truth neighbor is among the
 * first x returned ids.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <optional>
#include <string>
#include <strings.h>
#include <sys/stat.h>
#include <vector>

#include <cuda_runtime.h>

#include <raft/core/device_mdarray.hpp>
#include <raft/core/device_resources.hpp>
#include <raft/core/host_mdspan.hpp>
#include <raft/core/resource/cuda_stream.hpp>
#include <cuvs/neighbors/ivf_pq.hpp>
#include <cuvs/neighbors/refine.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "survey_profiler_cuvs_ivf.hpp"
#include "dataset_io.hpp"
#include "recall.hpp"

namespace {

using dsio::join_path;

void usage(const char* argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s build  <data_dir> [nlist] [m] [index_path] [json_out]\n"
            "  %s search <data_dir> [index_path] [nprobe] [k] [json_out]\n"
            "  %s all    <data_dir> [nlist] [m] [nprobe] [k] [json_out]\n",
            argv0,
            argv0,
            argv0);
}

cudaStream_t raft_stream(raft::device_resources& handle) {
    return static_cast<cudaStream_t>(raft::resource::get_cuda_stream(handle).value());
}

// cuVS defaults: lookup table and internal distance in float32.
// CUVS_FP16_LUT=1 selects float16. REFINE_RATIO (default 2; 0 or 1 disables refinement).
bool env_fp16_lut() {
    const char* v = getenv("CUVS_FP16_LUT");
    return v && (strcmp(v, "1") == 0 || strcasecmp(v, "true") == 0);
}

float env_refine_ratio() {
    const char* v = getenv("REFINE_RATIO");
    if (!v || !*v) {
        return 2.f; // refinement on by default, as recommended by cuVS
    }
    float r = static_cast<float>(atof(v));
    return r;
}

// Timed searches per setting after the warm-up (SEARCH_REPEATS, default 3)
int env_search_repeats() {
    const char* v = getenv("SEARCH_REPEATS");
    if (!v || !*v) {
        return 3;
    }
    int r = atoi(v);
    return r > 0 ? r : 3;
}

// IVF_METRIC=cosine|angular|ip|inner_product: L2-normalize and use InnerProduct
// (for angular datasets). Unset: L2Expanded.
bool env_use_cosine() {
    const char* v = getenv("IVF_METRIC");
    if (!v || !*v) {
        return false;
    }
    return strcasecmp(v, "cosine") == 0 || strcasecmp(v, "angular") == 0 ||
            strcasecmp(v, "ip") == 0 || strcasecmp(v, "inner_product") == 0;
}

cuvs::distance::DistanceType env_cuvs_metric() {
    return env_use_cosine() ? cuvs::distance::DistanceType::InnerProduct
                            : cuvs::distance::DistanceType::L2Expanded;
}

void l2_normalize_rows(float* x, size_t n, size_t d) {
    for (size_t i = 0; i < n; i++) {
        float* row = x + i * d;
        double ss = 0.0;
        for (size_t j = 0; j < d; j++) {
            ss += static_cast<double>(row[j]) * static_cast<double>(row[j]);
        }
        float inv = (ss > 0.0) ? static_cast<float>(1.0 / std::sqrt(ss)) : 0.f;
        for (size_t j = 0; j < d; j++) {
            row[j] *= inv;
        }
    }
}

bool is_ip_metric(cuvs::distance::DistanceType m) {
    return m == cuvs::distance::DistanceType::InnerProduct ||
            m == cuvs::distance::DistanceType::CosineExpanded;
}

void finalize_report(survey::SurveyReport& report, const char* json_out) {
    report.host_hwm_end = survey::host_mem_snap().vm_hwm_bytes;
    report.gpu_used_peak_global = 0;
    for (const auto& p : report.phases) {
        if (p.gpu_used_peak > report.gpu_used_peak_global) {
            report.gpu_used_peak_global = p.gpu_used_peak;
        }
    }
    report.print_all();
    report.write_json(json_out);
}

int cmd_build(int argc, char** argv) {
    const char* data_dir = argv[2];
    int nlist = argc > 3 ? atoi(argv[3]) : 4096;
    int m = argc > 4 ? atoi(argv[4]) : 16;
    const char* index_path = argc > 5 ? argv[5] : "cuvs_ivf_pq.cuvsindex";
    const char* json_out = argc > 6 ? argv[6] : "survey_metrics_build.json";
    const int nbits = 8;
    const int device = 0;

    survey::SurveyReport report;
    report.nlist = nlist;
    report.m = m;
    report.nbits = nbits;
    report.nprobe = 0;
    report.k = 0;
    report.device = device;
    report.use_fp16_lut = false; // search-time LUT; build has no LUT

    survey::cuda_check(cudaSetDevice(device), "cudaSetDevice");
    {
        auto g = survey::gpu_mem_snap();
        report.gpu_total_bytes = g.total_bytes;
        printf("GPU total memory: %.3f GiB (used now %.3f GiB)\n",
               g.total_bytes / (1024.0 * 1024.0 * 1024.0),
               g.used_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    raft::device_resources handle;
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
            rmm::mr::get_current_device_resource(), 1024ull * 1024ull * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);
    cudaStream_t stream = raft_stream(handle);
    printf("cuVS/raft stream = %p\n", (void*)stream);

    size_t d = 0, nt = 0, nb = 0;
    float* xt = nullptr;
    float* xb = nullptr;

    const bool use_cosine = env_use_cosine();
    const auto metric = env_cuvs_metric();
    const dsio::DatasetPaths paths = dsio::resolve_or_die(data_dir, /*need_gt=*/false, /*need_learn=*/true);

    report.phases.push_back(survey::measure_phase(
            "load_dataset_host",
            [&]() {
                dsio::load_base_and_learn(paths, &xb, &nb, &xt, &nt, &d);
                if (use_cosine) {
                    l2_normalize_rows(xt, nt, d);
                    l2_normalize_rows(xb, nb, d);
                }
            },
            nullptr,
            false));

    report.d = d;
    report.nb = nb;
    report.nt = nt;
    report.nq = 0;

    printf("BUILD d=%zu nb=%zu nt=%zu | IVF-PQ nlist=%d pq_dim(m)=%d pq_bits=%d metric=%s -> %s\n",
           d,
           nb,
           nt,
           nlist,
           m,
           nbits,
           use_cosine ? "IP(cosine)" : "L2",
           index_path);

    if (d % static_cast<size_t>(m) != 0) {
        fprintf(stderr, "dimension %zu must be divisible by m=%d\n", d, m);
        return 1;
    }

    // H2D microbenchmark and device copies of the train and base vectors
    auto d_train = raft::make_device_matrix<float, int64_t>(handle, static_cast<int64_t>(nt), static_cast<int64_t>(d));
    auto d_base = raft::make_device_matrix<float, int64_t>(handle, static_cast<int64_t>(nb), static_cast<int64_t>(d));
    report.h2d_base_bytes = nb * d * sizeof(float);
    report.h2d_query_bytes = nt * d * sizeof(float); // train set as the reference payload
    report.d2h_result_bytes = 0;

    report.phases.push_back(survey::measure_phase(
            "h2d_train_and_base",
            [&]() {
                survey::cuda_check(
                        cudaMemcpyAsync(
                                d_train.data_handle(),
                                xt,
                                nt * d * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream),
                        "H2D train");
                survey::cuda_check(
                        cudaMemcpyAsync(
                                d_base.data_handle(),
                                xb,
                                nb * d * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream),
                        "H2D base");
                survey::cuda_check(cudaStreamSynchronize(stream), "H2D sync");
            },
            stream,
            true));

    // Standalone timed copies of the same sizes (bandwidth reference)
    report.h2d_query_s = survey::timed_memcpy_h2d(
            d_train.data_handle(), xt, nt * d * sizeof(float), stream);
    report.h2d_base_s = survey::timed_memcpy_h2d(
            d_base.data_handle(), xb, nb * d * sizeof(float), stream);

    delete[] xt;
    xt = nullptr;
    delete[] xb;
    xb = nullptr;

    using namespace cuvs::neighbors;
    ivf_pq::index_params index_params;
    index_params.n_lists = static_cast<uint32_t>(nlist);
    index_params.metric = metric;
    index_params.pq_bits = static_cast<uint32_t>(nbits);
    index_params.pq_dim = static_cast<uint32_t>(m);
    index_params.kmeans_trainset_fraction = 1.0;
    index_params.add_data_on_build = false; // train on the train set, then extend with the base

    std::optional<ivf_pq::index<int64_t>> index;

    report.phases.push_back(survey::measure_phase(
            "build_train",
            [&]() {
                index = ivf_pq::build(
                        handle,
                        index_params,
                        raft::make_const_mdspan(d_train.view()));
            },
            stream,
            true));

    report.phases.push_back(survey::measure_phase(
            "build_extend_add",
            [&]() {
                std::optional<raft::device_vector_view<const int64_t, int64_t>> no_ids =
                        std::nullopt;
                ivf_pq::extend(
                        handle,
                        raft::make_const_mdspan(d_base.view()),
                        no_ids,
                        &index.value());
            },
            stream,
            true));

    printf("---- index ----\n");
    printf("  n_lists=%u size=%ld dim=%u pq_dim=%u pq_bits=%u\n",
           index->n_lists(),
           static_cast<long>(index->size()),
           index->dim(),
           index->pq_dim(),
           index->pq_bits());

    size_t pq_codes_bytes = nb * static_cast<size_t>(m);
    size_t centroids_bytes = static_cast<size_t>(nlist) * d * sizeof(float);
    printf("---- index size estimate (analytical) ----\n");
    printf("  PQ codes ~ %zu B (%.3f MiB)\n",
           pq_codes_bytes,
           pq_codes_bytes / (1024.0 * 1024.0));
    printf("  IVF centroids ~ %zu B (%.3f MiB)\n",
           centroids_bytes,
           centroids_bytes / (1024.0 * 1024.0));

    report.phases.push_back(survey::measure_phase(
            "serialize_index",
            [&]() { ivf_pq::serialize(handle, std::string(index_path), index.value()); },
            stream,
            true));

    finalize_report(report, json_out);
    return 0;
}

int cmd_search(int argc, char** argv) {
    const char* data_dir = argv[2];
    const char* index_path = argc > 3 ? argv[3] : "cuvs_ivf_pq.cuvsindex";
    int nprobe = argc > 4 ? atoi(argv[4]) : 64;
    int k_search = argc > 5 ? atoi(argv[5]) : 100;
    const char* json_out = argc > 6 ? argv[6] : "survey_metrics_search.json";
    const int device = 0;

    const bool fp16_lut = env_fp16_lut();
    const float refine_ratio = env_refine_ratio();
    const bool do_refine = refine_ratio > 1.f;
    const int k_candidates =
            do_refine ? static_cast<int>(std::ceil(refine_ratio * static_cast<float>(k_search)))
                      : k_search;

    survey::SurveyReport report;
    report.nprobe = nprobe;
    report.k = k_search;
    report.device = device;
    report.use_fp16_lut = fp16_lut;
    report.refine_ratio = do_refine ? refine_ratio : 0.f;
    report.nbits = 8;

    survey::cuda_check(cudaSetDevice(device), "cudaSetDevice");
    {
        auto g = survey::gpu_mem_snap();
        report.gpu_total_bytes = g.total_bytes;
    }

    raft::device_resources handle;
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
            rmm::mr::get_current_device_resource(), 1024ull * 1024ull * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);
    cudaStream_t stream = raft_stream(handle);

    size_t d = 0, nq = 0, gt_k = 0, nb = 0;
    float* xq = nullptr;
    float* xb = nullptr;
    int* gt_int = nullptr;
    const dsio::DatasetPaths paths = dsio::resolve_or_die(data_dir, /*need_gt=*/true, /*need_learn=*/false);

    report.phases.push_back(survey::measure_phase(
            "load_queries_host",
            [&]() {
                dsio::load_queries_and_gt(paths, &xq, &nq, &gt_int, &gt_k, &d);
                if (do_refine) {
                    size_t d2 = 0;
                    xb = dsio::floats_read(paths.base, &d2, &nb);
                    assert(d2 == d);
                }
            },
            nullptr,
            false));

    std::vector<int64_t> gt(nq * gt_k);
    for (size_t i = 0; i < nq * gt_k; i++) {
        gt[i] = gt_int[i];
    }
    delete[] gt_int;
    gt_int = nullptr;

    report.d = d;
    report.nq = nq;
    if (do_refine) {
        report.nb = nb;
    }

    using namespace cuvs::neighbors;
    ivf_pq::index<int64_t> index(handle);
    report.phases.push_back(survey::measure_phase(
            "deserialize_index",
            [&]() { ivf_pq::deserialize(handle, std::string(index_path), &index); },
            stream,
            true));

    // Inner-product indexes were built on normalized vectors: normalize the queries (and the base for refine).
    if (is_ip_metric(index.metric())) {
        l2_normalize_rows(xq, nq, d);
        if (do_refine && xb) {
            l2_normalize_rows(xb, nb, d);
        }
    }

    report.nlist = static_cast<int>(index.n_lists());
    report.m = static_cast<int>(index.pq_dim());
    report.nbits = static_cast<int>(index.pq_bits());
    if (!do_refine) {
        report.nb = static_cast<size_t>(index.size());
    }

    printf("SEARCH d=%zu nq=%zu | loaded nlist=%d m=%d nbits=%d size=%zu metric=%s | nprobe=%d k=%d "
           "k0=%d fp16_lut=%d refine_ratio=%.2f\n",
           d,
           nq,
           report.nlist,
           report.m,
           report.nbits,
           static_cast<size_t>(index.size()),
           is_ip_metric(index.metric()) ? "IP(cosine)" : "L2",
           nprobe,
           k_search,
           k_candidates,
           (int)fp16_lut,
           report.refine_ratio);

    auto d_queries = raft::make_device_matrix<float, int64_t>(
            handle, static_cast<int64_t>(nq), static_cast<int64_t>(d));
    // ANN candidates (k0) and the final top-k results
    auto d_candidates = raft::make_device_matrix<int64_t, int64_t>(
            handle, static_cast<int64_t>(nq), static_cast<int64_t>(k_candidates));
    auto d_cand_dists = raft::make_device_matrix<float, int64_t>(
            handle, static_cast<int64_t>(nq), static_cast<int64_t>(k_candidates));
    auto d_neighbors = raft::make_device_matrix<int64_t, int64_t>(
            handle, static_cast<int64_t>(nq), static_cast<int64_t>(k_search));
    auto d_distances = raft::make_device_matrix<float, int64_t>(
            handle, static_cast<int64_t>(nq), static_cast<int64_t>(k_search));

    std::optional<raft::device_matrix<float, int64_t>> d_base;
    report.h2d_query_bytes = nq * d * sizeof(float);
    report.d2h_result_bytes =
            nq * static_cast<size_t>(k_search) * (sizeof(int64_t) + sizeof(float));
    report.h2d_base_bytes = do_refine ? (nb * d * sizeof(float)) : 0;

    report.phases.push_back(survey::measure_phase(
            "h2d_queries",
            [&]() {
                survey::cuda_check(
                        cudaMemcpyAsync(
                                d_queries.data_handle(),
                                xq,
                                nq * d * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream),
                        "H2D queries");
                survey::cuda_check(cudaStreamSynchronize(stream), "H2D q sync");
            },
            stream,
            true));
    report.h2d_query_s = survey::timed_memcpy_h2d(
            d_queries.data_handle(), xq, report.h2d_query_bytes, stream);

    if (do_refine) {
        d_base = raft::make_device_matrix<float, int64_t>(
                handle, static_cast<int64_t>(nb), static_cast<int64_t>(d));
        report.phases.push_back(survey::measure_phase(
                "h2d_base_for_refine",
                [&]() {
                    survey::cuda_check(
                            cudaMemcpyAsync(
                                    d_base->data_handle(),
                                    xb,
                                    nb * d * sizeof(float),
                                    cudaMemcpyHostToDevice,
                                    stream),
                            "H2D base");
                    survey::cuda_check(cudaStreamSynchronize(stream), "H2D base sync");
                },
                stream,
                true));
        report.h2d_base_s = survey::timed_memcpy_h2d(
                d_base->data_handle(), xb, report.h2d_base_bytes, stream);
    }

    ivf_pq::search_params search_params;
    search_params.n_probes = static_cast<uint32_t>(nprobe);
    // float32 by default; float16 with CUVS_FP16_LUT=1
    if (fp16_lut) {
        search_params.lut_dtype = CUDA_R_16F;
        search_params.internal_distance_dtype = CUDA_R_16F;
    } else {
        search_params.lut_dtype = CUDA_R_32F;
        search_params.internal_distance_dtype = CUDA_R_32F;
    }

    auto run_ann = [&]() {
        ivf_pq::search(
                handle,
                search_params,
                index,
                raft::make_const_mdspan(d_queries.view()),
                d_candidates.view(),
                d_cand_dists.view());
    };
    auto run_refine = [&]() {
        refine(handle,
               raft::make_const_mdspan(d_base->view()),
               raft::make_const_mdspan(d_queries.view()),
               raft::make_const_mdspan(d_candidates.view()),
               d_neighbors.view(),
               d_distances.view(),
               index.metric());
    };
    auto run_pipeline = [&]() {
        run_ann();
        if (do_refine) {
            run_refine();
        } else {
            // candidates already have size k: copy them into the final buffers for the D2H phase
            survey::cuda_check(
                    cudaMemcpyAsync(
                            d_neighbors.data_handle(),
                            d_candidates.data_handle(),
                            nq * static_cast<size_t>(k_search) * sizeof(int64_t),
                            cudaMemcpyDeviceToDevice,
                            stream),
                    "copy I");
            survey::cuda_check(
                    cudaMemcpyAsync(
                            d_distances.data_handle(),
                            d_cand_dists.data_handle(),
                            nq * static_cast<size_t>(k_search) * sizeof(float),
                            cudaMemcpyDeviceToDevice,
                            stream),
                    "copy D");
        }
    };

    // Untimed warm-up of the whole search
    run_pipeline();
    survey::cuda_check(cudaDeviceSynchronize(), "warmup sync");

    const int kSearchRepeats = env_search_repeats();
    survey::RunTimer ann_runs(kSearchRepeats, stream);
    report.phases.push_back(survey::measure_phase(
            "search_ann",
            [&]() {
                for (int r = 0; r < kSearchRepeats; r++) {
                    ann_runs.start(r);
                    run_ann();
                    survey::cuda_check(cudaDeviceSynchronize(), "ann sync");
                    ann_runs.stop(r);
                }
            },
            stream,
            true));
    double ann_wall = 0;
    {
        auto& sp = report.phases.back();
        ann_runs.finish(sp);
        ann_wall = sp.wall_s;
        report.ann_qps = (ann_wall > 0) ? (nq / ann_wall) : 0.0;
    }

    double refine_wall = 0;
    if (do_refine) {
        // Run the ANN search once more so that refine works on fresh candidates.
        run_ann();
        survey::cuda_check(cudaDeviceSynchronize(), "pre-refine ann");
        survey::RunTimer refine_runs(kSearchRepeats, stream);
        report.phases.push_back(survey::measure_phase(
                "refine",
                [&]() {
                    for (int r = 0; r < kSearchRepeats; r++) {
                        // refine-only timing: refine the same candidates in every repetition
                        refine_runs.start(r);
                        run_refine();
                        survey::cuda_check(cudaDeviceSynchronize(), "refine sync");
                        refine_runs.stop(r);
                    }
                },
                stream,
                true));
        {
            auto& sp = report.phases.back();
            refine_runs.finish(sp);
            refine_wall = sp.wall_s;
            report.refine_qps = (refine_wall > 0) ? (nq / refine_wall) : 0.0;
        }
    }

    // End to end: ANN search and refine timed together; the last run gives the results for recall
    survey::RunTimer e2e_runs(kSearchRepeats, stream);
    report.phases.push_back(survey::measure_phase(
            "search_e2e",
            [&]() {
                for (int r = 0; r < kSearchRepeats; r++) {
                    e2e_runs.start(r);
                    run_pipeline();
                    survey::cuda_check(cudaDeviceSynchronize(), "e2e sync");
                    e2e_runs.stop(r);
                }
            },
            stream,
            true));
    {
        auto& sp = report.phases.back();
        e2e_runs.finish(sp);
        report.search_qps = (sp.wall_s > 0) ? (nq / sp.wall_s) : 0.0;
        printf("---- search breakdown ----\n");
        printf("  fp16_lut            = %d\n", (int)fp16_lut);
        printf("  refine_ratio        = %.2f  (k0=%d -> k=%d)\n",
               report.refine_ratio,
               k_candidates,
               k_search);
        printf("  search_repeats      = %d\n", kSearchRepeats);
        printf("  ann_wall_s          = %.6f\n", ann_wall);
        if (do_refine) {
            printf("  refine_wall_s       = %.6f\n", refine_wall);
        }
        printf("  e2e_wall_s          = %.6f\n", sp.wall_s);
        printf("  ann_QPS             = %.1f\n", report.ann_qps);
        if (do_refine) {
            printf("  refine_QPS          = %.1f\n", report.refine_qps);
        }
        printf("  e2e_QPS             = %.1f\n", report.search_qps);
    }

    std::vector<int64_t> I(nq * static_cast<size_t>(k_search));
    std::vector<float> D(nq * static_cast<size_t>(k_search));
    report.phases.push_back(survey::measure_phase(
            "d2h_results",
            [&]() {
                survey::cuda_check(
                        cudaMemcpyAsync(
                                I.data(),
                                d_neighbors.data_handle(),
                                I.size() * sizeof(int64_t),
                                cudaMemcpyDeviceToHost,
                                stream),
                        "D2H I");
                survey::cuda_check(
                        cudaMemcpyAsync(
                                D.data(),
                                d_distances.data_handle(),
                                D.size() * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                stream),
                        "D2H D");
                survey::cuda_check(cudaStreamSynchronize(stream), "D2H sync");
            },
            stream,
            true));

    double t_lab = survey::timed_memcpy_d2h(
            I.data(), d_neighbors.data_handle(), I.size() * sizeof(int64_t), stream);
    double t_dis = survey::timed_memcpy_d2h(
            D.data(), d_distances.data_handle(), D.size() * sizeof(float), stream);
    report.d2h_result_s = t_lab + t_dis;

    report.recall_at_1 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 1);
    report.recall_at_10 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 10);
    report.recall_at_100 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 100);

    delete[] xq;
    delete[] xb;
    finalize_report(report, json_out);
    return 0;
}

int cmd_all(int argc, char** argv) {
    // Build in memory, then search (no index file). Arguments as in faiss_ivf_pq.
    const char* data_dir = argv[2];
    int nlist = argc > 3 ? atoi(argv[3]) : 4096;
    int m = argc > 4 ? atoi(argv[4]) : 16;
    int nprobe = argc > 5 ? atoi(argv[5]) : 64;
    int k_search = argc > 6 ? atoi(argv[6]) : 100;
    const char* json_out = argc > 7 ? argv[7] : "survey_metrics_all.json";
    const int nbits = 8;
    const int device = 0;

    survey::SurveyReport report;
    report.nlist = nlist;
    report.m = m;
    report.nbits = nbits;
    report.nprobe = nprobe;
    report.k = k_search;
    report.device = device;
    report.use_fp16_lut = env_fp16_lut();
    report.refine_ratio = env_refine_ratio() > 1.f ? env_refine_ratio() : 0.f;

    survey::cuda_check(cudaSetDevice(device), "cudaSetDevice");
    {
        auto g = survey::gpu_mem_snap();
        report.gpu_total_bytes = g.total_bytes;
    }

    raft::device_resources handle;
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
            rmm::mr::get_current_device_resource(), 1024ull * 1024ull * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);
    cudaStream_t stream = raft_stream(handle);

    size_t d = 0, nt = 0, nb = 0, nq = 0, gt_k = 0;
    float* xt = nullptr;
    float* xb = nullptr;
    float* xq = nullptr;
    int* gt_int = nullptr;

    const bool use_cosine = env_use_cosine();
    const auto metric = env_cuvs_metric();
    const dsio::DatasetPaths paths = dsio::resolve_or_die(data_dir, /*need_gt=*/true, /*need_learn=*/true);

    report.phases.push_back(survey::measure_phase(
            "load_dataset_host",
            [&]() {
                dsio::load_base_and_learn(paths, &xb, &nb, &xt, &nt, &d);
                dsio::load_queries_and_gt(paths, &xq, &nq, &gt_int, &gt_k, &d);
                if (use_cosine) {
                    l2_normalize_rows(xt, nt, d);
                    l2_normalize_rows(xb, nb, d);
                    l2_normalize_rows(xq, nq, d);
                }
            },
            nullptr,
            false));

    std::vector<int64_t> gt(nq * gt_k);
    for (size_t i = 0; i < nq * gt_k; i++) {
        gt[i] = gt_int[i];
    }
    delete[] gt_int;
    gt_int = nullptr;

    report.d = d;
    report.nb = nb;
    report.nq = nq;
    report.nt = nt;

    printf("ALL d=%zu nb=%zu nq=%zu nt=%zu | nlist=%d m=%d nbits=%d nprobe=%d k=%d metric=%s\n",
           d,
           nb,
           nq,
           nt,
           nlist,
           m,
           nbits,
           nprobe,
           k_search,
           use_cosine ? "IP(cosine)" : "L2");

    if (d % static_cast<size_t>(m) != 0) {
        fprintf(stderr, "dimension %zu must be divisible by m=%d\n", d, m);
        return 1;
    }

    auto d_train = raft::make_device_matrix<float, int64_t>(handle, static_cast<int64_t>(nt), static_cast<int64_t>(d));
    auto d_base = raft::make_device_matrix<float, int64_t>(handle, static_cast<int64_t>(nb), static_cast<int64_t>(d));
    auto d_queries =
            raft::make_device_matrix<float, int64_t>(handle, static_cast<int64_t>(nq), static_cast<int64_t>(d));

    report.h2d_base_bytes = nb * d * sizeof(float);
    report.h2d_query_bytes = nq * d * sizeof(float);
    report.d2h_result_bytes =
            nq * static_cast<size_t>(k_search) * (sizeof(int64_t) + sizeof(float));

    report.phases.push_back(survey::measure_phase(
            "h2d_all_inputs",
            [&]() {
                survey::cuda_check(
                        cudaMemcpyAsync(d_train.data_handle(), xt, nt * d * sizeof(float), cudaMemcpyHostToDevice, stream),
                        "H2D train");
                survey::cuda_check(
                        cudaMemcpyAsync(d_base.data_handle(), xb, nb * d * sizeof(float), cudaMemcpyHostToDevice, stream),
                        "H2D base");
                survey::cuda_check(
                        cudaMemcpyAsync(d_queries.data_handle(), xq, nq * d * sizeof(float), cudaMemcpyHostToDevice, stream),
                        "H2D queries");
                survey::cuda_check(cudaStreamSynchronize(stream), "H2D sync");
            },
            stream,
            true));

    report.h2d_base_s =
            survey::timed_memcpy_h2d(d_base.data_handle(), xb, report.h2d_base_bytes, stream);
    report.h2d_query_s =
            survey::timed_memcpy_h2d(d_queries.data_handle(), xq, report.h2d_query_bytes, stream);

    delete[] xt;
    xt = nullptr;
    delete[] xb;
    xb = nullptr;

    using namespace cuvs::neighbors;
    ivf_pq::index_params index_params;
    index_params.n_lists = static_cast<uint32_t>(nlist);
    index_params.metric = metric;
    index_params.pq_bits = static_cast<uint32_t>(nbits);
    index_params.pq_dim = static_cast<uint32_t>(m);
    index_params.kmeans_trainset_fraction = 1.0;
    index_params.add_data_on_build = false;

    std::optional<ivf_pq::index<int64_t>> index;
    report.phases.push_back(survey::measure_phase(
            "build_train",
            [&]() {
                index = ivf_pq::build(
                        handle, index_params, raft::make_const_mdspan(d_train.view()));
            },
            stream,
            true));
    report.phases.push_back(survey::measure_phase(
            "build_extend_add",
            [&]() {
                std::optional<raft::device_vector_view<const int64_t, int64_t>> no_ids =
                        std::nullopt;
                ivf_pq::extend(
                        handle,
                        raft::make_const_mdspan(d_base.view()),
                        no_ids,
                        &index.value());
            },
            stream,
            true));

    auto d_neighbors = raft::make_device_matrix<int64_t, int64_t>(
            handle, static_cast<int64_t>(nq), static_cast<int64_t>(k_search));
    auto d_distances = raft::make_device_matrix<float, int64_t>(
            handle, static_cast<int64_t>(nq), static_cast<int64_t>(k_search));

    ivf_pq::search_params search_params;
    search_params.n_probes = static_cast<uint32_t>(nprobe);
    if (report.use_fp16_lut) {
        search_params.lut_dtype = CUDA_R_16F;
        search_params.internal_distance_dtype = CUDA_R_16F;
    } else {
        search_params.lut_dtype = CUDA_R_32F;
        search_params.internal_distance_dtype = CUDA_R_32F;
    }

    // Untimed warm-up of all queries, then the timed searches; recall from the last one
    {
        ivf_pq::search(
                handle,
                search_params,
                index.value(),
                raft::make_const_mdspan(d_queries.view()),
                d_neighbors.view(),
                d_distances.view());
        survey::cuda_check(cudaDeviceSynchronize(), "warmup sync");
    }

    const int kSearchRepeats = env_search_repeats();
    survey::RunTimer runs(kSearchRepeats, stream);
    report.phases.push_back(survey::measure_phase(
            "search",
            [&]() {
                for (int r = 0; r < kSearchRepeats; r++) {
                    runs.start(r);
                    ivf_pq::search(
                            handle,
                            search_params,
                            index.value(),
                            raft::make_const_mdspan(d_queries.view()),
                            d_neighbors.view(),
                            d_distances.view());
                    survey::cuda_check(cudaDeviceSynchronize(), "search sync");
                    runs.stop(r);
                }
            },
            stream,
            true));
    runs.finish(report.phases.back());

    std::vector<int64_t> I(nq * static_cast<size_t>(k_search));
    std::vector<float> D(nq * static_cast<size_t>(k_search));
    double t_lab = survey::timed_memcpy_d2h(
            I.data(), d_neighbors.data_handle(), I.size() * sizeof(int64_t), stream);
    double t_dis = survey::timed_memcpy_d2h(
            D.data(), d_distances.data_handle(), D.size() * sizeof(float), stream);
    report.d2h_result_s = t_lab + t_dis;

    {
        const auto& sp = report.phases.back();
        report.search_qps = (sp.wall_s > 0) ? (nq / sp.wall_s) : 0.0;
        printf("---- search breakdown (approx) ----\n");
        printf("  search_repeats      = %d\n", kSearchRepeats);
        printf("  search_wall_s       = %.6f  # mean over repeats\n", sp.wall_s);
        double cuda_s = (sp.cuda_event_ms >= 0) ? sp.cuda_event_ms / 1000.0 : sp.wall_s;
        printf("  search_cuda_event_s = %.6f  # mean over repeats\n", cuda_s);
    }

    report.recall_at_1 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 1);
    report.recall_at_10 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 10);
    report.recall_at_100 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 100);

    delete[] xq;
    finalize_report(report, json_out);
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        usage(argv[0]);
        return 1;
    }
    const char* mode = argv[1];
    if (strcmp(mode, "build") == 0) {
        return cmd_build(argc, argv);
    }
    if (strcmp(mode, "search") == 0) {
        return cmd_search(argc, argv);
    }
    if (strcmp(mode, "all") == 0) {
        return cmd_all(argc, argv);
    }
    usage(argv[0]);
    return 1;
}
