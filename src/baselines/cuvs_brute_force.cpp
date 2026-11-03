/*
 * Exact k-NN on the GPU with cuVS (cuvs::neighbors::brute_force), with per-phase timing and
 * memory metrics.
 *
 *   cuvs_brute_force build  <data_dir> [index_path] [json_out]
 *   cuvs_brute_force search <data_dir> [index_path] [k] [json_out]
 *   cuvs_brute_force all    <data_dir> [k] [json_out]
 *
 * Defaults:
 *   index_path=cuvs_brute_force.cuvsindex  k=100
 *   json_out=survey_metrics_{build|search|all}.json
 *
 * Metric: L2Expanded by default. IVF_METRIC=cosine|angular|ip: normalize and use InnerProduct.
 * Search: one untimed warm-up search of all queries on device pointers, then kTimedRuns (3)
 * timed searches of all queries; the search phase reports the mean wall and CUDA-event time per
 * search, their sample standard deviations and the number of timed searches;
 * search_qps = nq / mean wall time. Recall is computed from the results of the last timed search.
 * WRITE_GT=<path>: also write the returned ids as an ibin ground-truth file; without an input
 * ground truth the recall is reported as 1.0.
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
#include <cuvs/neighbors/brute_force.hpp>

#include <rmm/mr/device/device_memory_resource.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include "survey_profiler.hpp"
#include "dataset_io.hpp"
#include "recall.hpp"

namespace {

// Timed searches after the untimed warm-up
constexpr int kTimedRuns = 3;

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

using dsio::join_path;

void usage(const char* argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s build  <data_dir> [index_path] [json_out]\n"
            "  %s search <data_dir> [index_path] [k] [json_out]\n"
            "  %s all    <data_dir> [k] [json_out]\n",
            argv0,
            argv0,
            argv0);
}

cudaStream_t raft_stream(raft::device_resources& handle) {
    return static_cast<cudaStream_t>(raft::resource::get_cuda_stream(handle).value());
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

/** Recall@1, @10 and @100 as defined in src/common/recall.hpp. */
void fill_recall(survey::SurveyReport& report,
                 const std::vector<int64_t>& I,
                 const std::vector<int64_t>& gt,
                 size_t nq,
                 size_t gt_k,
                 int k_search) {
    report.recall_at_1 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 1);
    report.recall_at_10 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 10);
    report.recall_at_100 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 100);
}

int cmd_build(int argc, char** argv) {
    const char* data_dir = argv[2];
    const char* index_path = argc > 3 ? argv[3] : "cuvs_brute_force.cuvsindex";
    const char* json_out = argc > 4 ? argv[4] : "survey_metrics_build.json";
    const int device = 0;

    survey::SurveyReport report;
    report.device = device;
    report.use_fp16_lut = false;
    report.nlist = 0;
    report.m = 0;
    report.nbits = 0;
    report.nprobe = 0;
    report.k = 0;

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
            rmm::mr::get_current_device_resource(), 2ull * 1024ull * 1024ull * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);
    cudaStream_t stream = raft_stream(handle);
    printf("cuVS/raft stream = %p\n", (void*)stream);

    size_t d = 0, nb = 0;
    float* xb = nullptr;
    const bool use_cosine = env_use_cosine();
    const auto metric = env_cuvs_metric();
    const dsio::DatasetPaths paths = dsio::resolve_or_die(data_dir, /*need_gt=*/false, /*need_learn=*/false);

    report.phases.push_back(survey::measure_phase(
            "load_dataset_host",
            [&]() {
                xb = dsio::floats_read(paths.base, &d, &nb);
                if (use_cosine) {
                    l2_normalize_rows(xb, nb, d);
                }
            },
            nullptr,
            false));

    report.d = d;
    report.nb = nb;
    report.nq = 0;
    report.nt = 0;

    printf("BUILD d=%zu nb=%zu | brute_force %s -> %s\n",
           d,
           nb,
           use_cosine ? "IP(cosine)" : "L2Expanded",
           index_path);

    auto d_base =
            raft::make_device_matrix<float, int64_t>(handle, static_cast<int64_t>(nb), static_cast<int64_t>(d));
    report.h2d_base_bytes = nb * d * sizeof(float);
    report.h2d_query_bytes = 0;
    report.d2h_result_bytes = 0;

    report.phases.push_back(survey::measure_phase(
            "h2d_base",
            [&]() {
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

    report.h2d_base_s =
            survey::timed_memcpy_h2d(d_base.data_handle(), xb, report.h2d_base_bytes, stream);

    delete[] xb;
    xb = nullptr;

    using namespace cuvs::neighbors;
    brute_force::index_params index_params;
    index_params.metric = metric;

    std::optional<brute_force::index<float, float>> index;
    report.phases.push_back(survey::measure_phase(
            "build_index",
            [&]() {
                index = brute_force::build(
                        handle, index_params, raft::make_const_mdspan(d_base.view()));
            },
            stream,
            true));

    report.phases.push_back(survey::measure_phase(
            "serialize",
            [&]() {
                brute_force::serialize(handle, std::string(index_path), index.value(), true);
            },
            stream,
            true));

    finalize_report(report, json_out);
    return 0;
}

int cmd_search(int argc, char** argv) {
    const char* data_dir = argv[2];
    const char* index_path = argc > 3 ? argv[3] : "cuvs_brute_force.cuvsindex";
    int k_search = argc > 4 ? atoi(argv[4]) : 100;
    const char* json_out = argc > 5 ? argv[5] : "survey_metrics_search.json";
    const int device = 0;

    survey::SurveyReport report;
    report.k = k_search;
    report.device = device;
    report.use_fp16_lut = false;

    survey::cuda_check(cudaSetDevice(device), "cudaSetDevice");
    {
        auto g = survey::gpu_mem_snap();
        report.gpu_total_bytes = g.total_bytes;
    }

    raft::device_resources handle;
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
            rmm::mr::get_current_device_resource(), 2ull * 1024ull * 1024ull * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);
    cudaStream_t stream = raft_stream(handle);

    size_t d = 0, nq = 0, gt_k = 0;
    float* xq = nullptr;
    int* gt_int = nullptr;
    const char* write_gt = dsio::env_write_gt();
    const dsio::DatasetPaths paths = dsio::resolve_or_die(data_dir, /*need_gt=*/true, /*need_learn=*/false);

    report.phases.push_back(survey::measure_phase(
            "load_queries_host",
            [&]() {
                dsio::load_queries_and_gt(
                        paths, &xq, &nq, &gt_int, &gt_k, &d, /*require_gt=*/write_gt == nullptr);
            },
            nullptr,
            false));

    std::vector<int64_t> gt;
    if (gt_int) {
        gt.resize(nq * gt_k);
        for (size_t i = 0; i < nq * gt_k; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
        gt_int = nullptr;
    }

    report.d = d;
    report.nq = nq;

    using namespace cuvs::neighbors;
    brute_force::index<float, float> index(handle);
    report.phases.push_back(survey::measure_phase(
            "deserialize_index",
            [&]() { brute_force::deserialize(handle, std::string(index_path), &index); },
            stream,
            true));

    if (is_ip_metric(index.metric())) {
        l2_normalize_rows(xq, nq, d);
    }

    report.nb = index.size();
    report.d = index.dim();

    printf("SEARCH d=%zu nq=%zu | loaded size=%zu dim=%zu | k=%d\n",
           d,
           nq,
           report.nb,
           index.dim(),
           k_search);

    auto d_queries =
            raft::make_device_matrix<float, int64_t>(handle, static_cast<int64_t>(nq), static_cast<int64_t>(d));
    auto d_neighbors = raft::make_device_matrix<int64_t, int64_t>(
            handle, static_cast<int64_t>(nq), static_cast<int64_t>(k_search));
    auto d_distances = raft::make_device_matrix<float, int64_t>(
            handle, static_cast<int64_t>(nq), static_cast<int64_t>(k_search));

    report.h2d_query_bytes = nq * d * sizeof(float);
    report.d2h_result_bytes =
            nq * static_cast<size_t>(k_search) * (sizeof(int64_t) + sizeof(float));
    report.h2d_base_bytes = 0;

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

    brute_force::search_params search_params;

    auto run_search = [&]() {
        brute_force::search(
                handle,
                search_params,
                index,
                raft::make_const_mdspan(d_queries.view()),
                d_neighbors.view(),
                d_distances.view());
        survey::cuda_check(cudaStreamSynchronize(stream), "search sync");
    };

    // Untimed warm-up of all queries, then the timed searches; recall from the last one
    run_search();
    survey::cuda_check(cudaDeviceSynchronize(), "warmup sync");

    survey::RunTimer runs(kTimedRuns, stream);
    report.phases.push_back(survey::measure_phase(
            "search",
            [&]() {
                for (int r = 0; r < kTimedRuns; r++) {
                    runs.start(r);
                    run_search();
                    runs.stop(r);
                }
            },
            stream,
            true));
    runs.finish(report.phases.back());

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

    {
        const auto& sp = report.phases[report.phases.size() - 2];  // search
        report.search_qps = (sp.wall_s > 0) ? (nq / sp.wall_s) : 0.0;
        double xfer_est = report.h2d_query_s + report.d2h_result_s;
        double wall = sp.wall_s;
        double cuda_s = (sp.cuda_event_ms >= 0) ? sp.cuda_event_ms / 1000.0 : wall;
        printf("---- search breakdown (approx) ----\n");
        printf("  search_wall_s       = %.6f\n", wall);
        printf("  search_cuda_event_s = %.6f\n", cuda_s);
        printf("  est_payload_xfer_s  = %.6f\n", xfer_est);
        printf("  est_compute_proxy_s = %.6f\n", std::max(0.0, cuda_s - xfer_est));
    }

    if (!gt.empty()) {
        fill_recall(report, I, gt, nq, gt_k, k_search);
    } else {
        report.recall_at_1 = 1.f;
        report.recall_at_10 = 1.f;
        report.recall_at_100 = 1.f;
        printf("note: no input GT (WRITE_GT mode); recall set to 1.0 by definition\n");
    }

    if (write_gt) {
        std::vector<int> gt32(nq * static_cast<size_t>(k_search));
        for (size_t i = 0; i < gt32.size(); i++) {
            gt32[i] = static_cast<int>(I[i]);
        }
        dsio::write_ibin(write_gt, gt32.data(), nq, static_cast<size_t>(k_search));
    }

    delete[] xq;
    finalize_report(report, json_out);
    return 0;
}

int cmd_all(int argc, char** argv) {
    const char* data_dir = argv[2];
    int k_search = argc > 3 ? atoi(argv[3]) : 100;
    const char* json_out = argc > 4 ? argv[4] : "survey_metrics_all.json";
    const int device = 0;

    survey::SurveyReport report;
    report.k = k_search;
    report.device = device;
    report.use_fp16_lut = false;

    survey::cuda_check(cudaSetDevice(device), "cudaSetDevice");
    {
        auto g = survey::gpu_mem_snap();
        report.gpu_total_bytes = g.total_bytes;
    }

    raft::device_resources handle;
    rmm::mr::pool_memory_resource<rmm::mr::device_memory_resource> pool_mr(
            rmm::mr::get_current_device_resource(), 2ull * 1024ull * 1024ull * 1024ull);
    rmm::mr::set_current_device_resource(&pool_mr);
    cudaStream_t stream = raft_stream(handle);

    size_t d = 0, nb = 0, nq = 0, gt_k = 0;
    float* xb = nullptr;
    float* xq = nullptr;
    int* gt_int = nullptr;
    const bool use_cosine = env_use_cosine();
    const auto metric = env_cuvs_metric();
    const char* write_gt = dsio::env_write_gt();
    const dsio::DatasetPaths paths = dsio::resolve_or_die(data_dir, /*need_gt=*/true, /*need_learn=*/false);

    report.phases.push_back(survey::measure_phase(
            "load_dataset_host",
            [&]() {
                xb = dsio::floats_read(paths.base, &d, &nb);
                dsio::load_queries_and_gt(
                        paths, &xq, &nq, &gt_int, &gt_k, &d, /*require_gt=*/write_gt == nullptr);
                if (use_cosine) {
                    l2_normalize_rows(xb, nb, d);
                    l2_normalize_rows(xq, nq, d);
                }
            },
            nullptr,
            false));

    std::vector<int64_t> gt;
    if (gt_int) {
        gt.resize(nq * gt_k);
        for (size_t i = 0; i < nq * gt_k; i++) {
            gt[i] = gt_int[i];
        }
        delete[] gt_int;
        gt_int = nullptr;
    }

    report.d = d;
    report.nb = nb;
    report.nq = nq;

    printf("ALL d=%zu nb=%zu nq=%zu | brute_force %s k=%d\n",
           d,
           nb,
           nq,
           use_cosine ? "IP(cosine)" : "L2Expanded",
           k_search);

    auto d_base =
            raft::make_device_matrix<float, int64_t>(handle, static_cast<int64_t>(nb), static_cast<int64_t>(d));
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
                        cudaMemcpyAsync(
                                d_base.data_handle(),
                                xb,
                                nb * d * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream),
                        "H2D base");
                survey::cuda_check(
                        cudaMemcpyAsync(
                                d_queries.data_handle(),
                                xq,
                                nq * d * sizeof(float),
                                cudaMemcpyHostToDevice,
                                stream),
                        "H2D queries");
                survey::cuda_check(cudaStreamSynchronize(stream), "H2D sync");
            },
            stream,
            true));

    report.h2d_base_s =
            survey::timed_memcpy_h2d(d_base.data_handle(), xb, report.h2d_base_bytes, stream);
    report.h2d_query_s =
            survey::timed_memcpy_h2d(d_queries.data_handle(), xq, report.h2d_query_bytes, stream);

    using namespace cuvs::neighbors;
    brute_force::index_params index_params;
    index_params.metric = metric;

    std::optional<brute_force::index<float, float>> index;
    report.phases.push_back(survey::measure_phase(
            "build_index",
            [&]() {
                index = brute_force::build(
                        handle, index_params, raft::make_const_mdspan(d_base.view()));
            },
            stream,
            true));

    auto d_neighbors = raft::make_device_matrix<int64_t, int64_t>(
            handle, static_cast<int64_t>(nq), static_cast<int64_t>(k_search));
    auto d_distances = raft::make_device_matrix<float, int64_t>(
            handle, static_cast<int64_t>(nq), static_cast<int64_t>(k_search));

    brute_force::search_params search_params;

    auto run_search = [&]() {
        brute_force::search(
                handle,
                search_params,
                index.value(),
                raft::make_const_mdspan(d_queries.view()),
                d_neighbors.view(),
                d_distances.view());
        survey::cuda_check(cudaStreamSynchronize(stream), "search sync");
    };

    // Untimed warm-up of all queries, then the timed searches; recall from the last one
    run_search();
    survey::cuda_check(cudaDeviceSynchronize(), "warmup sync");

    survey::RunTimer runs(kTimedRuns, stream);
    report.phases.push_back(survey::measure_phase(
            "search",
            [&]() {
                for (int r = 0; r < kTimedRuns; r++) {
                    runs.start(r);
                    run_search();
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
    }

    if (!gt.empty()) {
        fill_recall(report, I, gt, nq, gt_k, k_search);
    } else {
        report.recall_at_1 = 1.f;
        report.recall_at_10 = 1.f;
        report.recall_at_100 = 1.f;
        printf("note: no input GT (WRITE_GT mode); recall set to 1.0 by definition\n");
    }

    if (write_gt) {
        std::vector<int> gt32(nq * static_cast<size_t>(k_search));
        for (size_t i = 0; i < gt32.size(); i++) {
            gt32[i] = static_cast<int>(I[i]);
        }
        dsio::write_ibin(write_gt, gt32.data(), nq, static_cast<size_t>(k_search));
    }

    delete[] xq;
    delete[] xb;
    finalize_report(report, json_out);
    return 0;
}

}  // namespace

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
