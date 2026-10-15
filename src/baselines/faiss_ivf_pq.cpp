/*
 * IVF-PQ on the GPU with Faiss (GpuIndexIVFPQ), with per-phase timing and memory metrics.
 *
 * Build and search are separate modes; the index is handed over as a CPU index file:
 *
 *   faiss_ivf_pq build  <data_dir> [nlist] [m] [index_path] [json_out]
 *   faiss_ivf_pq search <data_dir> [index_path] [nprobe] [k] [json_out]
 *   faiss_ivf_pq all    <data_dir> [nlist] [m] [nprobe] [k] [json_out]
 *
 * Defaults:
 *   nlist=4096 m=16 nprobe=64 k=100 (nbits=8, float16 lookup tables, device 0)
 *   index_path=faiss_ivf_pq.faissindex
 *   json_out=survey_metrics_{build|search|all}.json
 *
 * The index is trained on learn.fbin if the dataset has one, else on the first
 * LEARN_RATIO (default 0.1) of the base vectors.
 *
 * Build metrics: load / train / add wall time and CUDA events, host and GPU peaks,
 * Faiss allocator info, H2D microbenchmark, GPU-to-CPU export and write.
 * Search metrics: load index, search, recall, transfer microbenchmark.
 *
 * Search: queries are copied to the device first; one untimed warm-up search() of all
 * queries on device pointers, then the search phase calls it SEARCH_REPEATS times (default 3)
 * and reports the mean wall and CUDA-event time per call, their sample standard deviations and
 * the number of timed calls. search_qps = nq / mean wall time of one call. Recall is computed
 * from the results of the last timed call.
 * Recall: R@x is the fraction of queries whose nearest ground-truth neighbor is among
 * the first x returned ids.
 */

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <strings.h>
#include <sys/stat.h>
#include <vector>

#include <cuda_runtime.h>

#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>
#include <faiss/index_io.h>
#include <faiss/MetricType.h>

#include "survey_profiler.hpp"
#include "dataset_io.hpp"
#include "recall.hpp"

namespace {

// Timed searches per setting after the warm-up (SEARCH_REPEATS, default 3)
int env_search_repeats() {
    const char* v = getenv("SEARCH_REPEATS");
    if (!v || !*v) {
        return 3;
    }
    int r = atoi(v);
    return r > 0 ? r : 3;
}

// IVF_METRIC=cosine|angular|ip|inner_product: L2-normalize and use INNER_PRODUCT
// (for angular datasets). Unset: L2.
bool env_use_cosine() {
    const char* v = getenv("IVF_METRIC");
    if (!v || !*v) {
        return false;
    }
    return strcasecmp(v, "cosine") == 0 || strcasecmp(v, "angular") == 0 ||
            strcasecmp(v, "ip") == 0 || strcasecmp(v, "inner_product") == 0;
}

faiss::MetricType env_faiss_metric() {
    return env_use_cosine() ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;
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

void print_faiss_memory_info(faiss::gpu::StandardGpuResources& res, int device) {
    auto info = res.getMemoryInfo();
    printf("---- FAISS StandardGpuResources::getMemoryInfo (device %d) ----\n", device);
    auto it = info.find(device);
    if (it == info.end()) {
        printf("  (no entries)\n");
        return;
    }
    for (const auto& kv : it->second) {
        printf("  %-32s  allocs=%d  size=%zu (%.3f MiB)\n",
               kv.first.c_str(),
               kv.second.first,
               kv.second.second,
               kv.second.second / (1024.0 * 1024.0));
    }
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

void print_index_size_estimate(size_t nb, size_t d, int nlist, int m) {
    size_t pq_codes_bytes = nb * static_cast<size_t>(m);
    size_t centroids_bytes = static_cast<size_t>(nlist) * d * sizeof(float);
    printf("---- index size estimate (analytical) ----\n");
    printf("  PQ codes ~ %zu B (%.3f MiB)\n",
           pq_codes_bytes,
           pq_codes_bytes / (1024.0 * 1024.0));
    printf("  IVF centroids ~ %zu B (%.3f MiB)\n",
           centroids_bytes,
           centroids_bytes / (1024.0 * 1024.0));
}

int cmd_build(int argc, char** argv) {
    const char* data_dir = argv[2];
    int nlist = argc > 3 ? atoi(argv[3]) : 4096;
    int m = argc > 4 ? atoi(argv[4]) : 16;
    const char* index_path = argc > 5 ? argv[5] : "faiss_ivf_pq.faissindex";
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
    report.use_fp16_lut = true;

    survey::cuda_check(cudaSetDevice(device), "cudaSetDevice");
    {
        auto g = survey::gpu_mem_snap();
        report.gpu_total_bytes = g.total_bytes;
        printf("GPU total memory: %.3f GiB (used now %.3f GiB)\n",
               g.total_bytes / (1024.0 * 1024.0 * 1024.0),
               g.used_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    size_t d = 0, nt = 0, nb = 0;
    float* xt = nullptr;
    float* xb = nullptr;

    const bool use_cosine = env_use_cosine();
    const faiss::MetricType metric = env_faiss_metric();
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

    printf("BUILD d=%zu nb=%zu nt=%zu | IVF-PQ nlist=%d m=%d nbits=%d metric=%s -> %s\n",
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

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = device;
    config.useFloat16LookupTables = true;

    faiss::gpu::GpuIndexIVFPQ index(
            &res,
            static_cast<int>(d),
            nlist,
            m,
            nbits,
            metric,
            config);
    index.verbose = false;

    cudaStream_t faiss_stream = res.getDefaultStream(device);
    printf("FAISS default stream = %p\n", (void*)faiss_stream);

    // Standalone H2D microbenchmark of the train and base payloads (before Faiss uses the host pointers)
    report.h2d_base_bytes = nb * d * sizeof(float);
    report.h2d_query_bytes = nt * d * sizeof(float); // train set as the reference payload
    report.d2h_result_bytes = 0;
    {
        float* d_train = nullptr;
        float* d_base = nullptr;
        survey::cuda_check(cudaMalloc(&d_train, report.h2d_query_bytes), "cudaMalloc train");
        survey::cuda_check(cudaMalloc(&d_base, report.h2d_base_bytes), "cudaMalloc base");
        survey::cuda_check(
                cudaMemcpy(d_train, xt, d * sizeof(float), cudaMemcpyHostToDevice), "warmup");
        report.h2d_query_s =
                survey::timed_memcpy_h2d(d_train, xt, report.h2d_query_bytes, faiss_stream);
        report.h2d_base_s =
                survey::timed_memcpy_h2d(d_base, xb, report.h2d_base_bytes, faiss_stream);
        cudaFree(d_train);
        cudaFree(d_base);
    }

    report.phases.push_back(survey::measure_phase(
            "build_train",
            [&]() { index.train(nt, xt); },
            faiss_stream,
            true));
    delete[] xt;
    xt = nullptr;

    report.phases.push_back(survey::measure_phase(
            "build_add",
            [&]() { index.add(nb, xb); },
            faiss_stream,
            true));
    delete[] xb;
    xb = nullptr;

    print_faiss_memory_info(res, device);
    print_index_size_estimate(nb, d, nlist, m);

    // GPU memory right after the build (resident index and Faiss temporary buffers)
    {
        auto g = survey::gpu_mem_snap();
        char buf[64];
        printf("---- post-build resident ----\n");
        printf("  gpu_used_now = %s (%zu B)\n",
               survey::bytes_human(g.used_bytes, buf, sizeof buf),
               g.used_bytes);
        printf("  host_rss_now = %s\n",
               survey::bytes_human(survey::host_mem_snap().vm_rss_bytes, buf, sizeof buf));
        printf("  host_VmHWM   = %s\n",
               survey::bytes_human(survey::host_mem_snap().vm_hwm_bytes, buf, sizeof buf));
        printf("  index.ntotal = %lld\n", static_cast<long long>(index.ntotal));
    }

    report.phases.push_back(survey::measure_phase(
            "export_index_gpu_to_cpu_write",
            [&]() {
                faiss::Index* cpu_index = faiss::gpu::index_gpu_to_cpu(&index);
                faiss::write_index(cpu_index, index_path);
                delete cpu_index;
            },
            faiss_stream,
            true));

    finalize_report(report, json_out);
    return 0;
}

int cmd_search(int argc, char** argv) {
    const char* data_dir = argv[2];
    const char* index_path = argc > 3 ? argv[3] : "faiss_ivf_pq.faissindex";
    int nprobe = argc > 4 ? atoi(argv[4]) : 64;
    int k_search = argc > 5 ? atoi(argv[5]) : 100;
    const char* json_out = argc > 6 ? argv[6] : "survey_metrics_search.json";
    const int device = 0;

    survey::SurveyReport report;
    report.nprobe = nprobe;
    report.k = k_search;
    report.device = device;
    report.use_fp16_lut = true;
    report.nbits = 8;

    survey::cuda_check(cudaSetDevice(device), "cudaSetDevice");
    {
        auto g = survey::gpu_mem_snap();
        report.gpu_total_bytes = g.total_bytes;
        printf("GPU total memory: %.3f GiB (used now %.3f GiB)\n",
               g.total_bytes / (1024.0 * 1024.0 * 1024.0),
               g.used_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    size_t d = 0, nq = 0, gt_k = 0;
    float* xq = nullptr;
    int* gt_int = nullptr;
    const dsio::DatasetPaths paths = dsio::resolve_or_die(data_dir, /*need_gt=*/true, /*need_learn=*/false);

    report.phases.push_back(survey::measure_phase(
            "load_queries_host",
            [&]() { dsio::load_queries_and_gt(paths, &xq, &nq, &gt_int, &gt_k, &d); },
            nullptr,
            false));

    std::vector<faiss::idx_t> gt(nq * gt_k);
    for (size_t i = 0; i < nq * gt_k; i++) {
        gt[i] = gt_int[i];
    }
    delete[] gt_int;
    gt_int = nullptr;

    report.d = d;
    report.nq = nq;

    faiss::gpu::StandardGpuResources res;
    cudaStream_t faiss_stream = res.getDefaultStream(device);

    faiss::Index* cpu_index = nullptr;
    faiss::Index* gpu_index = nullptr;
    report.phases.push_back(survey::measure_phase(
            "load_index_cpu_to_gpu",
            [&]() {
                cpu_index = faiss::read_index(index_path);
                gpu_index = faiss::gpu::index_cpu_to_gpu(&res, device, cpu_index);
                delete cpu_index;
                cpu_index = nullptr;
            },
            faiss_stream,
            true));

    auto* index = dynamic_cast<faiss::gpu::GpuIndexIVFPQ*>(gpu_index);
    if (!index) {
        fprintf(stderr, "loaded index is not GpuIndexIVFPQ\n");
        delete gpu_index;
        return 1;
    }

    // Match build-time cosine: IP indexes expect L2-normalized queries.
    if (index->metric_type == faiss::METRIC_INNER_PRODUCT) {
        l2_normalize_rows(xq, nq, d);
    }

    report.nlist = index->nlist;
    report.m = index->pq.M;
    report.nbits = index->pq.nbits;
    report.nb = static_cast<size_t>(index->ntotal);
    report.use_fp16_lut = true;

    printf("SEARCH d=%zu nq=%zu | loaded nlist=%d m=%d nbits=%d ntotal=%zu metric=%s | nprobe=%d k=%d\n",
           d,
           nq,
           report.nlist,
           report.m,
           report.nbits,
           report.nb,
           index->metric_type == faiss::METRIC_INNER_PRODUCT ? "IP(cosine)" : "L2",
           nprobe,
           k_search);

    print_faiss_memory_info(res, device);

    // As in the cuVS program: H2D, device search and D2H are separate timed phases.
    // GpuIndex::search accepts device pointers and then does no host copies.
    report.h2d_base_bytes = 0;
    report.h2d_query_bytes = nq * d * sizeof(float);
    report.d2h_result_bytes =
            nq * static_cast<size_t>(k_search) * (sizeof(faiss::idx_t) + sizeof(float));

    float* d_queries = nullptr;
    faiss::idx_t* d_labels = nullptr;
    float* d_dists = nullptr;
    survey::cuda_check(cudaMalloc(&d_queries, report.h2d_query_bytes), "cudaMalloc q");
    survey::cuda_check(
            cudaMalloc(&d_labels, nq * k_search * sizeof(faiss::idx_t)), "cudaMalloc I");
    survey::cuda_check(cudaMalloc(&d_dists, nq * k_search * sizeof(float)), "cudaMalloc D");

    index->nprobe = nprobe;

    report.phases.push_back(survey::measure_phase(
            "h2d_queries",
            [&]() {
                survey::cuda_check(
                        cudaMemcpyAsync(
                                d_queries,
                                xq,
                                report.h2d_query_bytes,
                                cudaMemcpyHostToDevice,
                                faiss_stream),
                        "H2D queries");
                survey::cuda_check(cudaStreamSynchronize(faiss_stream), "H2D q sync");
            },
            faiss_stream,
            true));

    // Standalone memcpy microbenchmark (payload reference, not used for QPS).
    report.h2d_query_s =
            survey::timed_memcpy_h2d(d_queries, xq, report.h2d_query_bytes, faiss_stream);

    std::vector<faiss::idx_t> I(nq * static_cast<size_t>(k_search));
    std::vector<float> D(nq * static_cast<size_t>(k_search));

    // Untimed warm-up on device pointers (all queries).
    index->search(nq, d_queries, k_search, d_dists, d_labels);
    survey::cuda_check(cudaDeviceSynchronize(), "warmup sync");

    // Timed searches; the results of the last one are used for recall.
    survey::RunTimer runs(env_search_repeats(), faiss_stream);
    report.phases.push_back(survey::measure_phase(
            "search",
            [&]() {
                for (int r = 0; r < runs.runs(); r++) {
                    runs.start(r);
                    index->search(nq, d_queries, k_search, d_dists, d_labels);
                    survey::cuda_check(cudaDeviceSynchronize(), "search sync");
                    runs.stop(r);
                }
            },
            faiss_stream,
            true));
    runs.finish(report.phases.back());

    report.phases.push_back(survey::measure_phase(
            "d2h_results",
            [&]() {
                survey::cuda_check(
                        cudaMemcpyAsync(
                                I.data(),
                                d_labels,
                                I.size() * sizeof(faiss::idx_t),
                                cudaMemcpyDeviceToHost,
                                faiss_stream),
                        "D2H I");
                survey::cuda_check(
                        cudaMemcpyAsync(
                                D.data(),
                                d_dists,
                                D.size() * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                faiss_stream),
                        "D2H D");
                survey::cuda_check(cudaStreamSynchronize(faiss_stream), "D2H sync");
            },
            faiss_stream,
            true));

    {
        double t_lab = survey::timed_memcpy_d2h(
                I.data(), d_labels, I.size() * sizeof(faiss::idx_t), faiss_stream);
        double t_dis = survey::timed_memcpy_d2h(
                D.data(), d_dists, D.size() * sizeof(float), faiss_stream);
        report.d2h_result_s = t_lab + t_dis;
    }

    {
        const auto& sp = report.phases[report.phases.size() - 2]; // search
        report.search_qps = (sp.wall_s > 0) ? (nq / sp.wall_s) : 0.0;
        double xfer = report.h2d_query_s + report.d2h_result_s;
        double wall = sp.wall_s;
        double cuda_s = (sp.cuda_event_ms >= 0) ? sp.cuda_event_ms / 1000.0 : wall;
        printf("---- search breakdown (device pointers; H2D/D2H timed separately) ----\n");
        printf("  search_wall_s       = %.6f\n", wall);
        printf("  search_cuda_event_s = %.6f\n", cuda_s);
        printf("  payload_xfer_s      = %.6f  (microbench H2D+D2H)\n", xfer);
    }

    report.recall_at_1 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 1);
    report.recall_at_10 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 10);
    report.recall_at_100 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 100);

    cudaFree(d_queries);
    cudaFree(d_labels);
    cudaFree(d_dists);
    delete gpu_index;
    delete[] xq;
    finalize_report(report, json_out);
    return 0;
}

int cmd_all(int argc, char** argv) {
    // Build and search in one process with the same metrics; the index is written at the end.
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
    report.use_fp16_lut = true;

    survey::cuda_check(cudaSetDevice(device), "cudaSetDevice");
    {
        auto g = survey::gpu_mem_snap();
        report.gpu_total_bytes = g.total_bytes;
        printf("GPU total memory: %.3f GiB (used now %.3f GiB)\n",
               g.total_bytes / (1024.0 * 1024.0 * 1024.0),
               g.used_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    size_t d = 0, nt = 0, nb = 0, nq = 0, gt_k = 0;
    float* xt = nullptr;
    float* xb = nullptr;
    float* xq = nullptr;
    int* gt_int = nullptr;

    const bool use_cosine = env_use_cosine();
    const faiss::MetricType metric = env_faiss_metric();
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

    std::vector<faiss::idx_t> gt(nq * gt_k);
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

    faiss::gpu::StandardGpuResources res;
    faiss::gpu::GpuIndexIVFPQConfig config;
    config.device = device;
    config.useFloat16LookupTables = true;
    faiss::gpu::GpuIndexIVFPQ index(
            &res, static_cast<int>(d), nlist, m, nbits, metric, config);
    index.verbose = false;
    cudaStream_t faiss_stream = res.getDefaultStream(device);

    report.h2d_base_bytes = nb * d * sizeof(float);
    report.h2d_query_bytes = nq * d * sizeof(float);
    report.d2h_result_bytes =
            nq * static_cast<size_t>(k_search) * (sizeof(faiss::idx_t) + sizeof(float));
    {
        float* d_base = nullptr;
        float* d_query = nullptr;
        faiss::idx_t* d_labels = nullptr;
        float* d_dists = nullptr;
        survey::cuda_check(cudaMalloc(&d_base, report.h2d_base_bytes), "cudaMalloc base");
        survey::cuda_check(cudaMalloc(&d_query, report.h2d_query_bytes), "cudaMalloc query");
        survey::cuda_check(
                cudaMalloc(&d_labels, nq * k_search * sizeof(faiss::idx_t)), "cudaMalloc I");
        survey::cuda_check(cudaMalloc(&d_dists, nq * k_search * sizeof(float)), "cudaMalloc D");
        survey::cuda_check(
                cudaMemcpy(d_query, xq, d * sizeof(float), cudaMemcpyHostToDevice), "warmup");
        report.h2d_base_s =
                survey::timed_memcpy_h2d(d_base, xb, report.h2d_base_bytes, faiss_stream);
        report.h2d_query_s =
                survey::timed_memcpy_h2d(d_query, xq, report.h2d_query_bytes, faiss_stream);
        survey::cuda_check(cudaMemset(d_labels, 0, nq * k_search * sizeof(faiss::idx_t)), "memset");
        survey::cuda_check(cudaMemset(d_dists, 0, nq * k_search * sizeof(float)), "memset");
        std::vector<faiss::idx_t> tmpI(nq * k_search);
        std::vector<float> tmpD(nq * k_search);
        double t_lab = survey::timed_memcpy_d2h(
                tmpI.data(), d_labels, nq * k_search * sizeof(faiss::idx_t), faiss_stream);
        double t_dis = survey::timed_memcpy_d2h(
                tmpD.data(), d_dists, nq * k_search * sizeof(float), faiss_stream);
        report.d2h_result_s = t_lab + t_dis;
        cudaFree(d_base);
        cudaFree(d_query);
        cudaFree(d_labels);
        cudaFree(d_dists);
    }

    report.phases.push_back(survey::measure_phase(
            "build_train", [&]() { index.train(nt, xt); }, faiss_stream, true));
    delete[] xt;
    xt = nullptr;

    report.phases.push_back(survey::measure_phase(
            "build_add", [&]() { index.add(nb, xb); }, faiss_stream, true));
    delete[] xb;
    xb = nullptr;

    print_faiss_memory_info(res, device);
    print_index_size_estimate(nb, d, nlist, m);

    index.nprobe = nprobe;

    float* d_queries = nullptr;
    faiss::idx_t* d_labels = nullptr;
    float* d_dists = nullptr;
    survey::cuda_check(cudaMalloc(&d_queries, report.h2d_query_bytes), "cudaMalloc q");
    survey::cuda_check(
            cudaMalloc(&d_labels, nq * k_search * sizeof(faiss::idx_t)), "cudaMalloc I");
    survey::cuda_check(cudaMalloc(&d_dists, nq * k_search * sizeof(float)), "cudaMalloc D");

    report.phases.push_back(survey::measure_phase(
            "h2d_queries",
            [&]() {
                survey::cuda_check(
                        cudaMemcpyAsync(
                                d_queries,
                                xq,
                                report.h2d_query_bytes,
                                cudaMemcpyHostToDevice,
                                faiss_stream),
                        "H2D queries");
                survey::cuda_check(cudaStreamSynchronize(faiss_stream), "H2D q sync");
            },
            faiss_stream,
            true));

    std::vector<faiss::idx_t> I(nq * static_cast<size_t>(k_search));
    std::vector<float> D(nq * static_cast<size_t>(k_search));

    // Untimed warm-up of all queries, then the timed searches; recall from the last one.
    index.search(nq, d_queries, k_search, d_dists, d_labels);
    survey::cuda_check(cudaDeviceSynchronize(), "warmup sync");

    survey::RunTimer runs(env_search_repeats(), faiss_stream);
    report.phases.push_back(survey::measure_phase(
            "search",
            [&]() {
                for (int r = 0; r < runs.runs(); r++) {
                    runs.start(r);
                    index.search(nq, d_queries, k_search, d_dists, d_labels);
                    survey::cuda_check(cudaDeviceSynchronize(), "search sync");
                    runs.stop(r);
                }
            },
            faiss_stream,
            true));
    {
        auto& sp = report.phases.back();
        runs.finish(sp);
        report.search_qps = (sp.wall_s > 0) ? (nq / sp.wall_s) : 0.0;
    }

    report.phases.push_back(survey::measure_phase(
            "d2h_results",
            [&]() {
                survey::cuda_check(
                        cudaMemcpyAsync(
                                I.data(),
                                d_labels,
                                I.size() * sizeof(faiss::idx_t),
                                cudaMemcpyDeviceToHost,
                                faiss_stream),
                        "D2H I");
                survey::cuda_check(
                        cudaMemcpyAsync(
                                D.data(),
                                d_dists,
                                D.size() * sizeof(float),
                                cudaMemcpyDeviceToHost,
                                faiss_stream),
                        "D2H D");
                survey::cuda_check(cudaStreamSynchronize(faiss_stream), "D2H sync");
            },
            faiss_stream,
            true));

    report.recall_at_1 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 1);
    report.recall_at_10 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 10);
    report.recall_at_100 = dsio::recall_at_k(I.data(), k_search, gt.data(), gt_k, nq, 100);

    report.phases.push_back(survey::measure_phase(
            "export_index_gpu_to_cpu_write",
            [&]() {
                faiss::Index* cpu_index = faiss::gpu::index_gpu_to_cpu(&index);
                faiss::write_index(cpu_index, "faiss_ivf_pq.faissindex");
                delete cpu_index;
            },
            faiss_stream,
            true));

    cudaFree(d_queries);
    cudaFree(d_labels);
    cudaFree(d_dists);
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
