#pragma once
/*
 * Host and GPU profiling helpers of faiss_ivf_pq.cpp and cuvs_brute_force.cpp:
 * wall time, CUDA events, peak GPU memory and host RSS per phase, JSON report.
 * The library sources are not modified for these measurements.
 */

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <string>
#include <thread>
#include <vector>

#include <cuda_runtime.h>
#include <unistd.h>

namespace survey {

inline void cuda_check(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
        abort();
    }
}

inline double wall_sec() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct HostMemSnap {
    size_t vm_rss_bytes = 0; // current resident
    size_t vm_hwm_bytes = 0; // peak resident (VmHWM)
    size_t vm_size_bytes = 0;
};

inline size_t parse_kb_line(const char* key, const std::string& status) {
    auto pos = status.find(key);
    if (pos == std::string::npos) {
        return 0;
    }
    pos += std::strlen(key);
    long kb = 0;
    sscanf(status.c_str() + pos, "%ld", &kb);
    return size_t(kb) * 1024ULL;
}

inline HostMemSnap host_mem_snap() {
    HostMemSnap s;
    std::ifstream in("/proc/self/status");
    std::string all((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
    s.vm_rss_bytes = parse_kb_line("VmRSS:", all);
    s.vm_hwm_bytes = parse_kb_line("VmHWM:", all);
    s.vm_size_bytes = parse_kb_line("VmSize:", all);
    return s;
}

struct GpuMemSnap {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    size_t used_bytes = 0;
};

inline GpuMemSnap gpu_mem_snap() {
    GpuMemSnap s;
    cuda_check(cudaMemGetInfo(&s.free_bytes, &s.total_bytes), "cudaMemGetInfo");
    s.used_bytes = s.total_bytes - s.free_bytes;
    return s;
}

/** Background peak sampler for GPU used memory and host RSS. */
class PeakSampler {
   public:
    void start(int interval_us = 500) {
        stop();
        stop_flag_.store(false);
        peak_gpu_used_.store(gpu_mem_snap().used_bytes);
        peak_host_rss_.store(host_mem_snap().vm_rss_bytes);
        th_ = std::thread([this, interval_us] {
            while (!stop_flag_.load()) {
                auto g = gpu_mem_snap().used_bytes;
                auto h = host_mem_snap().vm_rss_bytes;
                size_t prev_g = peak_gpu_used_.load();
                while (g > prev_g &&
                       !peak_gpu_used_.compare_exchange_weak(prev_g, g)) {
                }
                size_t prev_h = peak_host_rss_.load();
                while (h > prev_h &&
                       !peak_host_rss_.compare_exchange_weak(prev_h, h)) {
                }
                usleep(interval_us);
            }
        });
    }

    void stop() {
        if (!th_.joinable()) {
            return;
        }
        stop_flag_.store(true);
        th_.join();
    }

    size_t peak_gpu_used() const {
        return peak_gpu_used_.load();
    }
    size_t peak_host_rss() const {
        return peak_host_rss_.load();
    }

    ~PeakSampler() {
        stop();
    }

   private:
    std::atomic<bool> stop_flag_{false};
    std::atomic<size_t> peak_gpu_used_{0};
    std::atomic<size_t> peak_host_rss_{0};
    std::thread th_;
};

/** CUDA event timer on a specific stream (the library's default stream). */
class CudaPhaseTimer {
   public:
    explicit CudaPhaseTimer(cudaStream_t stream = nullptr) : stream_(stream) {
        cuda_check(cudaEventCreate(&start_), "cudaEventCreate start");
        cuda_check(cudaEventCreate(&stop_), "cudaEventCreate stop");
    }
    ~CudaPhaseTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void record_start() {
        cuda_check(cudaEventRecord(start_, stream_), "cudaEventRecord start");
    }
    void record_stop() {
        cuda_check(cudaEventRecord(stop_, stream_), "cudaEventRecord stop");
    }
    /** Milliseconds of GPU-side work queued on stream_ between start/stop. */
    float elapsed_ms() {
        cuda_check(cudaEventSynchronize(stop_), "cudaEventSynchronize");
        float ms = 0.f;
        cuda_check(cudaEventElapsedTime(&ms, start_, stop_), "cudaEventElapsedTime");
        return ms;
    }

   private:
    cudaStream_t stream_;
    cudaEvent_t start_{};
    cudaEvent_t stop_{};
};

struct PhaseStats {
    std::string name;
    double wall_s = 0;
    float cuda_event_ms = -1.f; // <0 if not measured
    size_t gpu_used_before = 0;
    size_t gpu_used_after = 0;
    size_t gpu_used_peak = 0; // during phase (sampler)
    size_t host_rss_before = 0;
    size_t host_rss_after = 0;
    size_t host_rss_peak = 0;
    size_t host_hwm_after = 0;
    // Phases that repeat their search: wall_s and cuda_event_ms are the means over n_timed runs,
    // the *_std fields the sample standard deviations. One run: n_timed = 1, std = 0.
    double wall_s_std = 0;
    float cuda_event_ms_std = 0.f;
    int n_timed = 1;
};

inline const char* bytes_human(size_t b, char* buf, size_t n) {
    const double kb = 1024.0, mb = kb * 1024, gb = mb * 1024;
    if (b >= (size_t)gb) {
        snprintf(buf, n, "%.3f GiB", b / gb);
    } else if (b >= (size_t)mb) {
        snprintf(buf, n, "%.3f MiB", b / mb);
    } else if (b >= (size_t)kb) {
        snprintf(buf, n, "%.3f KiB", b / kb);
    } else {
        snprintf(buf, n, "%zu B", b);
    }
    return buf;
}

inline void print_phase(const PhaseStats& p) {
    char b0[64], b1[64], b2[64], b3[64], b4[64], b5[64];
    printf("---- phase: %s ----\n", p.name.c_str());
    printf("  wall_time_s          = %.6f\n", p.wall_s);
    if (p.cuda_event_ms >= 0.f) {
        printf("  cuda_event_ms        = %.3f   # stream event (FAISS work on that stream)\n",
               p.cuda_event_ms);
        printf("  cuda_event_s         = %.6f\n", p.cuda_event_ms / 1000.0);
    }
    printf("  gpu_used_before      = %s (%zu B)\n",
           bytes_human(p.gpu_used_before, b0, sizeof b0),
           p.gpu_used_before);
    printf("  gpu_used_after       = %s (%zu B)\n",
           bytes_human(p.gpu_used_after, b1, sizeof b1),
           p.gpu_used_after);
    printf("  gpu_used_delta       = %s\n",
           bytes_human(p.gpu_used_after > p.gpu_used_before
                               ? p.gpu_used_after - p.gpu_used_before
                               : 0,
                       b2,
                       sizeof b2));
    printf("  gpu_used_peak_phase  = %s (%zu B)\n",
           bytes_human(p.gpu_used_peak, b3, sizeof b3),
           p.gpu_used_peak);
    printf("  host_rss_before      = %s\n", bytes_human(p.host_rss_before, b4, sizeof b4));
    printf("  host_rss_after       = %s\n", bytes_human(p.host_rss_after, b5, sizeof b5));
    printf("  host_rss_peak_phase  = %s (%zu B)\n",
           bytes_human(p.host_rss_peak, b0, sizeof b0),
           p.host_rss_peak);
    printf("  host_VmHWM_after     = %s (%zu B)\n",
           bytes_human(p.host_hwm_after, b1, sizeof b1),
           p.host_hwm_after);
    if (p.n_timed > 1) {
        printf("  n_timed              = %d   # wall and CUDA-event times are means over these runs\n",
               p.n_timed);
        printf("  wall_time_s_std      = %.6f\n", p.wall_s_std);
        if (p.cuda_event_ms >= 0.f) {
            printf("  cuda_event_ms_std    = %.3f\n", p.cuda_event_ms_std);
        }
    }
}

inline double sample_std(const std::vector<double>& v) {
    if (v.size() < 2) {
        return 0.0;
    }
    double mean = 0.0;
    for (double x : v) {
        mean += x;
    }
    mean /= static_cast<double>(v.size());
    double ss = 0.0;
    for (double x : v) {
        ss += (x - mean) * (x - mean);
    }
    return std::sqrt(ss / static_cast<double>(v.size() - 1));
}

/**
 * Per-run times of a phase whose body repeats the search n times: the body calls start(r) before
 * run r and stop(r) after it (after its synchronization). The CUDA events are created before the
 * phase, so the timed body only records them.
 */
class RunTimer {
   public:
    RunTimer(int n, cudaStream_t stream)
            : stream_(stream), start_(n), stop_(n), t0_(n, 0.0), wall_(n, 0.0) {
        for (int r = 0; r < n; r++) {
            cuda_check(cudaEventCreate(&start_[r]), "cudaEventCreate run start");
            cuda_check(cudaEventCreate(&stop_[r]), "cudaEventCreate run stop");
        }
    }
    ~RunTimer() {
        for (size_t r = 0; r < start_.size(); r++) {
            cudaEventDestroy(start_[r]);
            cudaEventDestroy(stop_[r]);
        }
    }
    RunTimer(const RunTimer&) = delete;
    RunTimer& operator=(const RunTimer&) = delete;

    int runs() const {
        return static_cast<int>(start_.size());
    }
    void start(int r) {
        t0_[r] = wall_sec();
        cuda_check(cudaEventRecord(start_[r], stream_), "cudaEventRecord run start");
    }
    void stop(int r) {
        cuda_check(cudaEventRecord(stop_[r], stream_), "cudaEventRecord run stop");
        wall_[r] = wall_sec() - t0_[r];
    }

    /**
     * p covers all runs: its wall and CUDA-event times become the means per run (total / n), and
     * the per-run times give the sample standard deviations.
     */
    void finish(PhaseStats& p) {
        const int n = runs();
        p.wall_s /= static_cast<double>(n);
        if (p.cuda_event_ms >= 0.f) {
            p.cuda_event_ms /= static_cast<float>(n);
        }
        p.wall_s_std = sample_std(wall_);
        if (p.cuda_event_ms >= 0.f) {
            std::vector<double> ms(n, 0.0);
            for (int r = 0; r < n; r++) {
                cuda_check(cudaEventSynchronize(stop_[r]), "cudaEventSynchronize run stop");
                float x = 0.f;
                cuda_check(cudaEventElapsedTime(&x, start_[r], stop_[r]), "cudaEventElapsedTime run");
                ms[r] = x;
            }
            p.cuda_event_ms_std = static_cast<float>(sample_std(ms));
        }
        p.n_timed = n;
    }

   private:
    cudaStream_t stream_;
    std::vector<cudaEvent_t> start_;
    std::vector<cudaEvent_t> stop_;
    std::vector<double> t0_;
    std::vector<double> wall_;
};

/** Timed H2D / D2H of a contiguous buffer (payload transfer microbench). */
inline double timed_memcpy_h2d(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    CudaPhaseTimer t(stream);
    double w0 = wall_sec();
    t.record_start();
    cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyHostToDevice, stream), "H2D");
    t.record_stop();
    float ms = t.elapsed_ms();
    (void)w0;
    return ms / 1000.0;
}

inline double timed_memcpy_d2h(void* dst, const void* src, size_t bytes, cudaStream_t stream) {
    CudaPhaseTimer t(stream);
    t.record_start();
    cuda_check(cudaMemcpyAsync(dst, src, bytes, cudaMemcpyDeviceToHost, stream), "D2H");
    t.record_stop();
    return t.elapsed_ms() / 1000.0;
}

struct SurveyReport {
    // config
    int nlist = 0, m = 0, nbits = 0, nprobe = 0, k = 0, device = 0;
    size_t d = 0, nb = 0, nq = 0, nt = 0;
    bool use_fp16_lut = false;

    std::vector<PhaseStats> phases;

    // transfer microbenchmark (standalone copies, not inside the library)
    double h2d_base_s = 0, h2d_query_s = 0, d2h_result_s = 0;
    size_t h2d_base_bytes = 0, h2d_query_bytes = 0, d2h_result_bytes = 0;

    // quality
    double recall_at_1 = 0, recall_at_10 = 0, recall_at_100 = 0;
    double search_qps = 0;

    // process-wide peaks
    size_t gpu_total_bytes = 0;
    size_t host_hwm_end = 0;
    size_t gpu_used_peak_global = 0;

    void print_all() const {
        printf("\n========== SURVEY METRICS ==========\n");
        printf("config: d=%zu nb=%zu nq=%zu nt=%zu | nlist=%d m=%d nbits=%d nprobe=%d k=%d fp16_lut=%d\n",
               d,
               nb,
               nq,
               nt,
               nlist,
               m,
               nbits,
               nprobe,
               k,
               (int)use_fp16_lut);
        char buf[64];
        printf("gpu_total = %s\n", bytes_human(gpu_total_bytes, buf, sizeof buf));

        for (const auto& p : phases) {
            print_phase(p);
        }

        printf("---- transfer microbench (standalone cudaMemcpy; NOT FAISS-internal) ----\n");
        auto bw = [](size_t bytes, double s) {
            return (s > 0) ? (bytes / s) / (1024.0 * 1024.0 * 1024.0) : 0.0;
        };
        printf("  H2D base   : %.6f s, %zu B, %.2f GiB/s\n",
               h2d_base_s,
               h2d_base_bytes,
               bw(h2d_base_bytes, h2d_base_s));
        printf("  H2D query  : %.6f s, %zu B, %.2f GiB/s\n",
               h2d_query_s,
               h2d_query_bytes,
               bw(h2d_query_bytes, h2d_query_s));
        printf("  D2H result : %.6f s, %zu B, %.2f GiB/s\n",
               d2h_result_s,
               d2h_result_bytes,
               bw(d2h_result_bytes, d2h_result_s));
        printf("  note: search phase is device-pointer kernel time; "
               "H2D query / D2H result are separate phases (or microbench above).\n");

        printf("---- quality / throughput ----\n");
        printf("  R@1=%.4f R@10=%.4f R@100=%.4f\n", recall_at_1, recall_at_10, recall_at_100);
        printf("  search_QPS=%.1f\n", search_qps);
        char buf_hwm[64], buf_gpeak[64];
        printf("  host_VmHWM_end=%s gpu_used_peak_global=%s\n",
               bytes_human(host_hwm_end, buf_hwm, sizeof buf_hwm),
               bytes_human(gpu_used_peak_global, buf_gpeak, sizeof buf_gpeak));
        printf("====================================\n");
    }

    void write_json(const char* path) const {
        FILE* f = fopen(path, "w");
        if (!f) {
            perror("fopen survey json");
            return;
        }
        fprintf(f, "{\n");
        fprintf(f,
                "  \"d\": %zu, \"nb\": %zu, \"nq\": %zu, \"nt\": %zu,\n"
                "  \"nlist\": %d, \"m\": %d, \"nbits\": %d, \"nprobe\": %d, \"k\": %d,\n"
                "  \"use_fp16_lut\": %s,\n",
                d,
                nb,
                nq,
                nt,
                nlist,
                m,
                nbits,
                nprobe,
                k,
                use_fp16_lut ? "true" : "false");
        fprintf(f, "  \"phases\": [\n");
        for (size_t i = 0; i < phases.size(); ++i) {
            const auto& p = phases[i];
            fprintf(f,
                    "    {\"name\": \"%s\", \"wall_s\": %.8f, \"cuda_event_ms\": %.4f, "
                    "\"gpu_used_before\": %zu, \"gpu_used_after\": %zu, \"gpu_used_peak\": %zu, "
                    "\"host_rss_before\": %zu, \"host_rss_after\": %zu, \"host_rss_peak\": %zu, "
                    "\"host_hwm_after\": %zu, "
                    "\"wall_s_std\": %.8f, \"cuda_event_ms_std\": %.4f, \"n_timed\": %d}%s\n",
                    p.name.c_str(),
                    p.wall_s,
                    p.cuda_event_ms,
                    p.gpu_used_before,
                    p.gpu_used_after,
                    p.gpu_used_peak,
                    p.host_rss_before,
                    p.host_rss_after,
                    p.host_rss_peak,
                    p.host_hwm_after,
                    p.wall_s_std,
                    p.cuda_event_ms_std,
                    p.n_timed,
                    (i + 1 < phases.size()) ? "," : "");
        }
        fprintf(f, "  ],\n");
        fprintf(f,
                "  \"transfer\": {\n"
                "    \"h2d_base_s\": %.8f, \"h2d_base_bytes\": %zu,\n"
                "    \"h2d_query_s\": %.8f, \"h2d_query_bytes\": %zu,\n"
                "    \"d2h_result_s\": %.8f, \"d2h_result_bytes\": %zu\n"
                "  },\n",
                h2d_base_s,
                h2d_base_bytes,
                h2d_query_s,
                h2d_query_bytes,
                d2h_result_s,
                d2h_result_bytes);
        fprintf(f,
                "  \"recall\": {\"at1\": %.6f, \"at10\": %.6f, \"at100\": %.6f},\n"
                "  \"search_qps\": %.3f,\n"
                "  \"gpu_total_bytes\": %zu,\n"
                "  \"host_hwm_end\": %zu,\n"
                "  \"gpu_used_peak_global\": %zu\n"
                "}\n",
                recall_at_1,
                recall_at_10,
                recall_at_100,
                search_qps,
                gpu_total_bytes,
                host_hwm_end,
                gpu_used_peak_global);
        fclose(f);
        printf("wrote %s\n", path);
    }
};

/**
 * Run fn() under wall + optional CUDA event + peak memory sampling.
 * cuda_stream: the library's default stream; pass nullptr to skip CUDA event timing.
 */
template <typename Fn>
PhaseStats measure_phase(
        const char* name,
        Fn&& fn,
        cudaStream_t cuda_stream,
        bool use_cuda_event,
        int sample_us = 500) {
    PhaseStats p;
    p.name = name;
    auto gb = gpu_mem_snap();
    auto hb = host_mem_snap();
    p.gpu_used_before = gb.used_bytes;
    p.host_rss_before = hb.vm_rss_bytes;

    PeakSampler sampler;
    sampler.start(sample_us);

    CudaPhaseTimer* ctimer = nullptr;
    if (use_cuda_event) {
        ctimer = new CudaPhaseTimer(cuda_stream);
        // Ensure prior GPU work finished so the phase is clean
        cuda_check(cudaDeviceSynchronize(), "sync before phase");
        ctimer->record_start();
    }

    double t0 = wall_sec();
    fn();
    if (use_cuda_event) {
        ctimer->record_stop();
        p.cuda_event_ms = ctimer->elapsed_ms();
        delete ctimer;
    } else {
        cuda_check(cudaDeviceSynchronize(), "sync after phase");
    }
    p.wall_s = wall_sec() - t0;

    sampler.stop();
    auto ga = gpu_mem_snap();
    auto ha = host_mem_snap();
    p.gpu_used_after = ga.used_bytes;
    p.host_rss_after = ha.vm_rss_bytes;
    p.host_hwm_after = ha.vm_hwm_bytes;
    p.gpu_used_peak = sampler.peak_gpu_used();
    p.host_rss_peak = sampler.peak_host_rss();
    if (p.gpu_used_peak < p.gpu_used_before) {
        p.gpu_used_peak = p.gpu_used_before;
    }
    if (p.gpu_used_peak < p.gpu_used_after) {
        p.gpu_used_peak = p.gpu_used_after;
    }
    return p;
}

} // namespace survey
