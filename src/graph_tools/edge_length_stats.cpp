/*
 * Short / long edge statistics for one or more CSR graphs against a shared
 * sample with exact top-K neighbors (written by sample_top32).
 *
 * Definition:
 *   edge (u,v) is SHORT iff v is among the exact top-K neighbors of u
 *   (K from gt_topK.ibin, typically 32); otherwise LONG.
 *
 *   ./edge_length_stats <data_dir> <sample_dir> <out_dir> <algo=csr_path> ...
 *
 * sample_dir must contain:
 *   sample_ids.ibin , gt_topK.ibin  (K inferred from file header)
 *
 * Per algo writes:
 *   <out_dir>/<algo>/edges.tsv
 *     sample_idx  u  v  length_l2  is_short  rank_in_topK(-1 if long)
 *   <out_dir>/<algo>/edges.bin   (compact binary, see PROTOCOL)
 *   <out_dir>/<algo>/summary.txt
 * Aggregate:
 *   <out_dir>/summary.tsv
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <omp.h>

#include "dataset_io.hpp"
#include "csr_io.hpp"
#include "ibin_io.hpp"

namespace {

double elapsed_ms(std::chrono::steady_clock::time_point t0,
                  std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

float l2(const float* a, const float* b, size_t d) {
    double s = 0.0;
    for (size_t i = 0; i < d; i++) {
        double diff = static_cast<double>(a[i]) - static_cast<double>(b[i]);
        s += diff * diff;
    }
    return static_cast<float>(std::sqrt(s));
}

struct AlgoArg {
    std::string name;
    std::string csr_path;
};

bool parse_algo(const char* s, AlgoArg* out) {
    const char* eq = strchr(s, '=');
    if (!eq || eq == s || !*(eq + 1)) {
        return false;
    }
    out->name.assign(s, eq);
    out->csr_path = eq + 1;
    return true;
}

void usage(const char* argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s <data_dir> <sample_dir> <out_dir> <algo=csr_path> [algo=csr_path ...]\n"
            "\n"
            "sample_dir: sample_ids.ibin + gt_topK.ibin from sample_top32\n"
            "Short edge: neighbor in exact top-K of the sampled endpoint.\n",
            argv0);
}

std::string basename_algo(const std::string& name) {
    // graph name as a directory name
    std::string o = name;
    for (char& c : o) {
        if (c == '/' || c == ' ') {
            c = '_';
        }
    }
    return o;
}

#pragma pack(push, 1)
struct EdgeRec {
    int32_t sample_idx;
    int32_t u;
    int32_t v;
    float length_l2;
    int8_t is_short;
    int8_t rank;   // 0..K-1 if short, -1 if long
    int16_t pad;
};
#pragma pack(pop)

static_assert(sizeof(EdgeRec) == 20, "EdgeRec size");

}  // namespace

int main(int argc, char** argv) {
    if (argc < 5) {
        usage(argv[0]);
        return 1;
    }
    const char* data_dir = argv[1];
    const char* sample_dir = argv[2];
    const char* out_dir = argv[3];

    std::vector<AlgoArg> algos;
    for (int i = 4; i < argc; i++) {
        AlgoArg a;
        if (!parse_algo(argv[i], &a)) {
            fprintf(stderr, "bad algo arg (want name=path): %s\n", argv[i]);
            return 1;
        }
        algos.push_back(std::move(a));
    }

    {
        std::string cmd = std::string("mkdir -p '") + out_dir + "'";
        if (system(cmd.c_str()) != 0) {
            fprintf(stderr, "mkdir failed: %s\n", out_dir);
            return 1;
        }
    }

    // load sample + gt
    size_t S = 0, one = 0, K = 0;
    auto sample_ids = binio::read_ibin(dsio::join_path(sample_dir, "sample_ids.ibin"), &S, &one);
    if (one != 1) {
        fprintf(stderr, "sample_ids.ibin d must be 1, got %zu\n", one);
        return 1;
    }
    size_t S2 = 0;
    // ground truth file: gt_top32.ibin, else the first of a few other K values
    std::string gt_path = dsio::join_path(sample_dir, "gt_top32.ibin");
    if (!dsio::file_exists(gt_path)) {
        for (int kk : {32, 64, 16, 10, 100}) {
            std::string p = dsio::join_path(sample_dir, ("gt_top" + std::to_string(kk) + ".ibin").c_str());
            if (dsio::file_exists(p)) {
                gt_path = p;
                break;
            }
        }
    }
    if (!dsio::file_exists(gt_path)) {
        fprintf(stderr, "ERROR: no gt_topK.ibin under %s\n", sample_dir);
        return 1;
    }
    auto gt = binio::read_ibin(gt_path, &S2, &K);
    if (S2 != S) {
        fprintf(stderr, "S mismatch sample_ids=%zu gt=%zu\n", S, S2);
        return 1;
    }
    printf("loaded sample S=%zu K=%zu from %s\n", S, K, sample_dir);

    // per sample: exact top-K neighbor id -> rank
    std::vector<std::unordered_map<int, int>> top_rank(S);
    for (size_t i = 0; i < S; i++) {
        top_rank[i].reserve(K * 2);
        for (size_t j = 0; j < K; j++) {
            int v = gt[i * K + j];
            top_rank[i].emplace(v, static_cast<int>(j));
        }
    }

    // base vectors
    auto paths = dsio::resolve_or_die(data_dir, /*need_gt=*/false, /*need_learn=*/false);
    size_t d = 0, nb = 0;
    auto t0 = std::chrono::steady_clock::now();
    float* xb = dsio::floats_read(paths.base, &d, &nb);
    auto t1 = std::chrono::steady_clock::now();
    printf("loaded base nb=%zu d=%zu in %.2f ms\n", nb, d, elapsed_ms(t0, t1));

    const std::string summary_path = dsio::join_path(out_dir, "summary.tsv");
    FILE* sumf = fopen(summary_path.c_str(), "w");
    if (!sumf) {
        fprintf(stderr, "cannot write %s\n", summary_path.c_str());
        return 1;
    }
    fprintf(sumf,
            "algo\tn_sample\tK\tn_edges\tn_short\tn_long\tshort_ratio\t"
            "mean_len_all\tmean_len_short\tmean_len_long\t"
            "p50_len_all\tp90_len_all\n");

    for (const auto& algo : algos) {
        printf("=== %s  %s ===\n", algo.name.c_str(), algo.csr_path.c_str());
        t0 = std::chrono::steady_clock::now();
        csr::Graph g = csr::read(algo.csr_path);
        t1 = std::chrono::steady_clock::now();
        if (static_cast<size_t>(g.n) != nb) {
            fprintf(stderr,
                    "WARN: CSR n=%lld != base nb=%zu\n",
                    static_cast<long long>(g.n),
                    nb);
        }
        printf("  CSR n=%lld nnz=%lld avg_deg=%.2f load=%.1f ms\n",
               static_cast<long long>(g.n),
               static_cast<long long>(g.nnz),
               g.n ? static_cast<double>(g.nnz) / g.n : 0.0,
               elapsed_ms(t0, t1));

        // first pass: count edges from sampled nodes
        size_t n_edges = 0;
        for (size_t si = 0; si < S; si++) {
            int u = sample_ids[si];
            if (u < 0 || u >= g.n) {
                continue;
            }
            n_edges += static_cast<size_t>(g.offsets[u + 1] - g.offsets[u]);
        }

        std::vector<EdgeRec> edges(n_edges);
        std::vector<float> all_lens;
        all_lens.reserve(n_edges);

        t0 = std::chrono::steady_clock::now();
        size_t cursor = 0;
        size_t n_short = 0;
        double sum_all = 0, sum_short = 0, sum_long = 0;

        // OpenMP over samples into per-thread buffers, concatenated in thread order below
        std::vector<std::vector<EdgeRec>> tls(omp_get_max_threads());
#pragma omp parallel
        {
            int tid = omp_get_thread_num();
            auto& local = tls[tid];
            local.clear();
            local.reserve(n_edges / static_cast<size_t>(omp_get_num_threads()) + 64);
#pragma omp for schedule(static)
            for (int si = 0; si < static_cast<int>(S); si++) {
                int u = sample_ids[static_cast<size_t>(si)];
                if (u < 0 || static_cast<int64_t>(u) >= g.n) {
                    continue;
                }
                int64_t b = g.offsets[u];
                int64_t e = g.offsets[u + 1];
                const float* xu = xb + static_cast<size_t>(u) * d;
                for (int64_t p = b; p < e; p++) {
                    int v = g.nbrs[static_cast<size_t>(p)];
                    if (v < 0 || static_cast<size_t>(v) >= nb) {
                        continue;
                    }
                    float len = l2(xu, xb + static_cast<size_t>(v) * d, d);
                    EdgeRec rec{};
                    rec.sample_idx = si;
                    rec.u = u;
                    rec.v = v;
                    rec.length_l2 = len;
                    auto it = top_rank[static_cast<size_t>(si)].find(v);
                    if (it != top_rank[static_cast<size_t>(si)].end()) {
                        rec.is_short = 1;
                        rec.rank = static_cast<int8_t>(it->second);
                    } else {
                        rec.is_short = 0;
                        rec.rank = -1;
                    }
                    rec.pad = 0;
                    local.push_back(rec);
                }
            }
        }
        for (auto& local : tls) {
            for (const auto& rec : local) {
                if (cursor >= edges.size()) {
                    edges.push_back(rec);
                } else {
                    edges[cursor] = rec;
                }
                cursor++;
                all_lens.push_back(rec.length_l2);
                sum_all += rec.length_l2;
                if (rec.is_short) {
                    n_short++;
                    sum_short += rec.length_l2;
                } else {
                    sum_long += rec.length_l2;
                }
            }
        }
        edges.resize(cursor);
        n_edges = cursor;
        size_t n_long = n_edges - n_short;
        t1 = std::chrono::steady_clock::now();
        printf("  edges=%zu short=%zu long=%zu short_ratio=%.4f  (%.1f ms)\n",
               n_edges,
               n_short,
               n_long,
               n_edges ? static_cast<double>(n_short) / n_edges : 0.0,
               elapsed_ms(t0, t1));

        std::sort(all_lens.begin(), all_lens.end());
        auto pct = [&](double p) -> float {
            if (all_lens.empty()) {
                return 0.f;
            }
            size_t idx = static_cast<size_t>(p * (all_lens.size() - 1));
            return all_lens[idx];
        };

        const std::string adir =
                dsio::join_path(out_dir, basename_algo(algo.name).c_str());
        {
            std::string cmd = std::string("mkdir -p '") + adir + "'";
            if (system(cmd.c_str()) != 0) {
                fprintf(stderr, "mkdir failed: %s\n", adir.c_str());
                delete[] xb;
                return 1;
            }
        }

        // edges.tsv: one row per edge
        const std::string tsv_path = dsio::join_path(adir, "edges.tsv");
        FILE* tf = fopen(tsv_path.c_str(), "w");
        if (!tf) {
            fprintf(stderr, "cannot write %s\n", tsv_path.c_str());
            return 1;
        }
        fprintf(tf, "sample_idx\tu\tv\tlength_l2\tis_short\trank_in_topK\n");
        for (const auto& rec : edges) {
            fprintf(tf,
                    "%d\t%d\t%d\t%.6f\t%d\t%d\n",
                    rec.sample_idx,
                    rec.u,
                    rec.v,
                    rec.length_l2,
                    static_cast<int>(rec.is_short),
                    static_cast<int>(rec.rank));
        }
        fclose(tf);

        // edges.bin: uint64 n_edges, then EdgeRec[n]
        const std::string bin_path = dsio::join_path(adir, "edges.bin");
        FILE* bf = fopen(bin_path.c_str(), "wb");
        if (!bf) {
            fprintf(stderr, "cannot write %s\n", bin_path.c_str());
            return 1;
        }
        uint64_t ne = static_cast<uint64_t>(n_edges);
        fwrite(&ne, sizeof(ne), 1, bf);
        if (!edges.empty()) {
            fwrite(edges.data(), sizeof(EdgeRec), n_edges, bf);
        }
        fclose(bf);

        const std::string st_path = dsio::join_path(adir, "summary.txt");
        FILE* sf = fopen(st_path.c_str(), "w");
        double mean_all = n_edges ? sum_all / n_edges : 0;
        double mean_s = n_short ? sum_short / n_short : 0;
        double mean_l = n_long ? sum_long / n_long : 0;
        if (sf) {
            fprintf(sf, "algo\t%s\n", algo.name.c_str());
            fprintf(sf, "csr\t%s\n", algo.csr_path.c_str());
            fprintf(sf, "S\t%zu\n", S);
            fprintf(sf, "K\t%zu\n", K);
            fprintf(sf, "n_edges\t%zu\n", n_edges);
            fprintf(sf, "n_short\t%zu\n", n_short);
            fprintf(sf, "n_long\t%zu\n", n_long);
            fprintf(sf, "short_ratio\t%.6f\n", n_edges ? (double)n_short / n_edges : 0);
            fprintf(sf, "mean_len_all\t%.6f\n", mean_all);
            fprintf(sf, "mean_len_short\t%.6f\n", mean_s);
            fprintf(sf, "mean_len_long\t%.6f\n", mean_l);
            fprintf(sf, "p50_len_all\t%.6f\n", pct(0.50));
            fprintf(sf, "p90_len_all\t%.6f\n", pct(0.90));
            fprintf(sf, "edges_tsv\t%s\n", tsv_path.c_str());
            fprintf(sf, "edges_bin\t%s\n", bin_path.c_str());
            fclose(sf);
        }

        fprintf(sumf,
                "%s\t%zu\t%zu\t%zu\t%zu\t%zu\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\t%.6f\n",
                algo.name.c_str(),
                S,
                K,
                n_edges,
                n_short,
                n_long,
                n_edges ? (double)n_short / n_edges : 0,
                mean_all,
                mean_s,
                mean_l,
                pct(0.50),
                pct(0.90));
        fflush(sumf);
        printf("  wrote %s\n", tsv_path.c_str());
    }

    fclose(sumf);
    printf("wrote %s\n", summary_path.c_str());

    // PROTOCOL
    {
        const std::string proto = dsio::join_path(out_dir, "PROTOCOL.txt");
        FILE* pf = fopen(proto.c_str(), "w");
        if (pf) {
            fprintf(pf,
                    "Short edge = neighbor in exact top-K of sampled node (from sample_top32).\n"
                    "Long edge  = graph neighbor not in that top-K.\n"
                    "length_l2  = Euclidean distance on base vectors.\n"
                    "\n"
                    "edges.bin: uint64 n_edges; then records of 20 bytes:\n"
                    "  int32 sample_idx, u, v; float32 length_l2; int8 is_short; int8 rank; int16 pad\n"
                    "rank = position in top-K (0..K-1) if short, else -1.\n"
                    "\n"
                    "All algos share the same sample_ids / gt_topK under sample_dir.\n");
            fclose(pf);
        }
    }

    delete[] xb;
    return 0;
}
