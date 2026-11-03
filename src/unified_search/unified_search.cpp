// Common searcher: the same best-first beam search on any graph in CSR form, used to compare the graphs built
// by different builders.
//
// Search rule (identical for all graphs, no builder-specific optimization):
//   A candidate list of size L sorted by ascending distance. Each step expands the closest unexpanded candidate:
//   the distances of its out-neighbors are computed, only for points not visited before, and counted. The
//   search stops when the first L candidates have all been expanded. All graphs share the same entry points.
//   Distances are squared L2. The first k candidates are returned.
//
// Counters: ndist is the number of distance computations (entry points included), hops the number of
// expansions. Both depend only on the graph, the entry points and L, not on the machine load.
//
// Timing: per L, one pass over all queries is run as an untimed warm-up, then --reps passes are timed (wall time
// of the parallel search). search_s is the mean over the timed passes, search_s_std their sample standard
// deviation. Recall, counters and the hop curve are those of the last timed pass; every pass gives the same values.
//
// Usage:
//   unified_search search --base B --query Q --gt G --graph CSR --entry E --L 10,20,40 [--k 10] [--nq N]
//                         [--hop-curve out.tsv --max-hops 400] [--threads T] [--reps 3]
//   unified_search stats  --graph CSR
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>
#include <omp.h>

namespace {

struct Matrix {  // fbin / ibin: int32 n, int32 d, then the rows
    int64_t n = 0, d = 0;
    std::vector<char> buf;
    template <class T> const T* row(int64_t i) const { return reinterpret_cast<const T*>(buf.data()) + i * d; }
};

Matrix read_bin(const std::string& path, size_t elem) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    int32_t hdr[2];
    if (std::fread(hdr, 4, 2, f) != 2) throw std::runtime_error("bad header " + path);
    Matrix m; m.n = hdr[0]; m.d = hdr[1];
    m.buf.resize(size_t(m.n) * m.d * elem);
    if (std::fread(m.buf.data(), 1, m.buf.size(), f) != m.buf.size()) throw std::runtime_error("short read " + path);
    std::fclose(f);
    return m;
}

struct Csr {  // graph.csr: int64 n, int64 nnz, int64 offsets[n+1], int32 nbrs[nnz]
    int64_t n = 0, nnz = 0;
    std::vector<int64_t> off;
    std::vector<int32_t> adj;
};

Csr read_csr(const std::string& path) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) throw std::runtime_error("cannot open " + path);
    Csr g; int64_t hdr[2];
    if (std::fread(hdr, 8, 2, f) != 2) throw std::runtime_error("bad csr header");
    g.n = hdr[0]; g.nnz = hdr[1];
    g.off.resize(g.n + 1); g.adj.resize(g.nnz);
    if (std::fread(g.off.data(), 8, g.n + 1, f) != size_t(g.n + 1)) throw std::runtime_error("short offsets");
    if (std::fread(g.adj.data(), 4, g.nnz, f) != size_t(g.nnz)) throw std::runtime_error("short adj");
    std::fclose(f);
    if (g.off[g.n] != g.nnz) throw std::runtime_error("offsets[n] != nnz");
    return g;
}

inline float l2sq(const float* a, const float* b, int64_t d) {
    float s = 0.f;
    for (int64_t i = 0; i < d; ++i) { float t = a[i] - b[i]; s += t * t; }
    return s;
}

struct Cand { float d; int32_t id; bool expanded; };

struct Stats { int64_t ndist = 0, hops = 0; };

// Search of one query. The visited set is a stamp array, so it need not be cleared for every query.
Stats search_one(const float* q, const Matrix& base, const Csr& g, const std::vector<int32_t>& entry,
                 int L, int k, std::vector<uint32_t>& stamp, uint32_t tag, std::vector<Cand>& cand,
                 std::vector<int32_t>& topk, std::vector<float>* hop_recall, std::vector<int64_t>* hop_ndist,
                 const int32_t* gt, int gtk, int max_hops) {
    Stats st; cand.clear();
    auto push = [&](int32_t v) -> int {  // returns the insert position, -1 if not inserted
        if (stamp[v] == tag) return -1;
        stamp[v] = tag;
        float dv = l2sq(q, base.row<float>(v), base.d); ++st.ndist;
        if ((int)cand.size() == L && dv >= cand.back().d) return -1;
        auto it = std::lower_bound(cand.begin(), cand.end(), dv, [](const Cand& c, float x) { return c.d < x; });
        int pos = int(it - cand.begin());
        cand.insert(it, Cand{dv, v, false});
        if ((int)cand.size() > L) cand.pop_back();
        return pos;
    };
    auto recall_now = [&]() {
        int m = std::min<int>(k, cand.size()), hit = 0;
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < k && j < gtk; ++j) if (cand[i].id == gt[j]) { ++hit; break; }
        return float(hit) / k;
    };
    for (int32_t e : entry) push(e);
    size_t cur = 0;
    while (cur < cand.size()) {
        cand[cur].expanded = true;
        int32_t u = cand[cur].id; ++st.hops;
        size_t low = cur + 1;
        for (int64_t p = g.off[u]; p < g.off[u + 1]; ++p) {
            int32_t v = g.adj[p];
            if (v < 0 || v >= base.n) continue;
            int pos = push(v);
            if (pos >= 0 && size_t(pos) < low) low = pos;
        }
        cur = low;
        while (cur < cand.size() && cand[cur].expanded) ++cur;
        if (hop_recall && st.hops <= max_hops) {
            (*hop_recall)[st.hops - 1] = recall_now();
            (*hop_ndist)[st.hops - 1] = st.ndist;
        }
    }
    if (hop_recall) {  // a query that stopped early keeps its final values, so every hop averages over all queries
        float r = recall_now();
        for (int h = int(std::min<int64_t>(st.hops, max_hops)); h < max_hops; ++h) { (*hop_recall)[h] = r; (*hop_ndist)[h] = st.ndist; }
    }
    topk.clear();
    for (int i = 0; i < k && i < (int)cand.size(); ++i) topk.push_back(cand[i].id);
    return st;
}

std::map<std::string, std::string> parse(int argc, char** argv, int from) {
    std::map<std::string, std::string> a;
    for (int i = from; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
    return a;
}

int run_search(std::map<std::string, std::string>& a) {
    Matrix base = read_bin(a.at("--base"), 4), query = read_bin(a.at("--query"), 4), gt = read_bin(a.at("--gt"), 4);
    Matrix ent = read_bin(a.at("--entry"), 4);
    Csr g = read_csr(a.at("--graph"));
    if (g.n != base.n) throw std::runtime_error("graph n != base n");
    std::vector<int32_t> entry(ent.row<int32_t>(0), ent.row<int32_t>(0) + ent.n * ent.d);
    int k = a.count("--k") ? std::atoi(a["--k"].c_str()) : 10;
    int64_t nq = a.count("--nq") ? std::min<int64_t>(std::atoll(a["--nq"].c_str()), query.n) : query.n;
    int max_hops = a.count("--max-hops") ? std::atoi(a["--max-hops"].c_str()) : 400;
    const int reps = a.count("--reps") ? std::atoi(a["--reps"].c_str()) : 3;
    if (reps < 1) throw std::runtime_error("--reps must be >= 1");
    bool curve = a.count("--hop-curve") > 0;
    if (a.count("--threads")) omp_set_num_threads(std::atoi(a["--threads"].c_str()));
    std::vector<int> Ls;
    for (size_t s = 0, e; s < a.at("--L").size(); s = e + 1) {
        e = a["--L"].find(',', s); if (e == std::string::npos) e = a["--L"].size();
        Ls.push_back(std::atoi(a["--L"].substr(s, e - s).c_str()));
    }
    FILE* hc = curve ? std::fopen(a["--hop-curve"].c_str(), "w") : nullptr;
    if (hc) std::fprintf(hc, "L\thop\trecall_at_k\tmean_ndist\n");
    std::printf("L\tk\tnq\trecall_at_k\tmean_ndist\tmean_hops\tp50_ndist\tp99_ndist\tsearch_s\tsearch_s_std\tn_timed\n");
    for (int L : Ls) {
        if (L < k) throw std::runtime_error("L must be >= k");
        std::vector<int64_t> nd(nq), hp(nq); std::vector<float> rec(nq);
        std::vector<double> sum_rc(curve ? max_hops : 0, 0.0), sum_nd(curve ? max_hops : 0, 0.0);
        // One pass over all queries: fills rec, nd, hp and the hop-curve sums, returns its wall time in seconds
        auto pass = [&]() -> double {
            std::fill(sum_rc.begin(), sum_rc.end(), 0.0); std::fill(sum_nd.begin(), sum_nd.end(), 0.0);
            auto t0 = std::chrono::steady_clock::now();
#pragma omp parallel
            {
                std::vector<uint32_t> stamp(base.n, 0); uint32_t tag = 0;
                std::vector<Cand> cand; cand.reserve(L + 1); std::vector<int32_t> topk;
                std::vector<float> hr(curve ? max_hops : 0); std::vector<int64_t> hn(curve ? max_hops : 0);
                std::vector<double> lrc(curve ? max_hops : 0, 0.0), lnd(curve ? max_hops : 0, 0.0);
#pragma omp for schedule(dynamic, 16)
                for (int64_t i = 0; i < nq; ++i) {
                    const int32_t* gti = gt.row<int32_t>(i);
                    Stats st = search_one(query.row<float>(i), base, g, entry, L, k, stamp, ++tag, cand, topk,
                                          curve ? &hr : nullptr, curve ? &hn : nullptr, gti, int(gt.d), max_hops);
                    int hit = 0;
                    for (int32_t v : topk) for (int j = 0; j < k && j < gt.d; ++j) if (v == gti[j]) { ++hit; break; }
                    rec[i] = float(hit) / k; nd[i] = st.ndist; hp[i] = st.hops;
                    if (curve) for (int h = 0; h < max_hops; ++h) { lrc[h] += hr[h]; lnd[h] += hn[h]; }
                }
                if (curve) {
#pragma omp critical
                    for (int h = 0; h < max_hops; ++h) { sum_rc[h] += lrc[h]; sum_nd[h] += lnd[h]; }
                }
            }
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        };
        pass();  // warm-up, not timed
        std::vector<double> ts;
        for (int r = 0; r < reps; ++r) ts.push_back(pass());
        const double secs = std::accumulate(ts.begin(), ts.end(), 0.0) / ts.size();
        double var = 0; for (double t : ts) var += (t - secs) * (t - secs);
        const double sd = ts.size() > 1 ? std::sqrt(var / (ts.size() - 1)) : 0.0;
        std::vector<int64_t> snd = nd; std::sort(snd.begin(), snd.end());
        double mr = std::accumulate(rec.begin(), rec.end(), 0.0) / nq;
        double mn = std::accumulate(nd.begin(), nd.end(), 0.0) / nq;
        double mh = std::accumulate(hp.begin(), hp.end(), 0.0) / nq;
        std::printf("%d\t%d\t%lld\t%.5f\t%.1f\t%.1f\t%lld\t%lld\t%.3f\t%.4f\t%d\n", L, k, (long long)nq, mr, mn, mh,
                    (long long)snd[nq / 2], (long long)snd[std::min<int64_t>(nq - 1, nq * 99 / 100)], secs, sd, reps);
        std::fflush(stdout);
        if (hc) for (int h = 0; h < max_hops; ++h) std::fprintf(hc, "%d\t%d\t%.5f\t%.1f\n", L, h + 1, sum_rc[h] / nq, sum_nd[h] / nq);
    }
    if (hc) std::fclose(hc);
    return 0;
}

// Graph statistics: average, minimum and maximum out-degree, self loops, duplicate edges, out-of-range ids,
// weakly connected components (union-find), and the average out-degree without duplicates (the degree reported)
int run_stats(std::map<std::string, std::string>& a) {
    Csr g = read_csr(a.at("--graph"));
    std::vector<int32_t> parent(g.n); std::iota(parent.begin(), parent.end(), 0);
    auto find = [&](int32_t x) { while (parent[x] != x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
    int64_t self = 0, bad = 0, dup = 0, zero = 0, maxd = 0, mind = INT64_MAX, uniq = 0;
    std::vector<int32_t> tmp;
    for (int64_t u = 0; u < g.n; ++u) {
        int64_t deg = g.off[u + 1] - g.off[u];
        maxd = std::max(maxd, deg); mind = std::min(mind, deg); if (deg == 0) ++zero;
        tmp.assign(g.adj.begin() + g.off[u], g.adj.begin() + g.off[u + 1]);
        std::sort(tmp.begin(), tmp.end());
        for (size_t i = 0; i < tmp.size(); ++i) {
            int32_t v = tmp[i];
            if (v < 0 || v >= g.n) { ++bad; continue; }
            if (v == u) { ++self; continue; }
            if (i && tmp[i - 1] == v) { ++dup; continue; }
            ++uniq;
            int32_t ru = find(int32_t(u)), rv = find(v); if (ru != rv) parent[ru] = rv;
        }
    }
    int64_t cc = 0; for (int64_t u = 0; u < g.n; ++u) if (find(int32_t(u)) == u) ++cc;
    std::printf("n\tnnz\tavg_out_degree\tmin_out_degree\tmax_out_degree\tzero_out\tself_loops\tdup_edges\tout_of_range\tweak_cc\tavg_unique_out_degree\n");
    std::printf("%lld\t%lld\t%.3f\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\t%lld\t%.3f\n", (long long)g.n, (long long)g.nnz,
                double(g.nnz) / g.n, (long long)mind, (long long)maxd, (long long)zero, (long long)self, (long long)dup, (long long)bad, (long long)cc,
                double(uniq) / g.n);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: unified_search {search|stats} ...\n"); return 1; }
    try {
        auto a = parse(argc, argv, 2);
        std::string mode = argv[1];
        if (mode == "search") return run_search(a);
        if (mode == "stats") return run_stats(a);
        std::fprintf(stderr, "unknown mode %s\n", argv[1]); return 1;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what()); return 2;
    }
}
