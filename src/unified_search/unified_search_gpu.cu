// GPU version of the common searcher: the best-first beam search of unified_search.cpp run on the GPU, used to
// measure the QPS on the GPU.
//
// The search rule is that of the CPU version, step by step:
//   A candidate list of size L sorted by ascending distance. Each hop expands the closest unexpanded candidate,
//   computes and counts the distances of its out-neighbors that have not been visited, and merges them into the
//   list, keeping the first L. The search stops when the first L candidates have all been expanded. All graphs
//   share the same entry points, which count as distance computations. Distances are squared L2.
// The hops, distance computations and results of every query therefore equal those of the CPU version. Only the
// order of exactly equal distances may differ, and the last bits of a distance may differ because the GPU sums in
// another order; both affect very few ties.
//
// Parallelism: one warp per query. In each hop the 32 lanes of the warp check in parallel whether the neighbors
// were visited (one bitmap per query, exact), the new neighbors are split among teams, and each team of `team`
// lanes computes the distance of one neighbor; the batch of new points is then sorted and merged into the list.
// The candidate list, the batch of new points and the query vector are in shared memory.
//
// Timing: base vectors, graph and queries are copied to GPU memory beforehand. Each timed pass starts with
// clearing the visited bitmaps and ends when all queries are done, measured with CUDA events.
// Per L, one warm-up pass is not timed; then --reps passes (default 3) give the mean and the sample standard
// deviation. Recall and counters are those of the last timed pass; every pass gives the same values.
//
// Usage:
//   unified_search_gpu --base B --query Q --gt G --graph CSR --entry E --L 10,20,40 [--k 10] [--nq N]
//                      [--reps 3] [--device 0]
// The output columns are L k nq recall_at_k mean_ndist mean_hops p50_ndist p99_ndist search_s search_s_std
// warps_per_block team n_timed. search_s is the mean over the --reps timed passes, n_timed their number.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#define CK(x)                                                                                        \
    do {                                                                                             \
        cudaError_t e_ = (x);                                                                        \
        if (e_ != cudaSuccess) {                                                                     \
            std::fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); \
            std::exit(3);                                                                            \
        }                                                                                            \
    } while (0)

namespace {

constexpr uint32_t EXPANDED = 0x80000000u;  // the top bit of a candidate id marks it expanded (ids are below 2^31)
constexpr uint32_t FULL = 0xffffffffu;

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

}  // namespace

struct Params {
    const float* base; const float* query; const int64_t* off; const int32_t* adj; const int32_t* entry;
    int64_t n; int d; int L; int k; int nbuf; int n_entry; int team; int q0; int nq_batch;
    uint32_t* visited; int64_t vwords;
    int32_t* topk; int64_t* ndist; int64_t* hops;
};

// Shared memory layout of one warp: cand_d[L] cand_id[L] new_d[nbuf] new_id[nbuf] srt_d[nbuf] srt_id[nbuf] q[d]
__host__ __device__ inline size_t warp_smem_bytes(int L, int nbuf, int d) {
    size_t b = size_t(L) * 8 + size_t(nbuf) * 16 + size_t(d) * 4;
    return (b + 15) / 16 * 16;
}

// One team computes the squared L2 distance of one point to the query: its lanes split the dimensions, then reduce
__device__ inline float team_l2(const float* __restrict__ x, const float* __restrict__ q, int d, int lt, int team) {
    float s = 0.f;
    if ((d & 3) == 0) {
        const float4* x4 = reinterpret_cast<const float4*>(x);
        const float4* q4 = reinterpret_cast<const float4*>(q);
        for (int i = lt; i < d / 4; i += team) {
            float4 a = __ldg(x4 + i), b = q4[i];
            float t0 = a.x - b.x, t1 = a.y - b.y, t2 = a.z - b.z, t3 = a.w - b.w;
            s += t0 * t0 + t1 * t1 + t2 * t2 + t3 * t3;
        }
    } else {
        for (int i = lt; i < d; i += team) { float t = __ldg(x + i) - q[i]; s += t * t; }
    }
    for (int o = team / 2; o > 0; o /= 2) s += __shfl_xor_sync(FULL, s, o);
    return s;
}

// Computes the distances of the points in new_id[0..m), sorts them and merges them into the candidate list (first L
// kept). Returns the list position of the first new point, or INT32_MAX if no new point enters the list.
__device__ int merge_batch(const int L, const int d, const int team, const float* __restrict__ base,
                            const float* q, float* cand_d, uint32_t* cand_id, int& size,
                            float* new_d, int32_t* new_id, float* srt_d, int32_t* srt_id, int m, int lane) {
    if (m == 0) return INT32_MAX;
    const int teams = 32 / team, ti = lane / team, lt = lane % team;
    for (int b = 0; b < m; b += teams) {
        int i = b + ti;
        int v = i < m ? new_id[i] : new_id[0];
        float dv = team_l2(base + int64_t(v) * d, q, d, lt, team);
        if (i < m && lt == 0) new_d[i] = dv;
    }
    __syncwarp();
    for (int i = lane; i < m; i += 32) {
        float di = new_d[i]; int r = 0;
        for (int j = 0; j < m; ++j) { float dj = new_d[j]; r += (dj < di) || (dj == di && j > i); }
        srt_d[r] = di; srt_id[r] = new_id[i];
    }
    __syncwarp();
    // Position of every new point in the merged list, computed on the old list and stored in new_id (the original order is no longer needed)
    for (int r = lane; r < m; r += 32) {
        float dr = srt_d[r]; int lo = 0, hi = size;
        while (lo < hi) { int mid = (lo + hi) / 2; if (cand_d[mid] < dr) lo = mid + 1; else hi = mid; }
        new_id[r] = lo + r;
    }
    __syncwarp();
    const int first_pos = new_id[0];
    if (first_pos < L) {
        for (int top = size - 1; top >= first_pos; top -= 32) {
            int i = top - lane;
            bool act = i >= first_pos;
            float di = act ? cand_d[i] : 0.f; uint32_t idi = act ? cand_id[i] : 0u; int t = 0;
            if (act) {
                int lo = 0, hi = m;
                while (lo < hi) { int mid = (lo + hi) / 2; if (srt_d[mid] <= di) lo = mid + 1; else hi = mid; }
                t = i + lo;
            }
            __syncwarp();
            if (act && t < L) { cand_d[t] = di; cand_id[t] = idi; }
            __syncwarp();
        }
        for (int r = lane; r < m; r += 32) {
            int pos = new_id[r];
            if (pos < L) { cand_d[pos] = srt_d[r]; cand_id[pos] = uint32_t(srt_id[r]); }
        }
        __syncwarp();
    }
    size = min(L, size + m);
    return first_pos < L ? first_pos : INT32_MAX;
}

__global__ void search_kernel(Params p) {
    extern __shared__ __align__(16) unsigned char smem[];
    const int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    const int qi = blockIdx.x * (blockDim.x >> 5) + w;  // query index within the batch
    if (qi >= p.nq_batch) return;
    unsigned char* ws = smem + warp_smem_bytes(p.L, p.nbuf, p.d) * w;
    float* cand_d = reinterpret_cast<float*>(ws);
    uint32_t* cand_id = reinterpret_cast<uint32_t*>(cand_d + p.L);
    float* new_d = reinterpret_cast<float*>(cand_id + p.L);
    int32_t* new_id = reinterpret_cast<int32_t*>(new_d + p.nbuf);
    float* srt_d = reinterpret_cast<float*>(new_id + p.nbuf);
    int32_t* srt_id = reinterpret_cast<int32_t*>(srt_d + p.nbuf);
    float* q = reinterpret_cast<float*>(srt_id + p.nbuf);
    const int64_t gq = p.q0 + qi;
    for (int i = lane; i < p.d; i += 32) q[i] = p.query[gq * p.d + i];
    uint32_t* vis = p.visited + int64_t(qi) * p.vwords;
    __syncwarp();

    int size = 0; int64_t ndist = 0, hops = 0;
    const uint32_t lt_mask = (1u << lane) - 1u;
    // Collects the unvisited points of a neighbor range into new_id and returns their number. Of duplicate neighbors
    // only one lane wins the visited bit.
    auto gather = [&](const int32_t* ids, int64_t cnt) -> int {
        int m = 0;
        for (int64_t c = 0; c < cnt; c += 32) {
            int64_t j = c + lane; bool isnew = false; int32_t v = -1;
            if (j < cnt) {
                v = ids[j];
                if (v >= 0 && v < p.n) {
                    uint32_t bit = 1u << (v & 31);
                    isnew = !(atomicOr(vis + (v >> 5), bit) & bit);
                }
            }
            uint32_t mask = __ballot_sync(FULL, isnew);
            if (isnew) new_id[m + __popc(mask & lt_mask)] = v;
            m += __popc(mask);
        }
        __syncwarp();
        return m;
    };

    int m = gather(p.entry, p.n_entry);
    ndist += m;
    merge_batch(p.L, p.d, p.team, p.base, q, cand_d, cand_id, size, new_d, new_id, srt_d, srt_id, m, lane);
    int cur = 0;
    while (true) {
        // first unexpanded candidate
        int found = -1;
        for (int c = cur; c < size; c += 32) {
            int i = c + lane;
            bool un = i < size && !(cand_id[i] & EXPANDED);
            uint32_t mask = __ballot_sync(FULL, un);
            if (mask) { found = c + __ffs(mask) - 1; break; }
        }
        if (found < 0) break;
        cur = found;
        uint32_t u = cand_id[cur];
        __syncwarp();
        if (lane == 0) cand_id[cur] = u | EXPANDED;
        __syncwarp();
        ++hops;
        const int64_t b = p.off[u], e = p.off[u + 1];
        m = gather(p.adj + b, e - b);
        ndist += m;
        int first_pos = merge_batch(p.L, p.d, p.team, p.base, q, cand_d, cand_id, size, new_d, new_id, srt_d, srt_id, m, lane);
        cur = min(first_pos, cur + 1);
    }
    for (int i = lane; i < p.k; i += 32)
        p.topk[gq * p.k + i] = i < size ? int32_t(cand_id[i] & ~EXPANDED) : -1;
    if (lane == 0) { p.ndist[gq] = ndist; p.hops[gq] = hops; }
}

namespace {

std::map<std::string, std::string> parse(int argc, char** argv, int from) {
    std::map<std::string, std::string> a;
    for (int i = from; i + 1 < argc; i += 2) a[argv[i]] = argv[i + 1];
    return a;
}

template <class T> T* to_dev(const void* src, size_t bytes) {
    T* d = nullptr; CK(cudaMalloc(&d, bytes)); CK(cudaMemcpy(d, src, bytes, cudaMemcpyHostToDevice)); return d;
}

int run(std::map<std::string, std::string>& a) {
    CK(cudaSetDevice(a.count("--device") ? std::atoi(a["--device"].c_str()) : 0));
    Matrix base = read_bin(a.at("--base"), 4), query = read_bin(a.at("--query"), 4), gt = read_bin(a.at("--gt"), 4);
    Matrix ent = read_bin(a.at("--entry"), 4);
    Csr g = read_csr(a.at("--graph"));
    if (g.n != base.n) throw std::runtime_error("graph n != base n");
    if (base.n >= (int64_t(1) << 31)) throw std::runtime_error("n must be < 2^31");
    const int k = a.count("--k") ? std::atoi(a["--k"].c_str()) : 10;
    const int reps = a.count("--reps") ? std::atoi(a["--reps"].c_str()) : 3;
    if (reps < 1) throw std::runtime_error("--reps must be >= 1");
    const int64_t nq = a.count("--nq") ? std::min<int64_t>(std::atoll(a["--nq"].c_str()), query.n) : query.n;
    const int d = int(base.d), n_entry = int(ent.n * ent.d);
    std::vector<int> Ls;
    for (size_t s = 0, e; s < a.at("--L").size(); s = e + 1) {
        e = a["--L"].find(',', s); if (e == std::string::npos) e = a["--L"].size();
        Ls.push_back(std::atoi(a["--L"].substr(s, e - s).c_str()));
    }
    int64_t maxdeg = 0;
    for (int64_t u = 0; u < g.n; ++u) maxdeg = std::max(maxdeg, g.off[u + 1] - g.off[u]);
    const int nbuf = int(std::max<int64_t>(maxdeg, n_entry));
    const int d4 = (d % 4 == 0) ? d / 4 : d;
    int team = 4; while (team < 32 && team * 4 < d4) team *= 2;  // about 4 float4 per lane

    cudaDeviceProp prop; int dev; CK(cudaGetDevice(&dev)); CK(cudaGetDeviceProperties(&prop, dev));
    const size_t optin = prop.sharedMemPerBlockOptin;
    CK(cudaFuncSetAttribute(search_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, int(optin)));

    float* d_base = to_dev<float>(base.buf.data(), base.buf.size());
    float* d_query = to_dev<float>(query.buf.data(), size_t(nq) * d * 4);
    int64_t* d_off = to_dev<int64_t>(g.off.data(), g.off.size() * 8);
    int32_t* d_adj = to_dev<int32_t>(g.adj.data(), g.adj.size() * 4);
    int32_t* d_entry = to_dev<int32_t>(ent.row<int32_t>(0), size_t(n_entry) * 4);
    int32_t* d_topk; int64_t *d_nd, *d_hp;
    CK(cudaMalloc(&d_topk, size_t(nq) * k * 4)); CK(cudaMalloc(&d_nd, nq * 8)); CK(cudaMalloc(&d_hp, nq * 8));
    const int64_t vwords = (base.n + 31) / 32;
    // Visited bitmaps, one per query. If GPU memory is short the queries run in batches sized by the free memory
    size_t free_b, total_b; CK(cudaMemGetInfo(&free_b, &total_b));
    int64_t batch = std::min<int64_t>(nq, std::max<int64_t>(1, int64_t(free_b * 0.6) / (vwords * 4)));
    uint32_t* d_vis; CK(cudaMalloc(&d_vis, size_t(batch) * vwords * 4));
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));

    std::printf("L\tk\tnq\trecall_at_k\tmean_ndist\tmean_hops\tp50_ndist\tp99_ndist\tsearch_s\tsearch_s_std\twarps_per_block\tteam\tn_timed\n");
    for (int L : Ls) {
        if (L < k) throw std::runtime_error("L must be >= k");
        const size_t per_warp = warp_smem_bytes(L, nbuf, d);
        const int wpb = int(std::min<size_t>(8, optin / per_warp));
        if (wpb < 1) throw std::runtime_error("L too large for shared memory: " + std::to_string(L));
        auto run_once = [&]() {
            for (int64_t q0 = 0; q0 < nq; q0 += batch) {
                const int nb = int(std::min<int64_t>(batch, nq - q0));
                CK(cudaMemsetAsync(d_vis, 0, size_t(nb) * vwords * 4));
                Params p{d_base, d_query, d_off, d_adj, d_entry, base.n, d, L, k, nbuf, n_entry, team, int(q0), nb,
                         d_vis, vwords, d_topk, d_nd, d_hp};
                search_kernel<<<(nb + wpb - 1) / wpb, wpb * 32, per_warp * wpb>>>(p);
                CK(cudaGetLastError());
            }
        };
        run_once(); CK(cudaDeviceSynchronize());  // warm-up, not timed
        std::vector<double> ts;
        for (int r = 0; r < reps; ++r) {
            CK(cudaEventRecord(t0)); run_once(); CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float ms; CK(cudaEventElapsedTime(&ms, t0, t1)); ts.push_back(ms / 1000.0);
        }
        std::vector<int32_t> topk(size_t(nq) * k); std::vector<int64_t> nd(nq), hp(nq);
        CK(cudaMemcpy(topk.data(), d_topk, topk.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(nd.data(), d_nd, nq * 8, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(hp.data(), d_hp, nq * 8, cudaMemcpyDeviceToHost));
        double rec = 0;
        for (int64_t i = 0; i < nq; ++i) {
            const int32_t* gti = gt.row<int32_t>(i); int hit = 0;
            for (int j = 0; j < k; ++j) {
                int32_t v = topk[i * k + j];
                for (int t = 0; t < k && t < gt.d; ++t) if (v == gti[t]) { ++hit; break; }
            }
            rec += double(hit) / k;
        }
        std::vector<int64_t> snd = nd; std::sort(snd.begin(), snd.end());
        const double mean_t = std::accumulate(ts.begin(), ts.end(), 0.0) / ts.size();
        double var = 0; for (double t : ts) var += (t - mean_t) * (t - mean_t);
        const double sd_t = ts.size() > 1 ? std::sqrt(var / (ts.size() - 1)) : 0.0;
        std::printf("%d\t%d\t%lld\t%.5f\t%.1f\t%.1f\t%lld\t%lld\t%.4f\t%.4f\t%d\t%d\t%d\n", L, k, (long long)nq, rec / nq,
                    std::accumulate(nd.begin(), nd.end(), 0.0) / nq, std::accumulate(hp.begin(), hp.end(), 0.0) / nq,
                    (long long)snd[nq / 2], (long long)snd[std::min<int64_t>(nq - 1, nq * 99 / 100)], mean_t, sd_t, wpb, team,
                    reps);
        std::fflush(stdout);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        auto a = parse(argc, argv, 1);
        return run(a);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what()); return 2;
    }
}
