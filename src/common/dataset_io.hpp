#pragma once
/*
 * Dataset I/O shared by the C++ programs.
 * Formats:
 *   - fvecs / ivecs  (every vector prefixed by its int32 dimension)
 *   - fbin  / ibin   (header int32 n, int32 d, then the rows)
 *
 * A dataset directory (see datasets/README.md) holds base.fbin, query.fbin and gt.ibin,
 * optionally learn.fbin; the fvecs / ivecs names are accepted as well.
 */
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace dsio {

inline bool file_exists(const std::string& path) {
    struct stat st;
    return ::stat(path.c_str(), &st) == 0 && S_ISREG(st.st_mode);
}

inline std::string join_path(const std::string& dir, const char* name) {
    if (dir.empty()) {
        return name;
    }
    if (dir.back() == '/') {
        return dir + name;
    }
    return dir + "/" + name;
}

inline std::string join_path(const std::string& dir, const std::string& name) {
    return join_path(dir, name.c_str());
}

inline bool ends_with(const std::string& s, const char* suf) {
    size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

inline float* fvecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int d = 0;
    fread(&d, 1, sizeof(int), f);
    assert((d > 0 && d < 1000000) && "unreasonable dimension");
    fseek(f, 0, SEEK_SET);
    struct stat st;
    fstat(fileno(f), &st);
    size_t sz = static_cast<size_t>(st.st_size);
    assert(sz % ((d + 1) * 4) == 0 && "weird fvecs file size");
    size_t n = sz / ((d + 1) * 4);

    *d_out = static_cast<size_t>(d);
    *n_out = n;
    float* x = new float[n * (d + 1)];
    size_t nr = fread(x, sizeof(float), n * (d + 1), f);
    assert(nr == n * (d + 1) && "could not read whole fvecs file");
    for (size_t i = 0; i < n; i++) {
        memmove(x + i * d, x + 1 + i * (d + 1), d * sizeof(*x));
    }
    fclose(f);
    return x;
}

inline int* ivecs_read(const char* fname, size_t* d_out, size_t* n_out) {
    return reinterpret_cast<int*>(fvecs_read(fname, d_out, n_out));
}

// BigANN / Deep1B style: int32 n, int32 d, then n*d float32 (or int32 for ibin).
inline float* fbin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int n32 = 0, d32 = 0;
    if (fread(&n32, sizeof(int), 1, f) != 1 || fread(&d32, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "bad fbin header: %s\n", fname);
        abort();
    }
    assert(n32 > 0 && d32 > 0 && d32 < 1000000);
    size_t n = static_cast<size_t>(n32);
    size_t d = static_cast<size_t>(d32);

    struct stat st;
    fstat(fileno(f), &st);
    size_t expect = 8ull + n * d * sizeof(float);
    if (static_cast<size_t>(st.st_size) != expect) {
        fprintf(stderr,
                "fbin size mismatch %s: file=%lld expect=%zu (n=%zu d=%zu)\n",
                fname,
                static_cast<long long>(st.st_size),
                expect,
                n,
                d);
        abort();
    }

    float* x = new float[n * d];
    size_t nr = fread(x, sizeof(float), n * d, f);
    assert(nr == n * d && "could not read whole fbin file");
    fclose(f);
    *d_out = d;
    *n_out = n;
    return x;
}

inline int* ibin_read(const char* fname, size_t* d_out, size_t* n_out) {
    FILE* f = fopen(fname, "rb");
    if (!f) {
        fprintf(stderr, "could not open %s\n", fname);
        perror("");
        abort();
    }
    int n32 = 0, d32 = 0;
    if (fread(&n32, sizeof(int), 1, f) != 1 || fread(&d32, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "bad ibin header: %s\n", fname);
        abort();
    }
    assert(n32 > 0 && d32 > 0 && d32 < 1000000);
    size_t n = static_cast<size_t>(n32);
    size_t d = static_cast<size_t>(d32);

    struct stat st;
    fstat(fileno(f), &st);
    size_t expect = 8ull + n * d * sizeof(int);
    if (static_cast<size_t>(st.st_size) != expect) {
        fprintf(stderr,
                "ibin size mismatch %s: file=%lld expect=%zu (n=%zu d=%zu)\n",
                fname,
                static_cast<long long>(st.st_size),
                expect,
                n,
                d);
        abort();
    }

    int* x = new int[n * d];
    size_t nr = fread(x, sizeof(int), n * d, f);
    assert(nr == n * d && "could not read whole ibin file");
    fclose(f);
    *d_out = d;
    *n_out = n;
    return x;
}

inline float* floats_read(const std::string& path, size_t* d_out, size_t* n_out) {
    if (ends_with(path, ".fbin")) {
        printf("load floats fbin: %s\n", path.c_str());
        return fbin_read(path.c_str(), d_out, n_out);
    }
    printf("load floats fvecs: %s\n", path.c_str());
    return fvecs_read(path.c_str(), d_out, n_out);
}

inline int* ints_read(const std::string& path, size_t* d_out, size_t* n_out) {
    if (ends_with(path, ".ibin")) {
        printf("load ints ibin: %s\n", path.c_str());
        return ibin_read(path.c_str(), d_out, n_out);
    }
    printf("load ints ivecs: %s\n", path.c_str());
    return ivecs_read(path.c_str(), d_out, n_out);
}

inline std::string first_existing(const std::string& dir,
                                  const std::vector<const char*>& names) {
    for (const char* n : names) {
        std::string p = join_path(dir, n);
        if (file_exists(p)) {
            return p;
        }
    }
    return {};
}

struct DatasetPaths {
    std::string base;
    std::string query;
    std::string learn;        // empty: the first LEARN_RATIO of the base is used
    std::string groundtruth;
};

inline DatasetPaths resolve_dataset(const std::string& data_dir) {
    DatasetPaths p;
    p.base = first_existing(data_dir, {"base.fbin", "base.fvecs"});
    p.query = first_existing(data_dir, {"query.fbin", "query.fvecs"});
    p.learn = first_existing(data_dir, {"learn.fbin", "learn.fvecs"});
    p.groundtruth = first_existing(data_dir, {"gt.ibin", "gt.ivecs"});
    return p;
}

inline float env_learn_ratio() {
    const char* v = getenv("LEARN_RATIO");
    if (!v || !*v) {
        return 0.1f;
    }
    float r = static_cast<float>(atof(v));
    if (r <= 0.f) {
        return 0.1f;
    }
    if (r > 1.f) {
        return 1.f;
    }
    return r;
}

// If learn path is set, load it; else take the first ratio*nb rows of xb (copied).
inline float* make_learn_from_base(const float* xb,
                                   size_t nb,
                                   size_t d,
                                   size_t* nt_out,
                                   float ratio) {
    size_t nt = static_cast<size_t>(ratio * static_cast<double>(nb) + 0.5);
    if (nt < 1) {
        nt = 1;
    }
    if (nt > nb) {
        nt = nb;
    }
    float* xt = new float[nt * d];
    std::memcpy(xt, xb, nt * d * sizeof(float));
    *nt_out = nt;
    printf("learn: subsampled first %zu / %zu from base (LEARN_RATIO=%.4f)\n", nt, nb, ratio);
    return xt;
}

inline void print_resolved(const DatasetPaths& p) {
    printf("dataset resolve:\n");
    printf("  base  = %s\n", p.base.empty() ? "(missing)" : p.base.c_str());
    printf("  query = %s\n", p.query.empty() ? "(missing)" : p.query.c_str());
    printf("  learn = %s\n",
           p.learn.empty() ? "(none, subsample of base)" : p.learn.c_str());
    printf("  gt    = %s\n",
           p.groundtruth.empty() ? "(missing)" : p.groundtruth.c_str());
}

inline void write_ibin(const std::string& path, const int* data, size_t n, size_t d) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "could not write %s\n", path.c_str());
        perror("");
        abort();
    }
    int n32 = static_cast<int>(n);
    int d32 = static_cast<int>(d);
    if (fwrite(&n32, sizeof(int), 1, f) != 1 || fwrite(&d32, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "failed writing ibin header: %s\n", path.c_str());
        abort();
    }
    size_t nr = fwrite(data, sizeof(int), n * d, f);
    if (nr != n * d) {
        fprintf(stderr, "failed writing ibin body: %s\n", path.c_str());
        abort();
    }
    fclose(f);
    printf("wrote groundtruth ibin: %s (n=%zu k=%zu, %.3f MiB)\n",
           path.c_str(),
           n,
           d,
           (8.0 + n * d * 4.0) / (1024.0 * 1024.0));
}

// WRITE_GT=<path>: run without an input ground truth and write the neighbor ids (int32) found by the search.
inline const char* env_write_gt() {
    const char* v = getenv("WRITE_GT");
    if (!v || !*v) {
        return nullptr;
    }
    return v;
}

inline DatasetPaths resolve_or_die(const std::string& data_dir, bool need_gt, bool need_learn) {
    DatasetPaths p = resolve_dataset(data_dir);
    print_resolved(p);
    if (p.base.empty()) {
        fprintf(stderr, "ERROR: no base vectors under %s (need base.fbin)\n",
                data_dir.c_str());
        abort();
    }
    const bool writing_gt = env_write_gt() != nullptr;
    if (need_gt) {
        if (p.query.empty()) {
            fprintf(stderr, "ERROR: no query vectors under %s\n", data_dir.c_str());
            abort();
        }
        if (p.groundtruth.empty() && !writing_gt) {
            fprintf(stderr,
                    "ERROR: no ground truth under %s (need gt.ibin), "
                    "or set WRITE_GT=/path/out.ibin to write one.\n",
                    data_dir.c_str());
            abort();
        }
    }
    (void)need_learn;
    return p;
}

// Load base (+ optional learn). If learn file missing, subsample from base.
inline void load_base_and_learn(const DatasetPaths& p,
                                float** xb_out,
                                size_t* nb_out,
                                float** xt_out,
                                size_t* nt_out,
                                size_t* d_out) {
    size_t d = 0, nb = 0, nt = 0;
    float* xb = floats_read(p.base, &d, &nb);
    float* xt = nullptr;
    if (!p.learn.empty()) {
        size_t d2 = 0;
        xt = floats_read(p.learn, &d2, &nt);
        assert(d2 == d);
    } else {
        xt = make_learn_from_base(xb, nb, d, &nt, env_learn_ratio());
    }
    *xb_out = xb;
    *nb_out = nb;
    *xt_out = xt;
    *nt_out = nt;
    *d_out = d;
}

// need_gt=false when WRITE_GT is set and no input GT file exists.
inline void load_queries_and_gt(const DatasetPaths& p,
                                float** xq_out,
                                size_t* nq_out,
                                int** gt_out,
                                size_t* gt_k_out,
                                size_t* d_out,
                                bool require_gt = true) {
    size_t d = 0, nq = 0, gt_k = 0, nq2 = 0;
    float* xq = floats_read(p.query, &d, &nq);
    int* gt = nullptr;
    if (!p.groundtruth.empty()) {
        gt = ints_read(p.groundtruth, &gt_k, &nq2);
        if (nq2 != nq) {
            fprintf(stderr, "ERROR: nq=%zu but groundtruth has %zu rows\n", nq, nq2);
            abort();
        }
    } else if (require_gt && !env_write_gt()) {
        fprintf(stderr, "ERROR: groundtruth required but missing\n");
        abort();
    }
    *xq_out = xq;
    *nq_out = nq;
    *gt_out = gt;
    *gt_k_out = gt_k;
    *d_out = d;
}

}  // namespace dsio
