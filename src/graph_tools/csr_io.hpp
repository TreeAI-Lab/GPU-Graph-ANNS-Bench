#pragma once
/*
 * Reader for graphs in CSR form:
 *   int64 n, int64 nnz, int64 offsets[n+1], int32 nbrs[nnz]
 */
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace csr {

struct Graph {
    int64_t n = 0;
    int64_t nnz = 0;
    std::vector<int64_t> offsets;  // size n+1
    std::vector<int32_t> nbrs;      // size nnz
};

inline Graph read(const std::string& path) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open CSR %s\n", path.c_str());
        perror("");
        abort();
    }
    Graph g;
    if (fread(&g.n, sizeof(int64_t), 1, f) != 1 || fread(&g.nnz, sizeof(int64_t), 1, f) != 1) {
        fprintf(stderr, "ERROR: bad CSR header: %s\n", path.c_str());
        abort();
    }
    if (g.n < 0 || g.nnz < 0) {
        fprintf(stderr, "ERROR: negative CSR sizes in %s\n", path.c_str());
        abort();
    }
    g.offsets.resize(static_cast<size_t>(g.n) + 1);
    g.nbrs.resize(static_cast<size_t>(g.nnz));
    if (fread(g.offsets.data(), sizeof(int64_t), g.offsets.size(), f) != g.offsets.size() ||
        fread(g.nbrs.data(), sizeof(int32_t), g.nbrs.size(), f) != g.nbrs.size()) {
        fprintf(stderr, "ERROR: truncated CSR body: %s\n", path.c_str());
        abort();
    }
    fclose(f);
    if (g.offsets[0] != 0 || g.offsets.back() != g.nnz) {
        fprintf(stderr,
                "ERROR: CSR offsets inconsistent in %s (off0=%lld offn=%lld nnz=%lld)\n",
                path.c_str(),
                static_cast<long long>(g.offsets[0]),
                static_cast<long long>(g.offsets.back()),
                static_cast<long long>(g.nnz));
        abort();
    }
    return g;
}

}  // namespace csr
