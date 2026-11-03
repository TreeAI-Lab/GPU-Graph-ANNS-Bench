#pragma once
/*
 * fbin / ibin readers and writers (int32 n, int32 d, then n*d values).
 */
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace binio {

inline void write_ibin(const std::string& path, const int* data, size_t n, size_t d) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot write %s\n", path.c_str());
        perror("");
        abort();
    }
    int n32 = static_cast<int>(n);
    int d32 = static_cast<int>(d);
    if (fwrite(&n32, sizeof(int), 1, f) != 1 || fwrite(&d32, sizeof(int), 1, f) != 1 ||
        fwrite(data, sizeof(int), n * d, f) != n * d) {
        fprintf(stderr, "ERROR: write failed %s\n", path.c_str());
        abort();
    }
    fclose(f);
}

inline void write_fbin(const std::string& path, const float* data, size_t n, size_t d) {
    FILE* f = fopen(path.c_str(), "wb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot write %s\n", path.c_str());
        perror("");
        abort();
    }
    int n32 = static_cast<int>(n);
    int d32 = static_cast<int>(d);
    if (fwrite(&n32, sizeof(int), 1, f) != 1 || fwrite(&d32, sizeof(int), 1, f) != 1 ||
        fwrite(data, sizeof(float), n * d, f) != n * d) {
        fprintf(stderr, "ERROR: write failed %s\n", path.c_str());
        abort();
    }
    fclose(f);
}

inline std::vector<int> read_ibin(const std::string& path, size_t* n_out, size_t* d_out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        perror("");
        abort();
    }
    int n32 = 0, d32 = 0;
    if (fread(&n32, sizeof(int), 1, f) != 1 || fread(&d32, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "ERROR: bad ibin header %s\n", path.c_str());
        abort();
    }
    size_t n = static_cast<size_t>(n32);
    size_t d = static_cast<size_t>(d32);
    std::vector<int> data(n * d);
    if (fread(data.data(), sizeof(int), n * d, f) != n * d) {
        fprintf(stderr, "ERROR: truncated ibin %s\n", path.c_str());
        abort();
    }
    fclose(f);
    *n_out = n;
    *d_out = d;
    return data;
}

inline std::vector<float> read_fbin(const std::string& path, size_t* n_out, size_t* d_out) {
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) {
        fprintf(stderr, "ERROR: cannot open %s\n", path.c_str());
        perror("");
        abort();
    }
    int n32 = 0, d32 = 0;
    if (fread(&n32, sizeof(int), 1, f) != 1 || fread(&d32, sizeof(int), 1, f) != 1) {
        fprintf(stderr, "ERROR: bad fbin header %s\n", path.c_str());
        abort();
    }
    size_t n = static_cast<size_t>(n32);
    size_t d = static_cast<size_t>(d32);
    std::vector<float> data(n * d);
    if (fread(data.data(), sizeof(float), n * d, f) != n * d) {
        fprintf(stderr, "ERROR: truncated fbin %s\n", path.c_str());
        abort();
    }
    fclose(f);
    *n_out = n;
    *d_out = d;
    return data;
}

}  // namespace binio
