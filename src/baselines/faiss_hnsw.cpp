/*
 * Faiss HNSW on the CPU (IndexHNSWFlat): build and search.
 *
 *   faiss_hnsw build  <data_dir> [M] [efConstruction] [threads] [index_path]
 *   faiss_hnsw search <data_dir> <index_path> [efSearch] [k] [threads]
 *
 * Flags:
 *   -t, --threads N   OpenMP threads (e.g. 1, 10, 20)
 *   -b, --batch N     query batch size; search all queries in chunks of N
 *                     (default: all queries in one call)
 *   --ef-file PATH    sweep the efSearch values listed in PATH (one per line)
 * Env:
 *   HNSW_METRIC=cosine|angular|ip  L2-normalize and use INNER_PRODUCT (default L2)
 *   SEARCH_REPEATS=R               timed passes per efSearch (default 3)
 *
 * Each efSearch is searched once untimed as warm-up, then R timed passes over all queries;
 * wall_ms is the mean wall time of a pass, wall_ms_std its sample standard deviation,
 * QPS = nq / mean wall time, and recall is that of the last timed pass (every pass returns the
 * same ids).
 * Recall@k counts the returned ids that are among the first k ground-truth ids.
 *
 * Examples:
 *   faiss_hnsw build  $DATA_DIR/sift1M 32 200 10 hnsw_M32.faissindex
 *   faiss_hnsw search $DATA_DIR/sift1M hnsw_M32.faissindex 64 10 1
 *   faiss_hnsw search $DATA_DIR/sift1M hnsw_M32.faissindex 10 10 --threads 10 --ef-file ef.txt
 */

#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <strings.h>
#include <vector>

#include <omp.h>

#include <faiss/IndexHNSW.h>
#include <faiss/MetricType.h>
#include <faiss/index_io.h>

#include "dataset_io.hpp"

namespace {

struct Cli {
    std::string mode;
    std::vector<std::string> pos;
    int threads = -1;  // -1: use the positional value or the default
    int batch = -1;    // -1: all queries in one search call
    std::string ef_file;  // if set, sweep all efSearch values in the file
};

void usage(const char* argv0) {
    fprintf(stderr,
            "Usage:\n"
            "  %s build  <data_dir> [M] [efConstruction] [threads] [index_path]\n"
            "  %s search <data_dir> <index_path> [efSearch] [k] [threads]\n"
            "\n"
            "Flags:\n"
            "  -t, --threads N   OpenMP / FAISS search+build threads (e.g. 1, 10, 20)\n"
            "  -b, --batch N     query batch size (chunk all queries into batches of N)\n"
            "  --ef-file PATH    sweep efSearch list (one int/line; # comments ok)\n"
            "  -h, --help\n"
            "\n"
            "Defaults:\n"
            "  build:  M=32 efc=200 threads=omp_get_max_threads index=hnsw.faissindex\n"
            "  search: efSearch=64 k=10 threads=omp_get_max_threads batch=all\n"
            "  HNSW_METRIC=cosine|angular|ip: normalize and use INNER_PRODUCT\n",
            argv0,
            argv0);
}

std::vector<int> load_ef_schedule(const std::string& path) {
    std::vector<int> out;
    std::ifstream in(path);
    if (!in) {
        fprintf(stderr, "ERROR: cannot open ef-file: %s\n", path.c_str());
        return out;
    }
    std::string line;
    while (std::getline(in, line)) {
        // strip comments / whitespace
        auto hash = line.find('#');
        if (hash != std::string::npos) {
            line = line.substr(0, hash);
        }
        // trim
        size_t b = 0;
        while (b < line.size() && isspace(static_cast<unsigned char>(line[b]))) {
            b++;
        }
        size_t e = line.size();
        while (e > b && isspace(static_cast<unsigned char>(line[e - 1]))) {
            e--;
        }
        if (e <= b) {
            continue;
        }
        int v = atoi(line.substr(b, e - b).c_str());
        if (v > 0) {
            out.push_back(v);
        }
    }
    return out;
}

bool parse_cli(int argc, char** argv, Cli* out) {
    if (argc < 2) {
        return false;
    }
    out->mode = argv[1];
    for (int i = 2; i < argc; i++) {
        const char* a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) {
            return false;
        }
        if (!strcmp(a, "-t") || !strcmp(a, "--threads")) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a);
                return false;
            }
            out->threads = atoi(argv[++i]);
            continue;
        }
        if (!strncmp(a, "--threads=", 10)) {
            out->threads = atoi(a + 10);
            continue;
        }
        if (!strcmp(a, "-b") || !strcmp(a, "--batch")) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a);
                return false;
            }
            out->batch = atoi(argv[++i]);
            continue;
        }
        if (!strncmp(a, "--batch=", 8)) {
            out->batch = atoi(a + 8);
            continue;
        }
        if (!strcmp(a, "--ef-file")) {
            if (i + 1 >= argc) {
                fprintf(stderr, "missing value for %s\n", a);
                return false;
            }
            out->ef_file = argv[++i];
            continue;
        }
        if (!strncmp(a, "--ef-file=", 10)) {
            out->ef_file = a + 10;
            continue;
        }
        out->pos.emplace_back(a);
    }
    return out->mode == "build" || out->mode == "search";
}

bool env_use_cosine() {
    const char* v = getenv("HNSW_METRIC");
    if (!v || !*v) {
        return false;
    }
    return strcasecmp(v, "cosine") == 0 || strcasecmp(v, "angular") == 0 ||
            strcasecmp(v, "ip") == 0 || strcasecmp(v, "inner_product") == 0;
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

int resolve_threads(int flag_threads, int positional_threads) {
    int t = flag_threads > 0 ? flag_threads : positional_threads;
    if (t <= 0) {
        t = omp_get_max_threads();
    }
    if (t <= 0) {
        t = 1;
    }
    return t;
}

void apply_threads(int nthreads) {
    // Faiss parallel regions follow the OpenMP thread count
    omp_set_num_threads(nthreads);
    printf("threads=%d (omp_set_num_threads)\n", nthreads);
}

double elapsed_ms(std::chrono::steady_clock::time_point t0,
                  std::chrono::steady_clock::time_point t1) {
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

float recall_at_k(const faiss::idx_t* I,
                  const int* gt,
                  size_t nq,
                  int k,
                  int gt_k) {
    const int kk = std::min(k, gt_k);
    size_t hits = 0;
    for (size_t q = 0; q < nq; q++) {
        for (int i = 0; i < kk; i++) {
            int pred = static_cast<int>(I[q * k + i]);
            for (int j = 0; j < kk; j++) {
                if (pred == gt[q * gt_k + j]) {
                    hits++;
                    break;
                }
            }
        }
    }
    return static_cast<float>(hits) / static_cast<float>(nq * kk);
}

int cmd_build(const Cli& cli) {
    if (cli.pos.empty()) {
        fprintf(stderr, "build requires data_dir\n");
        return 1;
    }
    const char* data_dir = cli.pos[0].c_str();
    int M = cli.pos.size() > 1 ? atoi(cli.pos[1].c_str()) : 32;
    int efc = cli.pos.size() > 2 ? atoi(cli.pos[2].c_str()) : 200;
    int pos_threads = cli.pos.size() > 3 ? atoi(cli.pos[3].c_str()) : -1;
    const char* index_path =
            cli.pos.size() > 4 ? cli.pos[4].c_str() : "hnsw.faissindex";

    if (M <= 0 || efc <= 0) {
        fprintf(stderr, "M and efConstruction must be > 0\n");
        return 1;
    }
    const int nthreads = resolve_threads(cli.threads, pos_threads);
    apply_threads(nthreads);

    const bool use_cosine = env_use_cosine();
    const faiss::MetricType metric =
            use_cosine ? faiss::METRIC_INNER_PRODUCT : faiss::METRIC_L2;

    auto paths = dsio::resolve_or_die(data_dir, /*need_gt=*/false, /*need_learn=*/false);
    size_t d = 0, nb = 0;
    auto t0 = std::chrono::steady_clock::now();
    float* xb = dsio::floats_read(paths.base, &d, &nb);
    if (use_cosine) {
        l2_normalize_rows(xb, nb, d);
    }
    auto t1 = std::chrono::steady_clock::now();
    printf("loaded base nb=%zu d=%zu in %.2f ms | metric=%s\n",
           nb,
           d,
           elapsed_ms(t0, t1),
           use_cosine ? "IP(cosine)" : "L2");

    faiss::IndexHNSWFlat index(static_cast<int>(d), M, metric);
    index.hnsw.efConstruction = efc;
    index.verbose = true;

    printf("BUILD IndexHNSWFlat M=%d efConstruction=%d threads=%d -> %s\n",
           M,
           efc,
           nthreads,
           index_path);

    t0 = std::chrono::steady_clock::now();
    index.add(static_cast<faiss::idx_t>(nb), xb);
    t1 = std::chrono::steady_clock::now();
    const double add_ms = elapsed_ms(t0, t1);
    printf("add done: %.2f ms  (%.2f vec/s)\n",
           add_ms,
           nb / (add_ms / 1000.0));

    t0 = std::chrono::steady_clock::now();
    faiss::write_index(&index, index_path);
    t1 = std::chrono::steady_clock::now();
    printf("wrote %s in %.2f ms\n", index_path, elapsed_ms(t0, t1));

    delete[] xb;
    return 0;
}

int cmd_search(const Cli& cli) {
    if (cli.pos.size() < 2) {
        fprintf(stderr, "search requires data_dir and index_path\n");
        return 1;
    }
    const char* data_dir = cli.pos[0].c_str();
    const char* index_path = cli.pos[1].c_str();
    int ef_pos = cli.pos.size() > 2 ? atoi(cli.pos[2].c_str()) : 64;
    int k = cli.pos.size() > 3 ? atoi(cli.pos[3].c_str()) : 10;
    int pos_threads = cli.pos.size() > 4 ? atoi(cli.pos[4].c_str()) : -1;

    if (k <= 0) {
        fprintf(stderr, "k must be > 0\n");
        return 1;
    }

    std::vector<int> efs;
    if (!cli.ef_file.empty()) {
        efs = load_ef_schedule(cli.ef_file);
        if (efs.empty()) {
            fprintf(stderr, "ef-file empty or unreadable: %s\n", cli.ef_file.c_str());
            return 1;
        }
    } else {
        if (ef_pos <= 0) {
            fprintf(stderr, "efSearch must be > 0\n");
            return 1;
        }
        efs.push_back(ef_pos);
    }

    const int nthreads = resolve_threads(cli.threads, pos_threads);
    apply_threads(nthreads);

    int repeats = 3;
    if (const char* v = getenv("SEARCH_REPEATS"); v && *v) {
        repeats = std::max(1, atoi(v));
    }

    const bool use_cosine = env_use_cosine();
    auto paths = dsio::resolve_or_die(data_dir, /*need_gt=*/true, /*need_learn=*/false);

    size_t d = 0, nq = 0, gt_k = 0;
    float* xq = nullptr;
    int* gt = nullptr;
    auto t0 = std::chrono::steady_clock::now();
    dsio::load_queries_and_gt(paths, &xq, &nq, &gt, &gt_k, &d);
    if (use_cosine) {
        l2_normalize_rows(xq, nq, d);
    }
    auto t1 = std::chrono::steady_clock::now();
    printf("loaded queries nq=%zu d=%zu gt_k=%zu in %.2f ms | metric=%s\n",
           nq,
           d,
           gt_k,
           elapsed_ms(t0, t1),
           use_cosine ? "IP(cosine)" : "L2");

    t0 = std::chrono::steady_clock::now();
    faiss::Index* base = faiss::read_index(index_path);
    t1 = std::chrono::steady_clock::now();
    auto* index = dynamic_cast<faiss::IndexHNSW*>(base);
    if (!index) {
        fprintf(stderr, "index is not IndexHNSW*: %s\n", index_path);
        delete base;
        return 1;
    }
    if (static_cast<size_t>(index->d) != d) {
        fprintf(stderr, "dim mismatch index.d=%d query.d=%zu\n", index->d, d);
        delete base;
        return 1;
    }

    size_t batch = (cli.batch > 0) ? static_cast<size_t>(cli.batch) : nq;
    if (batch == 0) {
        fprintf(stderr, "batch must be > 0\n");
        delete base;
        return 1;
    }
    if (batch > nq) {
        batch = nq;
    }
    const size_t n_batches = (nq + batch - 1) / batch;

    printf("loaded %s in %.2f ms | ntotal=%lld M=%d k=%d "
           "threads=%d batch=%zu nq=%zu repeats=%d n_ef=%zu\n",
           index_path,
           elapsed_ms(t0, t1),
           static_cast<long long>(index->ntotal),
           index->hnsw.nb_neighbors(0) > 0 ? index->hnsw.nb_neighbors(0) / 2 : -1,
           k,
           nthreads,
           batch,
           nq,
           repeats,
           efs.size());

    std::vector<faiss::idx_t> I(nq * static_cast<size_t>(k));
    std::vector<float> D(nq * static_cast<size_t>(k));

    auto run_batched = [&]() {
        for (size_t off = 0; off < nq; off += batch) {
            const size_t n = std::min(batch, nq - off);
            index->search(static_cast<faiss::idx_t>(n),
                          xq + off * d,
                          k,
                          D.data() + off * static_cast<size_t>(k),
                          I.data() + off * static_cast<size_t>(k));
        }
    };

    printf("tsv\tthreads\tbatch\tefSearch\tk\tnq\tn_batches\twall_ms\tqps\trecall@%d\twall_ms_std\tn_timed\n",
           k);

    for (size_t ei = 0; ei < efs.size(); ei++) {
        const int efSearch = efs[ei];
        index->hnsw.efSearch = efSearch;

        // one untimed warm-up search per efSearch
        run_batched();

        // timed passes, back to back; recall from the results of the last one
        std::vector<double> pass_ms(repeats, 0.0);
        for (int r = 0; r < repeats; r++) {
            t0 = std::chrono::steady_clock::now();
            run_batched();
            t1 = std::chrono::steady_clock::now();
            pass_ms[r] = elapsed_ms(t0, t1);
        }
        double total_ms = 0.0;
        for (double x : pass_ms) {
            total_ms += x;
        }
        const double wall_ms = total_ms / repeats;
        double ss = 0.0;
        for (double x : pass_ms) {
            ss += (x - wall_ms) * (x - wall_ms);
        }
        const double wall_ms_std = repeats > 1 ? std::sqrt(ss / (repeats - 1)) : 0.0;
        const double qps = nq / (wall_ms / 1000.0);
        const float recall = recall_at_k(I.data(), gt, nq, k, static_cast<int>(gt_k));

        printf("SEARCH[%zu/%zu]: wall_ms=%.3f qps=%.2f recall@%d=%.6f  "
               "(threads=%d batch=%zu efSearch=%d)\n",
               ei + 1,
               efs.size(),
               wall_ms,
               qps,
               k,
               recall,
               nthreads,
               batch,
               efSearch);
        printf("row\t%d\t%zu\t%d\t%d\t%zu\t%zu\t%.4f\t%.2f\t%.6f\t%.4f\t%d\n",
               nthreads,
               batch,
               efSearch,
               k,
               nq,
               n_batches,
               wall_ms,
               qps,
               recall,
               wall_ms_std,
               repeats);
        fflush(stdout);
    }

    delete base;
    delete[] xq;
    delete[] gt;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    Cli cli;
    if (!parse_cli(argc, argv, &cli)) {
        usage(argv[0]);
        return 1;
    }
    if (cli.mode == "build") {
        return cmd_build(cli);
    }
    return cmd_search(cli);
}
