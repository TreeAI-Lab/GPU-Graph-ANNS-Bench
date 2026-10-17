# Baseline programs

Faiss HNSW on the CPU and the GPU non-graph baselines: IVF-PQ in Faiss and in cuVS, and cuVS brute force. They read a
dataset directory of `datasets/README.md` through `src/common/dataset_io.hpp`. Used by
`experiments/nongraph_baselines` (Figures 1 and 7) and `experiments/topk_degree` (Figure 11).

## Build

```bash
bash third_party/build/faiss.sh          # Faiss with GPU support, $W/build/faiss/install
bash third_party/build/cuvs.sh           # cuVS 25.02, $W/build/cuvs/install
bash src/baselines/build.sh faiss        # faiss_hnsw, faiss_ivf_pq      (environment $ENV_FAISS, CUDA 12.4)
bash src/baselines/build.sh cuvs         # cuvs_ivf_pq, cuvs_brute_force (environment $ENV_CUVS, CUDA 12.4)
```

The binaries go to `$BIN_DIR`. The cuVS programs take the RAFT, RMM and CCCL headers and two logger sources from the
dependency directory of the cuVS build (`CUVS_DEPS`, default `$W/build/cuvs/build/_deps`); override it or any other
Makefile variable after the target, e.g. `bash src/baselines/build.sh cuvs CUVS_DEPS=/path/_deps`. `EXTRA_LDFLAGS`
adds library directories, for example one with `libcurand.so` if the Faiss library needs it.

## Programs

| Program | Source | Library |
|---|---|---|
| `faiss_hnsw` | `faiss_hnsw.cpp` | Faiss `IndexHNSWFlat` (CPU, OpenMP) |
| `faiss_ivf_pq` | `faiss_ivf_pq.cpp`, `survey_profiler.hpp` | Faiss `GpuIndexIVFPQ` |
| `cuvs_ivf_pq` | `cuvs_ivf_pq.cpp`, `survey_profiler_cuvs_ivf.hpp` | `cuvs::neighbors::ivf_pq` and `refine` |
| `cuvs_brute_force` | `cuvs_brute_force.cpp`, `survey_profiler.hpp` | `cuvs::neighbors::brute_force` |

### faiss_hnsw

```
faiss_hnsw build  <data_dir> [M=32] [efConstruction=200] [threads] [index_path=hnsw.faissindex]
faiss_hnsw search <data_dir> <index_path> [efSearch=64] [k=10] [threads] [--threads N] [--batch N] [--ef-file PATH]
```

`--ef-file` sweeps the efSearch values of a file (one per line). `--batch N` splits the queries into search calls of N
queries (default: one call with all queries). Threads default to `omp_get_max_threads()`. For every efSearch the
program searches all queries once untimed as warm-up, then `SEARCH_REPEATS` times timed (default 3), and prints a
`row` line: threads, batch, efSearch, k, nq, n_batches, wall_ms (mean of the timed passes), qps = nq / mean wall time,
Recall@k (returned ids among the first k ground-truth ids, from the last timed pass), wall_ms_std (sample standard
deviation of the timed passes), n_timed. `HNSW_METRIC=cosine` normalizes the vectors and uses the inner product.

### faiss_ivf_pq and cuvs_ivf_pq

```
<prog> build  <data_dir> [nlist=4096] [m=16] [index_path] [json_out=survey_metrics_build.json]
<prog> search <data_dir> [index_path] [nprobe=64] [k=100] [json_out=survey_metrics_search.json]
<prog> all    <data_dir> [nlist=4096] [m=16] [nprobe=64] [k=100] [json_out=survey_metrics_all.json]
```

8-bit PQ codes, device 0. The coarse quantizer and the codebooks are trained on `learn.fbin` if the dataset has one
(MNIST8M), else on the first `LEARN_RATIO` (default 0.1) of the base vectors. `IVF_METRIC=cosine` normalizes base,
training and query vectors and uses the inner product (GloVe200 and NYTimes-full in the paper).

The search copies the queries to the GPU, searches all queries once untimed as warm-up and then `SEARCH_REPEATS` times
timed (default 3) on device pointers. The JSON holds every phase (wall time, CUDA-event time, GPU and host memory; for
the timed search phases the means per search, their sample standard deviations and the number of timed searches),
`search_qps` = nq / mean wall time of one search, and `recall.at1`, `at10`, `at100` from the results of the last timed
search: Recall@k for k = 1, 10, 100, the fraction of the k true nearest neighbors found among the first k returned ids
(`src/common/recall.hpp`; ids not returned count as misses).

| | faiss_ivf_pq | cuvs_ivf_pq |
|---|---|---|
| lookup tables | float16 | float32 (`CUVS_FP16_LUT=1`: float16) |
| refinement | none | `REFINE_RATIO` (default 2): ceil(2 k) candidates re-ranked with exact distances |
| timed phase used for QPS | `search` | `search_e2e` (ANN search and refinement; `search_ann` and `refine` also reported) |
| largest nprobe of the sweeps | 2048 (the Faiss GPU limit) | nlist |
| index file | Faiss CPU index (`faiss_ivf_pq.faissindex`) | cuVS serialized index (`cuvs_ivf_pq.cuvsindex`) |

In `cuvs_ivf_pq` one untimed warm-up of the whole search (ANN search and refinement) precedes the three timed phases
`search_ann`, `refine` and `search_e2e`, each of `SEARCH_REPEATS` timed searches; `refine` re-ranks the candidates of
one more untimed ANN search.

### cuvs_brute_force

```
cuvs_brute_force build  <data_dir> [index_path=cuvs_brute_force.cuvsindex] [json_out]
cuvs_brute_force search <data_dir> [index_path] [k=100] [json_out]
cuvs_brute_force all    <data_dir> [k=100] [json_out]
```

The index holds the base vectors on the GPU. The program searches all queries once untimed as warm-up and then three
times timed; the `search` phase holds the means per search, `search_qps` = nq / mean wall time, and recall comes from
the last timed search. Recall@k as for IVF-PQ above; ties in distance at the k-th neighbor can keep exact
search slightly below 1.0. `IVF_METRIC=cosine` as above. `WRITE_GT=<path>` also writes the returned ids as an
ibin ground-truth file.

## Output JSON

`survey_metrics_*.json`: configuration (d, nb, nq, nt, nlist, m, nbits, nprobe, k), `phases` (name, wall_s,
cuda_event_ms, GPU memory before, after and peak, host RSS, wall_s_std, cuda_event_ms_std, n_timed), standalone
transfer microbenchmarks (`transfer`), `recall`, `search_qps`, and for `cuvs_ivf_pq` also `refine_ratio`, `ann_qps`,
`refine_qps`. In the timed search phases (`search`; `search_ann`, `refine`, `search_e2e` of `cuvs_ivf_pq`) wall_s and
cuda_event_ms are means per search over the n_timed timed searches and the `_std` fields their sample standard
deviations; phases run once have n_timed 1 and standard deviations 0.
`experiments/nongraph_baselines/collect_points.py` turns them into tables; its `qps_kernel` is nq divided by the mean
CUDA-event time of the search phase.
