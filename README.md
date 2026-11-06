# GPU Graph Vector Search Benchmark

Code for the experiments of *A Survey and Experimental Study of Single-GPU Graph Vector Search Algorithms*.
The study evaluates six GPU graph-based approximate nearest neighbor search systems (SONG, GGNN, GANNS, CAGRA, and
the CPU-GPU hybrids BANG and PilotANN) against CPU graph indexes (HNSW in Faiss, the DiskANN disk index) and GPU
non-graph indexes (IVF-PQ in Faiss and cuVS, brute force in cuVS), on eight datasets of up to 100 million vectors.

## Layout

| Path | Content |
|---|---|
| `env.sh` | paths used by every script (work directory, datasets, indexes, outputs, conda environments) |
| `environment/` | conda environments, with lock files of the exact packages used |
| `third_party/` | pinned upstream commits of every system (`versions.tsv`), our patches, fetch and build scripts |
| `datasets/` | download, conversion and ground truth of the eight datasets |
| `src/` | our programs: search drivers, builders, the common searcher, baselines, FP16 variants, graph tools |
| `experiments/` | one folder per experiment with its run and summary scripts |
| `plots/` | one script per data figure; each reads one table and writes the PDFs |

Every script starts with a comment that gives its usage, inputs, parameters and outputs.

## Setup

```bash
source env.sh                       # set W, DATA_DIR, INDEX_DIR, OUT_DIR first to place them elsewhere
bash environment/create_envs.sh     # or --locked for the exact package builds
bash third_party/fetch.sh           # sources into $W/src, patches applied
bash third_party/build/all.sh       # third-party builds into $W/build
bash datasets/download.sh sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M
$ENV_BUILD/bin/python datasets/prepare.py sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M --gpu
```

`create_envs.sh` creates five conda environments under `$W/envs`: `build` (CUDA 12.4, gcc 12; SONG, GGNN, GANNS,
DiskANN, BANG and our drivers), `cagra` (cuVS 25.02.01 packages), `pilot` (PilotANN, faiss-cpu 1.15.1, PyTorch
2.5.1), `faiss-build` (Faiss with GPU support from source) and `cuvs-build` (cuVS 25.02 from source).

Our programs are compiled into `$BIN_DIR`:

```bash
bash src/search/build.sh all            # search drivers of CAGRA, GGNN, GANNS, SONG
bash src/build/build_ggnn_driver.sh     # GGNN graph builder
bash src/unified_search/build.sh        # common searcher, CPU and GPU
bash src/baselines/build.sh all         # Faiss HNSW, Faiss and cuVS IVF-PQ, cuVS brute force
bash src/bench/build.sh                 # host-to-device copy benchmark
bash src/fp16/build.sh                  # FP32 and FP16 search programs
bash src/graph_tools/build.sh           # edge statistics, CAGRA graph conversion
E=experiments/equal_degree
bash $E/build_ggnn.sh; bash $E/build_cagra.sh   # GGNN and CAGRA drivers of the equal-degree runs
# SONG and GANNS fix the dimension and the degree at compile time: one program per dataset, per dimension and distance
for ds in sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M; do bash $E/build_song.sh $ds; done
for d in 96 128 784 960; do bash $E/build_ganns.sh $d l2; done; for d in 200 256; do bash $E/build_ganns.sh $d cos; done
```

## Systems

`third_party/versions.tsv` pins every system to an upstream commit; `fetch.sh` downloads the sources and applies
our patches from `third_party/patches/`.

| Name | Upstream | Commit |
|---|---|---|
| song | github.com/sunbelbd/song | 9cb1f486fd72 |
| ggnn | github.com/cgtuebingen/ggnn | 2d03b36362bc |
| ganns | github.com/yuyuanhang/GANNS | 855e2cad3390 |
| cuvs | github.com/rapidsai/cuvs | 20b6775d4d0e (v25.02.01) |
| bang | github.com/karthik86248/BANG-Billion-Scale-ANN | e5a7c4d768fe |
| pilotann | github.com/ytgui/PilotANN | 0aee5f6c5d34 |
| faiss | github.com/facebookresearch/faiss | b0d9e80288da |
| faiss-hnsw | github.com/facebookresearch/faiss | 75c755c5a1fc (v1.15.1) |
| diskann | github.com/microsoft/DiskANN | 78256bbab468 |
| song-instrumented | github.com/sunbelbd/song | 9cb1f486fd72 |
| ggnn-instrumented | github.com/cgtuebingen/ggnn | 2d03b36362bc |
| ganns-instrumented | github.com/yuyuanhang/GANNS | 855e2cad3390 |
| diskann-instrumented | github.com/microsoft/DiskANN | 78256bbab468 |

An `-instrumented` build is the same commit with stage timers and counters added. These builds are used by
`experiments/equal_degree/` and build the SONG graphs and the DEEP100M GGNN graph of `experiments/qps_recall/`.

## Datasets

`download.sh` puts the raw files in `$DATA_DIR/raw`; `prepare.py` writes one directory per dataset with
`base.fbin`, `query.fbin` (int32 n, int32 d, then float32 vectors), the same vectors as fvecs, and `gt.ibin` (the
ids of the 100 nearest base vectors of every query).

| Name | Base | Queries | Dim | Source | Processing |
|---|---|---|---|---|---|
| `sift1M` | 1,000,000 | 10,000 | 128 | ann-benchmarks `sift-128-euclidean` | none |
| `gist` | 1,000,000 | 1,000 | 960 | ann-benchmarks `gist-960-euclidean` | none |
| `glove200` | 1,183,514 | 10,000 | 200 | ann-benchmarks `glove-200-angular` | L2-normalized, cosine |
| `nytimes` | 290,000 | 10,000 | 256 | ann-benchmarks `nytimes-256-angular` | L2-normalized, cosine |
| `deep1M` | 1,000,000 | 10,000 | 96 | first 1M vectors of Deep1B, official queries | ground truth computed |
| `deep10M` | 10,000,000 | 10,000 | 96 | first 10M vectors of Deep1B, official queries | ground truth computed |
| `deep100M` | 100,000,000 | 10,000 | 96 | first 100M vectors of Deep1B, official queries | ground truth computed |
| `mnist8M` | 8,090,000 | 10,000 | 784 | LIBSVM `mnist8m` (`MNIST8M_data.h5` on Hugging Face) | pixel values / 255, the last 10,000 vectors are the queries |

## Build settings

All graph indexes on a dataset share one maximum degree D: 32 on SIFT1M, DEEP1M and DEEP100M, 64 on DEEP10M and
NYTimes, 96 on GloVe200, 128 on GIST and MNIST8M. The construction candidate list L is 60, 120, 150 and 180 for
D = 32, 64, 96 and 128.

| System | Build parameters |
|---|---|
| SONG | FIXED_DEGREE D, construction L |
| GGNN | KBuild D, tau_build 0.3, Layer 4 |
| GANNS | NSW, d_min D/2, candidate list L |
| CAGRA | graph degree D, intermediate degree 2D, NN-descent; IVF-PQ graph construction on DEEP100M |
| BANG | Same construction settings as DiskANN: R = D, construction L, 80 GB budget |
| PilotANN | NSG with D neighbors, sample_ratio 0.25, d_principle 96 |
| HNSW | Faiss IndexHNSWFlat, M = D/2, efConstruction L |
| DiskANN disk index | `build_disk_index`, R = D, construction L, 80 GB budget (B = 80, M = 80) |
| IVF-PQ | nlist 2,048 to 131,072, m 16 to 48; per dataset and library the combination with the highest Recall@10 |

The DiskANN/BANG builders use the degree-dependent L schedule and 80 GB memory budgets by default.
PilotANN uses `d_principle = 96` and `sample_ratio = 0.25`, and the CAGRA recall-QPS drivers use team size 8.

For the standard graph recall-QPS comparisons, vary only the search iteration setting in [10, 500], with
search width 1 and CAGRA team size 8. Each GPU batch contains the full query set (10,000 queries, except 1,000
on GIST). For target recalls of 0.90 and 0.95, select the iteration setting while keeping the other search
parameters fixed. See `parameters/main_exp_settings.md` and `parameters/recall_settings.md` for details.

## Experiments

| Paper | Content | Folder | Plot |
|---|---|---|---|
| Table 3 | code versions | `third_party/versions.tsv` | |
| Table 4 | datasets | `datasets/` | |
| Tables 5, 6 | construction with an equal maximum degree | `equal_degree/` | |
| Table 7 | construction at a comparable recall-throughput point | `construction/` | |
| Figure 1 | SIFT1M, CAGRA and the non-graph indexes | `qps_recall/`, `nongraph_baselines/` | `fig01_intro.py` |
| Figure 5 | (a) search iterations, (b) hops of the common searcher | `equal_degree/`, `traversal_depth/` | `fig05_traversal_depth.py` |
| Figure 6, Table 8 | load once, then serve batches; composition of the load | `loading/` | `fig06_amortization.py` |
| Figure 7 | QPS and Recall@10 on eight datasets | `qps_recall/`, `nongraph_baselines/`, `equal_degree/` | `fig07_qps_recall.py` |
| Figure 8 | batch size | `batch_size/` | `fig08_batch_size.py` |
| Figure 9 | time per search stage | `equal_degree/` | `fig09_search_components.py` |
| Figure 10 | QPS near 95% Recall@10 on DEEP1M, DEEP10M, DEEP100M | `qps_recall/`, `equal_degree/` | `fig10_scalability.py` |
| Figure 11 | Recall@K for K = 10, 40, 70, 100 | `topk_degree/`, `qps_recall/`, `equal_degree/` | `fig11_topk.py` |
| Figure 12 | maximum degree 64 and 128 | `equal_degree/`, `topk_degree/` | `fig12_degree.py` |
| Figure 13 | CAGRA parameters | `cagra_parameters/` | `fig13_cagra_parameters.py` |
| Tables 9, 10, Figure 14 | hardware counters, FP16 vectors | `profiling/` | `fig14_fp16.py` |
| Figure 16 | CAGRA with and without reverse edges | `reverse_edges/` | `fig16_reverse_edges.py` |

Run every block after `source env.sh`. Scripts that take dataset arguments run all datasets without them.

`equal_degree/`: Tables 5 and 6, Figures 5(a), 9 and 12, and the HNSW and DiskANN curves of Figure 7.

```bash
E=experiments/equal_degree
bash $E/build_indexes.sh song sift1M         # song ggnn ganns cagra vamana pilotann; 5 builds each
bash $E/build_hnsw.sh                        # 5 builds each
for s in song ggnn ganns cagra bang pilotann vamana hnsw; do bash $E/run_$s.sh; done
for t in curves stages iterations build; do $ENV_BUILD/bin/python $E/tools/parse_logs.py $t --out $t.csv; done
```

`qps_recall/`: the GPU graph curves of Figures 1, 7 and 10.

```bash
E=experiments/qps_recall
bash $E/build_graphs.sh
bash $E/run_graph_systems.sh                 # CAGRA, GGNN, GANNS, SONG
bash $E/run_bang.sh
bash $E/run_pilotann.sh
$ENV_BUILD/bin/python $E/summarize.py --root $OUT_DIR/qps_recall
```

`nongraph_baselines/`: IVF-PQ and brute force in Figures 1 and 7.

```bash
E=experiments/nongraph_baselines; R=$OUT_DIR/nongraph_baselines
bash $E/run_ivf_sweep.sh                     # candidate indexes, nprobe sweep with k = 100
bash $E/run_bf.sh
$ENV_BUILD/bin/python $E/collect_points.py --ivf-root $R/ivf_sweep --bf-root $R/bf --out $R/points
$ENV_BUILD/bin/python $E/select_configs.py --points $R/points/qps_recall_curve.csv --out $R/selection
SELECTED_CSV=$R/selection/selected_configs.csv bash $E/run_ivf_k10.sh
$ENV_BUILD/bin/python $E/make_tables.py --k10-root $R/ivf_k10 --bf-root $R/bf --out $R/tables
```

`construction/`: Table 7.

```bash
E=experiments/construction
for ds in sift1M deep1M nytimes glove200; do "$ENV_BUILD/bin/python" src/unified_search/entry_points.py "$DATA_DIR/$ds"; done
bash $E/sweep.sh                             # parameter search, counts distance computations only
"$ENV_BUILD/bin/python" $E/collect_sweep.py $E/sweep_grid.tsv "$INDEX_DIR" > "$OUT_DIR/construction/sweep_trials.tsv"
bash $E/run_timing.sh                        # one build at a time: verify, then three timed builds
bash $E/run_gpu_qps.sh                       # QPS of the common searcher
bash $E/select_cagra_qps.sh
```

Under the bound B alone, CAGRA would select sparse graphs that reach only 0.39 to 0.92 of the QPS of GANNS.
`select_cagra_qps.sh` therefore also requires CAGRA to reach 95% of the QPS of GANNS at 90% Recall@10.

`traversal_depth/`: Figure 5(b).

```bash
for ds in sift1M deep1M deep10M gist; do "$ENV_BUILD/bin/python" src/unified_search/entry_points.py "$DATA_DIR/$ds"; done
bash experiments/traversal_depth/run.sh
```

`loading/`: Figure 6 and Table 8.

```bash
E=experiments/loading
for ds in deep1M deep10M deep100M; do bash $E/gt_5m.sh $ds; done
bash $E/run_resident_r95.sh deep1M; bash $E/run_resident_r95.sh deep10M; bash $E/run_deep100M_r95.sh
bash $E/run_h2d_bench.sh
bash experiments/equal_degree/run_hnsw.sh deep1M deep10M  # the HNSW_20 line, needs equal_degree/build_hnsw.sh
$ENV_BUILD/bin/python $E/per_query_table.py --h2d $OUT_DIR/loading/h2d/summary.tsv \
    --root $OUT_DIR/loading --hnsw-root $OUT_DIR/equal_degree/hnsw --out $OUT_DIR/loading/fig06_amortization.csv
$ENV_BUILD/bin/python $E/payload_table.py --out $OUT_DIR/loading/payload.tsv
```

`batch_size/`: Figure 8, batch sizes 1 to 10,000 on SIFT1M at 95% Recall@10.

```bash
E=experiments/batch_size
bash $E/build_indexes.sh
bash $E/run_gpu_systems.sh; bash $E/run_bang.sh; bash $E/run_pilotann.sh; bash $E/run_hnsw.sh
$ENV_BUILD/bin/python $E/summarize.py --root $OUT_DIR/batch_size
```

`topk_degree/`: Figures 11 and 12.

```bash
E=experiments/topk_degree
bash $E/run_gpu_topk.sh
$ENV_BUILD/bin/python $E/summarize_gpu_topk.py --root $OUT_DIR/topk_degree/gpu
bash $E/run_hnsw_topk.sh
for k in 40 70 100; do K=$k bash experiments/equal_degree/run_vamana.sh sift1M; done
for s in ggnn song ganns cagra bang vamana hnsw pilotann; do bash $E/run_${s}_degree.sh; done
```

`cagra_parameters/`: Figure 13.

```bash
bash experiments/cagra_parameters/run_cagra_params.sh
$ENV_BUILD/bin/python experiments/cagra_parameters/summarize.py --root $OUT_DIR/cagra_parameters
```

`profiling/`: Tables 9 and 10 and Figure 14, on the A10 (see Measurement).

```bash
E=experiments/profiling
bash $E/build_bang_graphs.sh
bash $E/platform/run_smoke.sh; bash $E/platform/collect_platform.sh
bash $E/check_points.sh
bash $E/bang_sweep.sh deep1M; bash $E/bang_sweep.sh gist
bash $E/find_kernels_all.sh
bash $E/run_all.sh                           # Nsight Compute, Table 10 QPS, Figure 14 curves
bash $E/profile_pilotann.sh
bash $E/make_tables.sh > "$OUT_DIR/profiling/tables.tex"
```

`reverse_edges/`: Figure 16.

```bash
E=experiments/reverse_edges
bash $E/build_graphs.sh; bash $E/edge_stats.sh; bash $E/search.sh
$ENV_BUILD/bin/python $E/contrast.py
```

Plots: `bash plots/plot_all.sh` draws every figure from the tables in `results/` (or `--results DIR`) into
`plots/out/` (or `--out DIR`). `results/` holds one CSV per figure, assembled from the summary tables of the
experiments above; the header of each plot script names its input file.

## Measurement

All experiments except the profiling subsection ran on one server with Ubuntu 22.04 LTS, an Intel Xeon Gold 6348
CPU at 2.60 GHz and one NVIDIA A800 GPU with 80 GB HBM2e memory. The manuscript specifies CUDA 12.4, `-O3`, and
benchmark drivers in C++17 and CUDA. No other job runs on the server during measurement. Base vectors and the
index are copied from pageable host memory, and query batches from pinned memory. Tables 9 and 10 and Figure 14
ran on a virtual machine with an NVIDIA A10 and 24 GB GDDR6, because the driver administrator rights required to
read performance counters were unavailable on the A800 server.

Every search returns the top 10 neighbors unless K is varied, and Recall@10 is the fraction of the 10 true nearest
neighbors found. All queries of a dataset form one batch, except in the loading, top-K, and batch-size experiments.
Every search process starts after `src/common/clear_caches.sh` has emptied the host page cache, and
with empty device memory, because it is a new process. A process searches one setting, or, when the index stays
resident in device memory, all settings of one sweep. Each setting (iteration count, L, ef, nprobe, batch size,
operating point) is searched once as an untimed warm-up and then timed three times; the mean of the three timed runs
is reported. The manuscript's Experimental Settings describes end-to-end latency from host query receipt to the
return of top-K results to the host. Its Search Efficiency subsection and Figure 7 explicitly use a different
measure: for the pure GPU indexes (SONG, GGNN, GANNS, CAGRA, IVF-PQ, brute force), QPS is the number of queries
divided by the mean search kernel time; for BANG, PilotANN and the CPU indexes it uses the mean time of the whole
search call. Figure 8 uses the wall time of the whole batch for every system, including the query copy and the result
copy. Figure 6 includes query uploads, search, and result downloads; it reports average query time both with
and without the one-time vector and index loading cost. Its scripts time every batch of the stream once and
leave out the first batch as warm-up. Query processing time excludes the waiting time needed to form a batch.
GGNN searches DEEP100M partition by partition, so its Figure 7 curve does not represent a fully GPU-resident index.

## License

The code in this repository is released under the MIT License (`LICENSE`). The patches in `third_party/patches/`
modify the listed upstream projects and are covered by their licenses.
