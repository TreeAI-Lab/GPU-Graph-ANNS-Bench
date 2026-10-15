# Measurements of the original manuscript

Part of the results of the paper come from the measurements of the original manuscript, made on the same NVIDIA
A800 host with the search code of the systems instrumented with per-stage timers and counters. This folder holds
the code of those runs: Tables 5 and 6 (index construction), Figure 5(a) (search iterations of each system's own
search at 90% Recall@10), Figure 9 (share of search time per stage), the GPU-system and single-thread HNSW curves
of Figure 11 for K = 40, 70, 100, Figure 12 except PilotANN and multi-thread HNSW, Figure 13 (CAGRA parameters),
and in Figure 7 the HNSW and Vamana curves, all graph curves on MNIST8M and BANG on DEEP100M. Not every item has
code; see "What has no code" below.

## Programs

| System | Source | Built by | Instrumentation used here |
|---|---|---|---|
| SONG | `song-orig` + `song_orig.patch` | `build_song.sh` | stage percentages, kernel time, `path_length`, `calculate_count` |
| GGNN | `ggnn-orig` + `ggnn_orig.patch`, driver `src/ggnn` | `build_ggnn.sh` | fetch, calc and update clocks, `calc count`, `path count`, kernel time |
| GANNS | `ganns-orig` + `ganns_orig.patch` | `build_ganns.sh` | six stage shares, `cal_count`, `path_length`, graph analysis |
| CAGRA | libcuvs 25.02 (`ENV_CAGRA`), driver `src/cagra` | `build_cagra.sh` | none |
| BANG | `bang` + `bang.patch` | `third_party/build/bang.sh <R>` | STATS timers, `calc_count`, `path_length` |
| PilotANN | `pilotann` + `pilotann.patch` | `third_party/build/pilotann.sh` | none printed |
| Vamana | `diskann-orig` + `diskann_orig.patch` | `third_party/build/diskann_orig.sh` (`build_search_disk_index.sh`) | stage times, hops, distance computations, path length |

`build_song.sh` and `build_ganns.sh` copy the source tree once per dataset (SONG) or per dimension and distance
(GANNS), because these are compile-time constants. `build_song.sh` stops when the tree lacks the search of the
original runs: `main.cu` taking `max_iterations` and the graph directory and printing `path_length`, and
`kernelgraph.h` taking the degree from `config.h`. That program loads `bfsg.graph` and `bfsg.data` from the graph
directory. `build_ganns.sh` warns when `template/` prints no stage shares.

Prerequisites: the environments `build`, `cagra` and `pilot` (environment/README.md), the sources fetched by
`third_party/fetch.sh`, and the datasets sift1M, gist, glove200, nytimes-full, deep1M, deep10M, deep100M, mnist8M.

## Datasets

The original runs read texmex SIFT1M and GIST, ann-benchmarks GloVe200 and NYTimes, LIBSVM MNIST8M, and the first
1M, 10M and 100M vectors of Deep1B. The scripts use these release datasets:

| Original | Release | Dim | Degree D | Distance (SONG, GGNN, GANNS) |
|---|---|---|---|---|
| sift | `sift1M` | 128 | 32 | l2 |
| gist | `gist` | 960 | 128 | l2 |
| glove | `glove200` | 200 | 96 | cos |
| nytimes | `nytimes-full` | 256 | 64 | cos |
| deep1m | `deep1M` | 96 | 32 | l2 |
| deep, deep10m | `deep10M` | 96 | 64 | l2 (GGNN: cos) |
| deep100m | `deep100M` | 96 | 32 | l2 |
| mnist | `mnist8M` | 784 | 128 | l2 |

D is the KBuild of every dataset in the GGNN run script of the original release, and the maximum degree of the
CAGRA and Vamana graphs in Tables 5 and 6; every system is built with degree D (`common.sh`). The release
GloVe200 vectors are L2-normalized while the original runs read the raw vectors. For the cosine runs this does not
change the neighbors; for BANG and Vamana, which search with L2, it does.

## Commands

```bash
# programs (BANG once per degree used: 32, 64, 96, 128)
bash third_party/build/bang.sh 32; bash third_party/build/pilotann.sh; bash third_party/build/diskann_orig.sh
bash experiments/original/build_search_disk_index.sh
bash experiments/original/build_ggnn.sh
bash experiments/original/build_cagra.sh
bash experiments/original/build_song.sh sift1M           # per dataset
bash experiments/original/build_ganns.sh 128 l2          # per dimension and distance

# Tables 5 and 6: one index per system and dataset (also the graphs searched below)
bash experiments/original/build_indexes.sh song sift1M   # song ggnn ganns cagra vamana pilotann

# searches (dataset arguments are optional)
bash experiments/original/run_song.sh sift1M
bash experiments/original/run_ggnn.sh sift1M
bash experiments/original/run_ganns.sh sift1M
bash experiments/original/run_cagra.sh sift1M
bash experiments/original/run_bang.sh sift1M
bash experiments/original/run_pilotann.sh sift1M
bash experiments/original/run_vamana.sh sift1M
bash experiments/original/run_cagra_params.sh            # Figure 13

# Figure 11 (also run_song.sh, run_ganns.sh, run_bang.sh)
K=40 bash experiments/original/run_ggnn.sh sift1M
# Figure 12 (SONG also needs build_song.sh sift1M 64)
bash experiments/original/build_indexes.sh ggnn sift1M 64
DEGREE=64 bash experiments/original/run_ggnn.sh sift1M

# tables for the figures
python3 experiments/original/tools/parse_logs.py curves     --out curves.csv
python3 experiments/original/tools/parse_logs.py stages     --out stages.csv
python3 experiments/original/tools/parse_logs.py iterations --out iterations.csv
python3 experiments/original/tools/parse_logs.py build      --out build.csv
```

The sweep values are environment variables of the run scripts. Their defaults are those of the run scripts of
the original release; the values of the paper curves are the `param` column of the result tables.

| Script | Swept | Default | Kept constant |
|---|---|---|---|
| `run_song.sh` | `IT_LIST` max_iterations | 30 to 60, step 2 | pq_size 60 (build), K 10 |
| `run_ggnn.sh` | `IT_LIST` max_iterations | 30 to 60, step 2 | tau_query 0.5, KQuery = K |
| `run_ganns.sh` | `E_LIST` e | 30 to 60, step 5 | NSW graph |
| `run_cagra.sh` | `IT_LIST` max_iterations | 30 to 60, step 2 | search width 1, internal top-k 128, team size 8, single CTA, k 10 |
| `run_bang.sh` | worklist length L, or `L_LIST` | automatic mode of bang_search | none |
| `run_pilotann.sh` | `EF_LIST` ef_search | 30 to 80, step 2 | NSG, 1,024 random queries drawn once per process |
| `run_vamana.sh` | `L_LIST` L | 10 to 40 step 2, 45 to 100 step 5, 120 to 400 step 20 | 48 threads, 1,000 cached nodes, beam width 2 |
| `run_cagra_params.sh` | width, internal top-k, team size, degree | 1-32, 16-256, 8-32, 16-128 | max_iterations 32 and the other parameters at the run_cagra.sh values |

Index settings (`build_indexes.sh`): SONG FIXED_DEGREE = D; GGNN KBuild = D, tau_build 0.3, 3 refinement
iterations; GANNS NSW with d_min = D/2 and e = 60 (the example of the GANNS README); CAGRA graph degree D,
intermediate degree 2D, the default build algorithm of the library; Vamana `build_disk_index -R D -L 50 -B 10
-M 10` (the BANG README of the original release), converted for BANG by `bang_preprocess.py`; PilotANN
`search-pilot` with NSG of D neighbors and the defaults of its build script (d_principle 96, sample ratio 0.25,
router32x32). The parameter record of the original runs gives only: SONG degree 32 and pq_size 60, GGNN tau 0.3,
GANNS NSW, CAGRA IVF-PQ graph construction on DEEP100M and NN-descent elsewhere, BANG with the DiskANN defaults,
PilotANN NSG with defaults. The per-dataset degrees above follow the average degrees of Tables 5 and 6.

## Timing

Every search setting is searched once as an untimed warm-up and then three times, and `parse_logs.py` reports
the mean of the three timed runs: times and counters are means, QPS is the number of queries divided by the mean
time, recall is the mean recall of the timed runs (the runs return the same neighbors). Where the repetition
happens depends on the program:

| System | Warm-up and timed runs |
|---|---|
| SONG, GGNN, GANNS | `repeat_runs` in `common.sh` starts the program `WARMUP_RUNS` (1) + `TIMED_RUNS` (3) times per setting, each run a new process that loads the index |
| BANG | `bang_search` searches every L once untimed and then three times (`third_party/patches/bang.patch`) |
| PilotANN | `tools/pilotann_bench.py` sets the profiler to 1 untimed and 3 timed searches per ef, of the same 1,024 queries |
| Vamana | `run_vamana.sh` passes L to `search_disk_index` `WARMUP_RUNS` + `TIMED_RUNS` times, so one process searches it four times |
| CAGRA | one run per setting: `my_cagra` prints no search time |

## Outputs

Every search writes `$OUT_DIR/original/<system>/<run>/k<K>/` with `meta.tsv` (system, dataset, K, number of
queries) and one log per setting (`it<n>.log`, `e<n>.log`, `ef<n>.log`, `L<n>.log`, `auto.log`), plus `.recall`
files where the recall is computed outside the program (`tools/cal_recall.py`, merged from the three recall
scripts of the original release; it reads `gt.ibin` or ivecs). The logs of SONG, GGNN and GANNS hold the output
of the warm-up and the timed runs, each after a line `### run warmup 1/1` or `### run timed <i>/3`, and their
`.recall` files one line per timed run; the first line of a Vamana log gives the number of warm-up rows per L.
`<run>` is the dataset name, `<dataset>-d<D>` for another degree, `<dataset>-<parameter>-<value>` for Figure 13.
Build logs are in `$OUT_DIR/original/build/`. GANNS and PilotANN read the ground truth converted to ivecs under
`$OUT_DIR/original/gt/`.

| `parse_logs.py` | Columns | Results table |
|---|---|---|
| `curves` | run, dataset, system, k, param, recall, qps, source, total_time_us, total_time_us_std, n_timed | fig07_qps_recall.csv, fig11_topk.csv, fig12_degree.csv, fig13_cagra_parameters.csv |
| `stages` | dataset, system, param, recall, gpu_cl, cpu_cl, gpu_dc, cpu_dc, gpu_dm, cpu_dm, total_time_us, run, source, total_time_us_std, n_timed | fig09_search_components.csv |
| `iterations` | panel, dataset, system, param, recall, path_total, nq, hops_per_query, path_from, run, source, n_timed | fig05_traversal_depth.csv (panel a) |
| `build` | system, run, ct_s, avg_degree, components, peak_host_gb | tab05_06_construction.csv |

`stages` and `iterations` take, per run, the setting with Recall@10 closest to 0.9. The operating points of
Figures 5(a) and 9 mostly follow this rule; a few are a neighboring setting. QPS and `total_time_us` (all queries
of the dataset) come from: SONG `kernel takes`, GGNN `GPU kernel time`, GANNS the first `Query speed` and `query time`,
BANG the summary line of each L, PilotANN the search time of 1,024 queries, Vamana the QPS column; each is the mean
over the timed runs, `total_time_us_std` the sample standard deviation and `n_timed` the number of timed runs
(empty for CAGRA and for logs without warm-up and timed runs). Stage shares and counters are averaged over the
timed runs. Stage shares are the printed ones for SONG and GGNN; the groupings of the GANNS, CAGRA and BANG stages
are listed in the header of `parse_logs.py` and are not recorded anywhere else. Path counts: GGNN `path count`, GANNS and BANG
`path_length`, SONG `path_length` (its counter starts at 100, as in the paper values, e.g. 50,100 for 1,000 GIST
queries at 50 iterations), CAGRA max_iterations times the queries (the paper values equal this product), Vamana
the `path_length` column when search_disk_index prints it (the pinned DiskANN does not).

Tables 5 and 6: construction time from the build log (SONG `Build time`, GGNN `Graph built in`, GANNS `Running
time`, CAGRA `CAGRA index built in`, DiskANN `Indexing time`, PilotANN `Build completed in`); peak host memory from
GNU time; average degree and components from `tools/song_graph_stats.py`, `tools/vamana_graph_stats.py` (Vamana
and BANG; nodes larger than one 4 KiB sector are read sector by sector), `tools/pilotann_graph_stats.py`, and for
GANNS from the graph analysis that its query program prints in every search log.

## What has no code

- Figure 9, CAGRA: `my_cagra.cu` prints no stage clocks. cuVS prints per-query clocks only when built with the
  `_CLK_BREAKDOWN` switch of `search_single_cta_kernel-inl.cuh`, which is off in the packages. `parse_logs.py`
  reads these prints, but nothing records that the paper values (cycle sums) came from them.
- Figure 9, PilotANN: the patched PilotANN allocates path, calc and per-stage time counters for its GPU and CPU
  search but prints none of them; the program that printed them is not in the sources.
- Figure 9, BANG and GANNS: the timers exist (BANG STATS, GANNS six stage shares); how they were grouped into
  CL, DC and DM, and into GPU and CPU for BANG, is not recorded.
- Figure 5(a), PilotANN and HNSW: no counter output.
- CAGRA time: `my_cagra.cu` has no search timer, so there is no code for the time column of Figure 13, for CAGRA
  QPS on MNIST8M in Figure 7, or for CAGRA in Figures 11 and 12. It returns k = 10 only, so CAGRA at K = 40, 70,
  100 (Figure 11) has no code either. The base setting of the Figure 13 sweeps is not recorded.
- HNSW (Figure 7 at 1, 10, 20, 40 threads; Figure 11 single thread; Tables 5 and 6): no script. PilotANN's
  `script/common.py` has a Faiss `IndexHNSWFlat` build (M = n_neighbors / 2, efConstruction 80) and hnswlib and
  Faiss search paths, but it samples 1,024 queries, while the HNSW recall values of the paper are multiples of
  1e-5 (10,000 queries x 10 neighbors). Nothing links the HNSW results to it.
- Vamana: the DiskANN build of the original runs had a `search_disk_index` that printed seven more columns
  (fetch, calc and update time, hops, distance computations, path length, calc count), and the only Vamana search
  script of those runs calls `search_disk_index` with 48 threads and 1,000 cached nodes. That build is not one of
  the pinned sources, so `run_vamana.sh` runs the pinned DiskANN and gets QPS and recall only, and Figure 5(a) for
  Vamana has no code. The L values of the paper curves are not recorded.
- Tables 5 and 6: peak GPU memory. The original runs sampled peak host and GPU memory during each build; the
  sampler is not in the sources. GGNN and CAGRA have no degree or component analysis; their degree is set at build time.
- BANG on GIST: `bang_preprocess.py` reads several nodes per 4 KiB sector and cannot convert nodes larger than a
  sector (GIST with R = 128).
