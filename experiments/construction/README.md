# Construction at a comparable recall-throughput point

Table 7 compares what each builder spends to reach the same search quality. Every graph of SONG, GGNN, GANNS,
CAGRA, PilotANN, HNSW and Vamana is built with the smallest parameters whose graph reaches Recall@10 = 0.9 within a
budget B of distance computations per query, both measured with the common searcher (`src/unified_search`) from
shared entry points, so that the builders are compared at equal recall and nearly equal search work. The table
reports the construction time (mean and standard deviation of three builds), average degree, connected components,
peak host and GPU memory, and the QPS of the GPU version of the common searcher at Recall@10 = 0.9.

## Prerequisites

- Datasets `sift1M`, `deep1M`, `nytimes`, `glove200` in `$DATA_DIR` (`datasets/README.md`), and their entry points.
- Conda environments `build`, `cagra`, `pilot` (`environment/README.md`).
- Programs: `unified_search` and `unified_search_gpu` (`src/unified_search/build.sh`), `ggnn_build`
  (`src/build/build_ggnn_driver.sh`), and the third-party builds of GGNN, DiskANN and PilotANN. SONG and GANNS are
  compiled by `src/build/build.sh` on first use. See `src/build/README.md`.
- One GPU. The timings in the paper come from one A800 80GB with 14 CPU threads for the builders.

## Commands

```bash
source env.sh
bash third_party/build/ggnn.sh && bash src/build/build_ggnn_driver.sh
bash third_party/build/diskann.sh
bash third_party/build/pilotann.sh
bash src/unified_search/build.sh
for ds in sift1M deep1M nytimes glove200; do "$ENV_BUILD/bin/python" src/unified_search/entry_points.py "$DATA_DIR/$ds"; done

# 1. Parameter search (optional: its outcome is selected_configs.tsv)
GPUS="0 1 2 3" bash experiments/construction/sweep.sh
"$ENV_BUILD/bin/python" experiments/construction/collect_sweep.py experiments/construction/sweep_grid.tsv "$INDEX_DIR" \
  > "$OUT_DIR/construction/sweep_trials.tsv"

# 2. Verification and timing, one build at a time on an otherwise idle machine
EXCLUDE=song bash experiments/construction/run_timing.sh
SYSTEMS=song bash experiments/construction/run_timing.sh

# 3. QPS of the common searcher on the selected configurations
bash experiments/construction/run_gpu_qps.sh
```

For the paper, step 2 timed the six other systems first and SONG afterwards; both commands add to the same tables.
SONG builds on one CPU thread and takes most of the time: on the A800 machine one build took about 11 minutes on
SIFT1M, 8 on DEEP1M, 13 on NYTimes and 80 on GloVe200, and step 2 builds every configuration four times (about
9 hours in total, 7.7 of them for SONG). The parameter search
only counts distance computations, so it may run on another machine, in parallel, and without exclusive use;
step 2 verifies every configuration again on the timing machine. All timings must come from one machine.

## Operating point and parameter selection

| Dataset | B (distance computations per query at Recall@10 = 0.9) |
|---|---|
| SIFT1M | 1000 |
| DEEP1M | 1000 |
| NYTimes | 6000 |
| GloVe200 | 14500 |

B is the smallest round number that every builder can reach by raising its parameters. The
distance computations are counted by the common searcher, so they do not depend on the machine or its load, and at
equal recall they largely determine the QPS of that searcher.

For each builder and dataset the selection takes the smallest degree parameter that reaches the point, and then,
at that degree, the smallest effort parameter that still reaches it. The degree is not equalized across builders;
it is reported in the table.

| System | Degree parameter | Effort parameters |
|---|---|---|
| SONG | `SD` (with `SHIFT` the smallest that fits `2*SD+1`) | `BUDGET` |
| GANNS | `K` | `M` |
| GGNN | `K` | `TAU`, `REFINE` |
| CAGRA | `GD`, with `IGD = 2 GD` | `NITER` |
| PilotANN (NSG) | `NN` | `EFC` |
| HNSW | `M` | `EFC` |
| Vamana | `R` | `L` (alpha is a pruning parameter and stays at 1.2) |

The search started from the default parameters of every builder and, in several rounds, raised them where the
defaults missed the point and lowered them where they reached it. Where no degree reached the point at the default
effort (HNSW on NYTimes and GloVe200 with efConstruction 40), the effort was raised first and the degree then
minimized at that effort. `sweep_grid.tsv` lists all 215 configurations that were built and
searched, and `collect_sweep.py` turns their graph directories into the trial table from which
`selected_configs.tsv` was read. `tuning_log.tsv` is that trial table as measured for the paper (215
configurations, columns as in `sweep_trials.tsv` below). Configurations that failed to build are not in the grid: GANNS with M = K
(K 32, M 32 on SIFT1M; K 64, M 64 on NYTimes) crashed, GGNN rejects odd K, and PilotANN with EFC 640 on GloVe200
fails an assertion in Faiss.

`selected_configs.tsv` has the columns `dataset B system build_args ndist_at_r90 fallback_args`. `ndist_at_r90` is
the value found in the parameter search. Configurations less than 1.5% below B carry `fallback_args`, a configuration of
the parameter search with more margin, which step 2 uses if the rebuilt graph exceeds B; `-` means none.

## Outputs

Graphs go to `$INDEX_DIR/<dataset>/<system>/<tag><suffix>/` (`src/build/README.md`): `_sweep` for the parameter
search, `_verify` and `_rep1` to `_rep3` for step 2 (graph files deleted, logs and tables kept), `_qps` for step 3.
Tables and logs go to `$OUT_DIR/construction/`.

`sweep_trials.tsv`, one row per configuration of the grid:

| Column | Meaning |
|---|---|
| `dataset system tag` | configuration |
| `uniq_out_degree weak_cc` | average out-degree without duplicate edges, weakly connected components |
| `ndist_at_r90 ndist_at_r95` | distance computations per query at Recall@10 = 0.90 and 0.95, NA if not reached |
| `max_L recall_at_max_L` | largest L of the sweep and its recall |
| `wall_s_reference` | build wall time on the search machine, reference only |

`verify.tsv`, one row per verification build: `dataset system which build_args ndist_sweep ndist_verify B result
qps_verify qps_n_timed`, where `which` is `selected` or `fallback`, `ndist_sweep` the value of the parameter search,
`ndist_verify` the value on the rebuilt graph, `result` PASS when it does not exceed B, `qps_verify` the CPU QPS
at Recall@10 = 0.9 and `qps_n_timed` the number of timed passes behind it.

`timing.tsv`, one row per configuration that passed:

| Column | Meaning |
|---|---|
| `reps` | timed builds counted (3) |
| `build_s_mean build_s_std` | self-reported build time, seconds (CT in Table 7) |
| `wall_s_mean wall_s_std` | wall time of the build process, seconds |
| `peak_host_gb peak_gpu_gb` | peak resident host memory and peak GPU memory, GiB (PMF); NA for CPU builders |
| `uniq_out_degree weak_cc` | degree (AD) and connected components (CC) of the verification graph |
| `ndist_at_r90 qps_at_r90` | distance computations and CPU QPS (THREADS threads) at Recall@10 = 0.9 on the verification graph |
| `qps_n_timed` | timed passes per L behind `qps_at_r90` (3) |

`gpu_qps/<dataset>_<system>.cpu.tsv` and `.gpu.tsv` hold the raw sweeps of the two versions of the searcher
(columns in `src/unified_search/README.md`). `gpu_qps.tsv` has `dataset system build_args ndist_at_r90_cpu
ndist_at_r90_gpu qps_cpu_at_r90 qps_gpu_at_r90 gpu_over_cpu n_timed_cpu n_timed_gpu`; `qps_gpu_at_r90` is the QPS
column of Table 7, and the last two columns are the numbers of timed passes per L of the two versions.
Table 7 leaves out PilotANN on NYTimes and GloVe200, which the scripts still build.

## Notes

- Timing: the verification build is the warm-up and is not timed, then three builds are timed. Builds run strictly
  one at a time, and `run_timing.sh` refuses to start when other processes use the GPU. The self-reported build
  times of the builders are defined in `src/build/README.md`; Table 7 reports them. HNSW, Vamana and the Faiss part
  of PilotANN use `THREADS` (14) threads, SONG is single-threaded.
- Peak GPU memory is the largest of the samples taken every 100 ms during the build minus the use before it.
- QPS: step 3 rebuilds the graphs with the selected parameters. On each graph the CPU version (14 threads) and the
  GPU version (base vectors, graph and queries in GPU memory) sweep L; for every L each runs one untimed warm-up
  pass over all queries and then `REPS` (3) timed passes, and QPS is nq divided by the mean time of the timed
  passes. The CPU QPS of `verify.tsv` and `timing.tsv` comes from the sweep of the verification build, measured
  the same way. Distance computations and QPS at Recall@10 = 0.9 are interpolated linearly between the two L whose
  recall brackets 0.9.
- The L lists are 10 to 512 on SIFT1M and DEEP1M, to 2048 on NYTimes and to 4096 on GloVe200.
- SONG reads LIBSVM text, written once per dataset to `$DATA_DIR/<dataset>/base.libsvm` (about 3 GB for GloVe200).
- GPU code is compiled for sm_70 and sm_80. For other GPUs set `GENCODE` for `src/unified_search/build.sh` and
  `src/build/build_ggnn_driver.sh`, and the architectures of the third-party builds.
