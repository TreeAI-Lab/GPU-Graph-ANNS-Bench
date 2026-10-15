# QPS versus recall of the GPU graph systems (Figures 1, 7 and 10)

Figure 7 plots QPS against Recall@10 (k = 10) on eight datasets. These scripts produce the curves of SONG, GGNN,
GANNS and CAGRA on SIFT1M, DEEP1M, DEEP10M, GloVe200, NYTimes, GIST and DEEP100M (CAGRA and GGNN only), of BANG on
the first six of them and of PilotANN on SIFT1M, DEEP1M, DEEP10M and GIST. Each curve is a sweep of the search
effort (iterations, BANG's worklist length L, PilotANN's ef) over one batch made of all queries of the dataset.
Figure 1 is the CAGRA curve on SIFT1M from the same runs. Figure 10 is the kernel time per 10,000 queries at 90%
Recall@10 of SONG, GGNN, GANNS and CAGRA on DEEP1M, DEEP10M and DEEP100M, taken from the same runs. The MNIST8M
panel, the non-graph baselines (IVF-PQ, brute force) and the CPU curves come from other experiments, and the BANG
and PilotANN bars of Figure 10 from the runs of the original manuscript.

## Prerequisites

- Datasets under `$DATA_DIR` (see `datasets/README.md`): `sift1M`, `deep1M`, `deep10M-annb` (DEEP10M in the
  figures), `glove200`, `nytimes`, `gist`, `deep100M`, each with `base.fbin`, `query.fbin`, `gt.ibin`, and
  `base.fvecs` for GGNN and GANNS (all but DEEP100M).
- Environments `$ENV_BUILD` (CUDA 12.4 toolchain), `$ENV_CAGRA` (cuVS 25.02.01 Python), `$ENV_PILOT` (PilotANN),
  `$ENV_CUVS` (cuVS built from source), see `environment/README.md`.
- Third-party builds (`third_party/build/`): cuVS (`cuvs.sh`), GGNN libraries (`ggnn.sh`), GANNS per dimension
  (`ganns.sh <DIM>`, called when missing), DiskANN `build_disk_index` in `$W/build/diskann`, BANG with
  MAX_R = 128 in `$W/build/bang/r128` (`bang.sh 128`), PilotANN installed in `$ENV_PILOT` (`pilotann.sh`).
  The SONG graph builder is the original-manuscript SONG with the iteration-limited kernel and the fbin reader,
  compiled per FIXED_DEGREE, shift and dimension into `$W/build/song-orig/fd<FD>_s<SHIFT>_dim<DIM>/song` by the
  song-orig build of `third_party/build/`. The DEEP100M GGNN graph is built by the sharded build driver of the
  original-manuscript GGNN, `$W/build/ggnn-orig/ggnn_main_gpu_data_sharded`. BANG's converters `bang_preprocess.py`
  and `bang_preprocess_mul.py` come with `third_party/patches/bang.patch` in `$W/src/bang/BANG_Base`.
- Programs in `$BIN_DIR`: the search drivers (`bash src/search/build.sh`) and the GGNN graph builder `ggnn_build`
  (`bash src/build/build_ggnn_driver.sh`); the CAGRA graph builder is `src/build/cagra_build.py`, the PilotANN index
  builder and sweep are `src/search/pilotann/pilotann_build.py` and `pilotann_sweep.py`.
- One GPU with 80 GB of memory (A800 80GB PCIe in the paper) for DEEP100M; 14 CPU threads for BANG and PilotANN.

## Commands

```bash
source env.sh
bash src/search/build.sh
bash src/build/build_ggnn_driver.sh
bash experiments/qps_recall/build_graphs.sh            # all datasets and systems; or: build_graphs.sh gist
bash experiments/qps_recall/run_graph_systems.sh       # CAGRA, GGNN, GANNS, SONG
bash experiments/qps_recall/run_bang.sh
bash experiments/qps_recall/run_pilotann.sh
$ENV_BUILD/bin/python experiments/qps_recall/summarize.py --root $OUT_DIR/qps_recall
```

`build_graphs.sh` and `run_graph_systems.sh` take dataset ids as arguments; `SYSTEMS="cagra ggnn"` and `ONLY=song`
select systems. Finished indexes and complete sweeps are skipped unless `FORCE=1`.

## Outputs

Indexes go to `$INDEX_DIR/<data>/<system>/<tag>/` with `params.txt` and `build.log` (tags below). Results go to
`$OUT_DIR/qps_recall/<dataset>/<system>/`, and `<dataset>/gt.k10.ibin` is the top-10 slice of `gt.ibin`.

| File | Content |
|---|---|
| `<system>/sweep.tsv` (CAGRA, GGNN, GANNS, SONG) | per point the means over the three timed runs: `it`, `itopk` (CAGRA) or `M` (GANNS candidate list), `nq warmup_kernel_ms h2d_query_ms kernel_ms d2h_topk_ms wall_ms qps recall@10`, `gpu_used_mib` (CAGRA), then `kernel_ms_std wall_ms_std n_timed` |
| `<system>/it_batches.tsv` | the warm-up and the three timed runs of every point |
| `<system>/gpu_mem_cuda.tsv`, `load_timing.tsv`, `params.txt`, `run.log` | GPU and host memory after loading and after the sweep, load time, settings, log |
| `bang/raw.log`, `bang/sweep.tsv` | output of `bang_search`; per L the means over the three timed runs: `it itopk` (= L) `nq wall_ms qps recall@10`, BANG's counters and stage times `iterations calc_count path_length pq_table_ms gpu_distance_ms sort_merge_ms neighbor_filter_ms prefetch_ms l2_rerank_gpu_ms transfer_ms neighbor_seek_cpu_ms`, then `wall_ms_std n_timed` |
| `pilotann/sweep.tsv` | per ef the means over the three timed runs: `it itopk` (= ef) `nq warmup_pipeline_ms h2d_query_ms pipeline_ms d2h_topk_ms wall_ms qps recall@10 gpu_used_mib host_rss_mib svd_ms entry_ms parts_sum_ms pipeline_ms_std wall_ms_std n_timed` |
| `summary.tsv` | all points: `dataset algo it itopk nq warmup_kernel_ms h2d_query_ms kernel_ms d2h_topk_ms wall_ms qps recall@10 gpu_used_mib host_rss_mib kernel_ms_std wall_ms_std n_timed` (SONG memory left empty: its 16 GB device heap counts as used); `<dataset>/summary.tsv` per dataset |
| `fig07_points.tsv` | `dataset system param recall_at_10 qps` as plotted in Figures 1 and 7 |
| `fig10.tsv` | `system dataset time_us_per_10k` of Figure 10; empty when no point reaches 0.9 or the system was not run |

QPS in the figures: for CAGRA, GGNN, GANNS and SONG nq / `kernel_ms` (mean time of the search kernel, base and
graph already in GPU memory); for BANG and PilotANN the `qps` column, nq / mean wall time of the whole search,
because both also work on the CPU. Every point is searched once as an untimed warm-up and then timed three times. Figure 10: 10,000 / (highest such QPS among the points with Recall@10 >= 0.9), in microseconds.

## Settings

| Dataset | Data | n | Queries | Max degree D | CAGRA | GGNN | GANNS | SONG |
|---|---|---|---|---|---|---|---|---|
| SIFT1M | `sift1M` | 1,000,000 | 10,000 | 32 | `gd32_igd64_it20` | `k32_tau0.5_r2` | `k16_m64` | `fd32_shift6_deg16` |
| DEEP1M | `deep1M` | 1,000,000 | 10,000 | 32 | `gd32_igd64_it20` | `k32_tau0.5_r2` | `k16_m64` | `fd32_shift6_deg16` |
| DEEP10M | `deep10M-annb` | 9,990,000 | 10,000 | 32 | `gd32_igd64_it20` | `k32_tau0.5_r2` | `k16_m64` | `fd32_shift6_deg16` |
| GloVe200 | `glove200` | 1,183,514 | 10,000 | 64 | `gd64_igd128_it20` | `k64_tau0.5_r2` | `k32_m128` | `fd64_shift7_deg32` |
| NYTimes | `nytimes` | 289,761 | 9,991 | 32 | `gd32_igd64_it20` | `k32_tau0.5_r2` | `k16_m64` | `fd32_shift6_deg16` |
| GIST | `gist` | 1,000,000 | 1,000 | 128 | `gd128_igd256_it20` | `k128_tau0.5_r2` | `k64_m256` | `fd128_shift8_deg64` |
| DEEP100M | `deep100M` | 100,000,000 | 10,000 | 32 | `gd32_igd64_ivfpq` | `k32_tau0.5_r2` | not run | not run |

CAGRA: graph degree D, intermediate degree 2D, NN-descent with 20 iterations (`cagra_build.py`); on DEEP100M the
IVF-PQ graph builder with the default IVF-PQ parameters of cuVS (`cagra_build_ivfpq`). GGNN: KBuild D, tau_build 0.5,
2 refinement iterations; on DEEP100M 5 shards of 20,000,000 vectors. GANNS: NSW with k = D/2 and M = 2D. SONG:
FIXED_DEGREE D, forward degree D/2, `song build` with search_top_k 10, return_top_k 10, L2, max_iter 55.

| BANG (DiskANN disk index, PQ bytes 0) | R | L | M | B | QD |
|---|---|---|---|---|---|
| SIFT1M | 32 | 100 | 16 | 4 | 32 |
| DEEP1M | 64 | 100 | 16 | 4 | 32 |
| DEEP10M | 64 | 100 | 32 | 16 | 32 |
| GIST | 64 | 100 | 32 | 8 | 64 |
| GloVe200 | 96 | 100 | 40 | 4 | 40 |
| NYTimes | 64 | 100 | 32 | 4 | 32 |

PilotANN: NSG graph with n_neighbors 32 (SIFT1M, DEEP1M), 64 (DEEP10M) and 128 (GIST), ef_construct 80,
d_principle min(192, dim), sample_ratio 0.25 (tag `nsg<n>_efc80_dp192_r0.25`).

| System | Effort sweep | Timed runs per point | QPS |
|---|---|---|---|
| CAGRA | max_iterations from `it_schedule.txt` (68 values, 15 to 500); itopk = max(32, it rounded up to a multiple of 32); search width 1 | 1 warm-up, 3 timed | nq / mean kernel time |
| GGNN | max_iterations from `it_schedule.txt`; tau_query 0.5 | 1 warm-up, 3 timed | nq / mean kernel time |
| GANNS | n_e from `it_schedule.txt` | 1 warm-up, 3 timed | nq / mean kernel time |
| SONG | max_iterations from `it_schedule_song.txt` (59 values, 15 to 240) | 1 warm-up, 3 timed | nq / mean kernel time |
| BANG | L = 10 to 506, step 8 (`BANG_L_STEP`) | 1 warm-up, 3 timed | nq / mean wall time of the search call (each run in whole milliseconds) |
| PilotANN | ef = 16 to 64 step 4, 68 to 128 step 8, 144 to 256 step 16, 288 to 512 step 32 | 1 warm-up, 3 timed | nq / mean wall time of the search |

## Notes

- CAGRA (all datasets but DEEP100M) is searched through the cuVS 25.02 Python API, which cannot attach the base
  vectors to the graph-only `index.cagra`. The first sweep therefore builds the index once more with the same
  parameters, with the dataset, into `cagra/<tag>/_search_cache/index_with_dataset.cagra`, and every sweep searches
  that index. On DEEP100M the C++ driver attaches the base to the IVF-PQ graph in memory.
- SONG stops at 240 iterations: with 256 or more the kernel did not finish within the time limit of the sweep. Each
  batch is one kernel launch.
- GANNS reads the ground truth as rows of exactly 10 ids, so all drivers get `gt.k10.ibin`.
- On GIST a node of the DiskANN disk index (3,840 bytes of vector, a degree and 64 ids) exceeds one 4096-byte
  sector, so `build_graphs.sh` converts it with `bang_preprocess_mul.py` instead of `bang_preprocess.py`.
- BANG's search runs on the GPU and the CPU (neighbor reads), PilotANN's on the CPU (projection, refinement) and
  the GPU (subgraph traversal); their QPS therefore includes host work and transfers.
