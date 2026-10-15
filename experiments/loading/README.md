# Loading and serving (Figure 6)

Figure 6 shows the time per query when the index is loaded once and then serves batches of 10,000 queries:
CAGRA, GGNN, GANNS and SONG on DEEP1M and DEEP10M, CAGRA on DEEP100M, each at the smallest search effort that
reaches 95% Recall@10 with a GPU-resident index. The one-time load is the measured host-to-device copy time of the
base vectors plus the CAGRA graph (5 trials each, mean), the same for all systems of a dataset. Table 8 (shares of
the loading payload on DEEP1M) is taken from the original manuscript and has no code here.

## Prerequisites

- Datasets `deep1M`, `deep10M-annb` and `deep100M` in `$DATA_DIR` (`datasets/README.md`), and the 5,000,000-query file
  `$DATA_DIR/deep100M/query.5M.fbin` built by `datasets/sample_deep1b_queries.py`: the 10,000 official Deep1B queries
  followed by Deep1B vectors outside the first 100M. All three datasets are served with this file.
- Graphs of Figure 7 under `$INDEX_DIR/<dataset>/<system>/<tag>/` (final degree about 32), built by
  `experiments/qps_recall/build_graphs.sh deep1M deep10M deep100M`:

| System | deep1M, deep10M-annb | deep100M | File |
|---|---|---|---|
| CAGRA | `gd32_igd64_it20` | `gd32_igd64_ivfpq` (graph-only, IVF-PQ graph builder) | `index.cagra` |
| GGNN | `k32_tau0.5_r2` | | the GGNN graph directory |
| GANNS | `k16_m64` | | `graph.nsw` |
| SONG | `fd32_shift6_deg16` | | `bfsg.graph` |

- Programs in `$BIN_DIR`: `ggnn_resident_search`, `ganns_resident_search_d96`, `song_resident_search_d96_fd32` and
  `cagra_resident_search` (`bash src/search/build.sh`), `h2d_copy_bench` (`bash src/bench/build.sh`).
  CAGRA on deep1M and deep10M-annb runs `cagra_resident_search.py` of this folder in `$ENV_CAGRA` (cuVS 25.02.01);
  on deep100M the C++ `cagra_resident_search` attaches the base vectors to the graph-only index.
- Environments `$ENV_CAGRA` (Python cuVS, ground truth and CAGRA driver) and `$ENV_BUILD` (Python for the summaries).

## Commands

```bash
source env.sh
bash src/bench/build.sh
E=experiments/loading

# Ground truth (top 10) of the 5M queries per dataset
bash $E/gt_5m.sh deep1M
bash $E/gt_5m.sh deep10M-annb
bash $E/gt_5m.sh deep100M
# Resident serving at 95% Recall@10: calibration on the first batch, then all 500 batches
bash $E/run_resident_r95.sh deep1M
bash $E/run_resident_r95.sh deep10M-annb
bash $E/run_deep100M_r95.sh
# One-time load: host-to-device copies of the base vectors and the CAGRA graphs
bash $E/run_h2d_bench.sh
# Figure 6 table
$ENV_BUILD/bin/python $E/per_query_table.py --h2d $OUT_DIR/loading/h2d/summary.tsv \
    --root $OUT_DIR/loading --out $OUT_DIR/loading/fig06_amortization.csv
```

## Outputs

Under `$OUT_DIR/loading/`:

| File | Content |
|---|---|
| `<dataset>/<system>/calib_r95_b10000_k10/` | `calib_sweep.tsv` (it, recall@10, probe_dir), `calib_chosen.txt` (chosen_it, achieved_recall@10), `probe_it<it>/` |
| `deep100M/cagra/calib_r95_b10000_k10_itopklin/` | the same, with an itopk column |
| `<dataset>/<system>/search_r1w1_b10000_it<it>_k10_r95/batch_timing.tsv` | batch_id, nq, h2d_query_ms, kernel_ms, d2h_topk_ms, wall_ms, qps, recall@10 |
| same directory | `params.txt`, `load_timing.tsv`, `recall.tsv`, `run.log` |
| `<dataset>/summary_resident_search_r95.tsv` | algo, chosen_it, calib_recall@10, n_batches_summary, mean_h2d_ms, mean_kernel_ms, mean_d2h_ms, mean_wall_ms, mean_qps, mean_recall@10, tag, out_dir, status (means over batches 1 to 499) |
| `h2d/summary.tsv` | kind (dataset or graph), dataset, algorithm, config, file, bytes, median_ms, min_ms, max_ms, mean_ms, median_GBps, trials_ms |
| `fig06_amortization.csv` | dataset, system, kind (amortized or resident), queries_served, ms_per_query, load_ms |
| `$DATA_DIR/<dataset>/gt.5M.k10.ibin` | ground truth of the 5M queries |

## Settings

| Setting | deep1M, deep10M-annb | deep100M |
|---|---|---|
| batch size, k | 10,000, 10 | 10,000, 10 |
| effort | max_iterations (CAGRA, GGNN, SONG), n_e (GANNS) | CAGRA max_iterations |
| candidates | 11 16 20 24 32 40 48 55 64 80 96 128 160 200 256 | the same, then 320 384 448 512 |
| CAGRA itopk_size | 64 | max(32, it) rounded up to a multiple of 32 |
| target not reached | the largest candidate is used and its recall recorded | the run stops |

- Calibration uses only batch 0 (the first 10,000 queries); between the last failing and the first passing candidate a
  binary search finds the smallest passing effort. The full run then serves all 500 batches with that effort.
- GGNN searches with KQuery 10 and tau 0.5; GANNS with a candidate list of max(64, next power of 2 >= n_e); SONG with
  one kernel launch per batch. CAGRA uses search_width 1.
- Every batch is timed with CUDA events as H2D copy of the queries (from pinned memory), search kernel and D2H copy of
  the top-10 ids. Batch 0 is a warm-up and is left out of every mean.
- `per_query_table.py`: per batch t = H2D + kernel + D2H. Amortized after n batches = (load + t of batches 1 to n) /
  (n x 10,000), for n = 1, 5, 10, 50, 100 and 499 (4,990,000 queries); resident = mean t / 10,000. The query copies
  carry the same payload for every system; as in the paper, those of the CAGRA run are used for all systems of a
  dataset (`--shared-xfer none` uses each run's own).
- The load copies `base.fvecs` (deep1M, deep10M-annb) or `base.fbin` (deep100M) and the CAGRA `index.cagra` from a
  pageable host buffer: 2 untimed and 5 timed `cudaMemcpy` calls; the figure uses the mean of the 5.
- The cuVS 25.02 Python API cannot attach a dataset to a graph-only index, so `cagra_resident_search.py` rebuilds a
  graph-only `index.cagra` once with the same parameters and caches it as `_search_cache/index_with_dataset.cagra`
  in the index directory; an existing file there is loaded directly. The Figure 7 CAGRA sweep
  (`experiments/qps_recall`) uses the same cache file, so Figures 6 and 7 search the same index.
- `gt_5m.sh` checks the first 10,000 rows against `gt.ibin` for deep1M and deep100M, whose official queries they are.
