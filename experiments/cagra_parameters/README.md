# CAGRA parameters (Figure 13)

Figure 13 varies one search parameter of CAGRA at a time on SIFT1M and keeps the others at the base setting: search
width 1, internal top-k 32, graph degree 32, team size 8, 40 iterations, k = 10. Every setting is searched with the
CAGRA sweep of Figure 7 (`src/search/cagra/cagra_iter_sweep.py`), so time and recall are measured as in Figure 7.

## Prerequisites

- Dataset `sift1M` in `$DATA_DIR` (`datasets/README.md`).
- The environment `cagra` (`$ENV_CAGRA`, libcuvs 25.02.01 Python, `environment/README.md`).

## Commands

```bash
source env.sh
bash experiments/cagra_parameters/run_cagra_params.sh     # builds the CAGRA graphs of degree 16 to 128 if missing
python3 experiments/cagra_parameters/summarize.py --root $OUT_DIR/cagra_parameters
python3 plots/fig13_cagra_parameters.py --results $OUT_DIR/cagra_parameters
```

## Outputs

| File | Content |
|---|---|
| `$INDEX_DIR/sift1M/cagra/gd<D>_igd<2D>_it20/index.cagra` | CAGRA graph of degree D (degree 32 is the graph of Figure 7) |
| `$OUT_DIR/cagra_parameters/<parameter>-<value>/sweep.tsv` | it, itopk, nq, warmup_kernel_ms, h2d_query_ms, kernel_ms, d2h_topk_ms, wall_ms, qps, recall@10, gpu_used_mib, kernel_ms_std, wall_ms_std, n_timed; one row, times are means over the three timed runs |
| `$OUT_DIR/cagra_parameters/<parameter>-<value>/params.txt` | settings of the run, including itopk_policy, search_width and team_size |
| `$OUT_DIR/cagra_parameters/fig13_cagra_parameters.csv` | parameter, value, recall_at_10, time_us, kernel_ms, wall_ms, nq |

## Settings

| Parameter | Values | Other parameters |
|---|---|---|
| search width | 1, 2, 4, 8, 16, 32 | base setting |
| internal top-k | 16, 32, 64, 128, 256 | base setting |
| graph degree | 16, 32, 64, 96, 128 | base setting |
| team size | 8, 16, 32 | base setting |

- Graphs: graph degree D, intermediate degree 2D, 20 NN-descent iterations (`src/build/cagra_build.py`), as for
  Figure 7.
- Queries: the 10,000 queries as one batch. Each setting runs once as an untimed warm-up, then three timed runs.
  `time_us` is the mean search kernel time scaled to 10,000 queries; the plot shows QPS = 10,000 / `time_us`, the QPS
  of the pure GPU systems in Figure 7.
- The iteration count stays at 40 in every run, so a longer internal top-k list is searched for at most 40
  iterations.
- cuVS rounds the internal top-k up to a multiple of 32, so the setting 16 runs as 32.
- The base values and the value lists are environment variables of `run_cagra_params.sh` (`IT`, `BASE_*`,
  `SEARCH_WIDTHS`, `ITOPKS`, `DEGREES`, `TEAM_SIZES`).
