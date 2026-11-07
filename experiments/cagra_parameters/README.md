# CAGRA Parameters

Paper: Figure 14.

## Settings

We study hyperparameter sensitivity on CAGRA. On SIFT1M, we sweep four knobs that jointly shape index quality and query-time behavior: graph degree, internal top-k size during search, search width during greedy/beam expansion, and team size for intra kernel parallelism. Internal top-k is the candidate size and search width is the number of points popped in an iteration. For each configuration we report QPS together with Recall@10.

In the scripts, the base setting is search width 1, internal top-k 32, graph degree 32, team size 8, and 40 iterations, and one parameter varies at a time.

## Run

Run from the repository root after `source env.sh`.

```bash
bash experiments/cagra_parameters/run_cagra_params.sh
$ENV_BUILD/bin/python experiments/cagra_parameters/summarize.py --root $OUT_DIR/cagra_parameters
```
