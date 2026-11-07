# Construction at a Comparable Point

Paper: Table 5.

## Settings

We tune SONG, GGNN, GANNS, CAGRA, PilotANN (only on SIFT1M and DEEP1M), HNSW, and DiskANN on NYTimes, SIFT1M, DEEP1M, and GloVe200 to reach 90% Recall@10 at a similar QPS under one common search procedure. Under this procedure, the number of distance computations per query approximates QPS. We therefore control the comparable point with a bound B on this number, so on each dataset every builder must reach 90% Recall@10 within the same B: 6000 on NYTimes, 1000 on SIFT1M and DEEP1M, and 14500 on GloVe200.

CT is the mean and standard deviation over three builds. QPS (k): thousands of queries per second at 90% Recall@10 with the search of Algorithm 1, the mean and standard deviation of five runs.

See `parameters/comparable_point_settings.md` for the selected parameters. `sweep_grid.tsv` and `tuning_log.tsv` record all tuned configurations.

## Run

Run from the repository root after `source env.sh`.

```bash
E=experiments/construction
for ds in sift1M deep1M nytimes glove200; do "$ENV_BUILD/bin/python" src/unified_search/entry_points.py "$DATA_DIR/$ds"; done
bash $E/sweep.sh                             # parameter search, counts distance computations only
"$ENV_BUILD/bin/python" $E/collect_sweep.py $E/sweep_grid.tsv "$INDEX_DIR" > "$OUT_DIR/construction/sweep_trials.tsv"
bash $E/run_timing.sh                        # one build at a time: verify, then three timed builds
bash $E/run_gpu_qps.sh                       # QPS of the common searcher
bash $E/select_cagra_qps.sh
```
