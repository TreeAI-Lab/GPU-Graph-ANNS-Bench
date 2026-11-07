# Index Loading and Resident Search

Paper: Figure 9 and Table 6.

## Settings

Table 6 reports the vector, graph, and query data transferred for an initial DEEP1M run with 10,000 queries.

Figure 9 evaluates SONG, GGNN, GANNS, CAGRA, and HNSW with 20 threads on DEEP1M and DEEP10M, and CAGRA on DEEP100M. The full dataset vectors and graph index are loaded once and remain on the GPU for up to five million held-out Deep1B queries, processed in batches of 10,000 at 95% Recall@10. On DEEP100M, SONG and GANNS run out of memory during construction, while GGNN searches one partition at a time, so only CAGRA is included. We report two measures of average time per query. For the time including loading (solid lines), we add the one-time loading of the dataset vectors and graph index to the total query execution time. We then divide this sum by the total number of queries served. The resident-index time (dashed lines) uses query execution time alone, divided by the number of queries. In both measures, query execution includes query uploads, search, and result downloads.

In the scripts, each batch of the stream is timed once, and the first batch is left out as warm-up.

## Run

Run from the repository root after `source env.sh`.

```bash
E=experiments/loading
for ds in deep1M deep10M deep100M; do bash $E/gt_5m.sh $ds; done
bash $E/run_resident_r95.sh deep1M; bash $E/run_resident_r95.sh deep10M; bash $E/run_deep100M_r95.sh
bash $E/run_h2d_bench.sh
bash experiments/equal_degree/run_hnsw.sh deep1M deep10M  # the HNSW_20 line, needs equal_degree/build_hnsw.sh
$ENV_BUILD/bin/python $E/per_query_table.py --h2d $OUT_DIR/loading/h2d/summary.tsv \
    --root $OUT_DIR/loading --hnsw-root $OUT_DIR/equal_degree/hnsw --out $OUT_DIR/loading/fig09_amortization.csv
$ENV_BUILD/bin/python $E/payload_table.py --out $OUT_DIR/loading/payload.tsv
```
