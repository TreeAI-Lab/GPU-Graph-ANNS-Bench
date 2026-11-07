# Reverse Edges in CAGRA

Paper: Figure 17.

## Settings

Reverse edges in CAGRA construction on SIFT1M. (a) Recall@10 as the iteration budget grows. (b) Share of short and long edges. R/NR: with/without reverse edges. SE/LE: short/long edges.

In the scripts, one NN-descent graph is optimized twice with CAGRA, with and without reverse edges, so both graphs start from the same kNN graph.

## Run

Run from the repository root after `source env.sh`.

```bash
E=experiments/reverse_edges
bash $E/build_graphs.sh; bash $E/edge_stats.sh; bash $E/search.sh
$ENV_BUILD/bin/python $E/contrast.py
```
