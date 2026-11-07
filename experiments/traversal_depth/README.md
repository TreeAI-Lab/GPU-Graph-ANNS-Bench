# Search Iterations under One Common Search

Paper: Figure 6(a).

## Settings

Unlike previous surveys that assess graph quality based on topological deviation from the exact KNNG, we adopt a search-oriented metric: the search iterations required to achieve a target accuracy. We run two groups of searches. The first searches every graph with Algorithm 1 from the same entry points, so differences come from the graph alone. The second uses each algorithm's own search and is in `experiments/equal_degree/`. We evaluated this metric on four datasets with graph degrees fixed at 128 for GIST, 64 for DEEP10M, and 32 for DEEP1M and SIFT1M, respectively, measuring the search iterations when Recall@10 reaches 90%.

## Run

Run from the repository root after `source env.sh`.

```bash
for ds in sift1M deep1M deep10M gist; do "$ENV_BUILD/bin/python" src/unified_search/entry_points.py "$DATA_DIR/$ds"; done
bash experiments/traversal_depth/run.sh
```
