# Common searcher

One search procedure for the graphs of every builder, used for Table 7 and Figure 5(b). It is a best-first beam
search with a candidate list of size L: each hop expands the closest unexpanded candidate and computes the squared
L2 distance of every out-neighbor not visited before; the search stops when the first L candidates are all
expanded and returns the first k. All graphs share the same entry points (the dataset medoid and 7 random points),
hierarchical graphs are searched on their bottom layer only, and directed graphs in their own direction.

The counters `ndist` (distance computations, entry points included) and `hops` (expansions) depend only on the
graph, the entry points and L. The GPU version follows the same rule, one warp per query, and gives the same counts
up to ties between exactly equal distances.

## Files

| File | Use |
|---|---|
| `unified_search.cpp` | CPU version (OpenMP), modes `search` and `stats` |
| `unified_search_gpu.cu` | GPU version, mode `search` only |
| `build.sh` | builds both into `$BIN_DIR` (`bash src/unified_search/build.sh [cpu] [gpu]`) |
| `entry_points.py` | writes `<dataset_dir>/entry_m8.ibin` (medoid first, then 7 points drawn with seed 0) |
| `operating_point.py` | value of a sweep column at Recall@10 = 0.9, interpolated between the two L around it; `--check <search.tsv> <B>` prints PASS or FAIL against a distance computation bound |
| `selfcheck_hnsw.py` | builds a Faiss HNSW index and prints the recall and distance computations of the Faiss search |

Graphs are read as `graph.csr` (int64 n, int64 nnz, int64 offsets[n+1], int32 neighbors[nnz]), written by
`src/build/to_csr.py`.

## Arguments

```
unified_search search --base base.fbin --query query.fbin --gt gt.ibin --graph graph.csr --entry entry_m8.ibin
                      --L 10,16,24 [--k 10] [--nq N] [--threads T] [--reps 3] [--hop-curve out.tsv --max-hops 400]
unified_search stats  --graph graph.csr
unified_search_gpu    --base ... --query ... --gt ... --graph ... --entry ... --L 10,16,24
                      [--k 10] [--nq N] [--reps 3] [--device 0]
```

| Argument | Meaning |
|---|---|
| `--L` | comma-separated list of candidate list sizes; one output row per L, each L at least k |
| `--k` | neighbors returned and recall depth (Recall@k against the first k ids of the ground truth) |
| `--nq` | number of queries searched (default all) |
| `--threads` | OpenMP threads of the CPU version |
| `--hop-curve`, `--max-hops` | CPU version: also write the mean recall and distance computations after each hop, up to `--max-hops` |
| `--reps` | timed passes per L after one untimed warm-up pass (default 3); `search_s` is their mean, `search_s_std` their sample standard deviation |
| `--device` | GPU version: CUDA device |

`search` prints one row per L with the columns `L k nq recall_at_k mean_ndist mean_hops p50_ndist p99_ndist
search_s search_s_std n_timed`; the GPU version prints `search_s search_s_std warps_per_block team n_timed` after
`p99_ndist`. For every L both versions search all queries once as an untimed warm-up and then `--reps` times
timed; `search_s` is the mean time of a timed pass and `n_timed` the number of timed passes. Recall, counters and
the hop curve are those of the last timed pass; every pass gives the same values. The CPU time of a pass is the
wall time of the parallel search over the queries. The GPU time of a pass covers clearing the visited bitmaps and
the search of all queries, with base vectors, graph and queries already in GPU memory.

`stats` prints `n nnz avg_out_degree min_out_degree max_out_degree zero_out self_loops dup_edges out_of_range
weak_cc avg_unique_out_degree`; the paper reports `avg_unique_out_degree` as the average degree and `weak_cc` as the
number of connected components.

## Check against a library search

On SIFT1M, a Faiss HNSW index (M 32, efConstruction 40) searched with Faiss at efSearch = 16 to 256 and with the
common searcher at L = efSearch on its bottom layer:

```bash
source env.sh
mkdir -p "$INDEX_DIR/sift1M/hnsw/selfcheck"
"$ENV_PILOT/bin/python" src/unified_search/entry_points.py "$DATA_DIR/sift1M"
"$ENV_PILOT/bin/python" src/unified_search/selfcheck_hnsw.py "$DATA_DIR/sift1M" "$INDEX_DIR/sift1M/hnsw/selfcheck/index.faiss"
"$ENV_PILOT/bin/python" src/build/to_csr.py faiss_hnsw "$INDEX_DIR/sift1M/hnsw/selfcheck/index.faiss" "$INDEX_DIR/sift1M/hnsw/selfcheck/graph.csr"
"$BIN_DIR/unified_search" search --base "$DATA_DIR/sift1M/base.fbin" --query "$DATA_DIR/sift1M/query.fbin" \
  --gt "$DATA_DIR/sift1M/gt.ibin" --graph "$INDEX_DIR/sift1M/hnsw/selfcheck/graph.csr" \
  --entry "$DATA_DIR/sift1M/entry_m8.ibin" --L 16,32,64,128,256
```

In our run the two recalls differed by at most 0.001 and the distance computations by at most 2% at every L.
