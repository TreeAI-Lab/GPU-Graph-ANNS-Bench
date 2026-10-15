# Reverse edges in CAGRA construction (Figure 16)

Figure 16 isolates the reverse edges that CAGRA adds when it optimizes its kNN graph. One NN-descent graph of
SIFT1M is optimized twice, with reverse edges (R) and without them (NR), so the two graphs differ only in that step.
Panel (a) is the Recall@10 of both graphs at each iteration budget under the same search settings. Panel (b) is
the share of short and long edges in a sample of 10,000 nodes with 32 neighbors each.

## Prerequisites

- Dataset `$DATA_DIR/sift1M` (`base.fbin`, `base.fvecs`, `query.fbin`, `gt.ibin`), see `datasets/README.md`.
- cuVS built from source with `third_party/patches/cuvs_no_reverse.patch` (`third_party/build/cuvs.sh`) and the
  conda environment `$ENV_CUVS`. The patch adds the switch `CUVS_CAGRA_NO_REVERSE` to `cagra::optimize`.
- Programs in `$BIN_DIR`: `cagra_build_csr`, `sample_top32` and `edge_length_stats` from `src/graph_tools`
  (`bash src/graph_tools/build.sh`), and `make_graph_to_cagra` and `attach_dataset` from `src/search`
  (`bash src/search/build.sh cagra`).
- The CAGRA sweep of Figure 7, `src/search/cagra/cagra_iter_sweep.py`, in `$ENV_CAGRA` (cuVS 25.02.01 Python).
- Python 3 with numpy (`PYTHON`, default `python3`) for `abin_to_logdeg.py` and the top-10 ground truth. One GPU.

## Commands

```bash
source env.sh
bash src/graph_tools/build.sh
bash experiments/reverse_edges/build_graphs.sh   # both graphs from one NN-descent run
bash experiments/reverse_edges/edge_stats.sh     # sample, exact top-32, short and long edges
bash experiments/reverse_edges/search.sh         # iteration sweep on both graphs
python3 experiments/reverse_edges/contrast.py    # contrast.tsv and contrast_edge.tsv
```

## Outputs

Graphs go to `$INDEX_DIR/sift1M/cagra/reverse_ab/{with_reverse,no_reverse}/`: `graph.csr`, `graph.abin`
(int32 n, int32 degree, then the neighbor ids), `params.txt`, and after `search.sh` also `graph_logdeg.bin`,
`index_graph_only.cagra` and `index.cagra` (the graph with the base vectors attached). Everything else goes to
`$OUT_DIR/reverse_edges/`, with `search/gt.k10.ibin`, the top-10 slice of `gt.ibin`.

| File | Content |
|---|---|
| `build.log`, `compare.txt` | log of `cagra_build_csr --ab`; edge overlap of the two graphs (`edges_identical_position`, `edges_same_as_set`, `mean_neighbor_jaccard`, `edges_changed_set`) |
| `sample/` | `sample_ids.ibin` (sampled node ids, sorted), `gt_top32.ibin` and `gt_top32_dist.fbin` (exact 32 nearest neighbors and their L2 distances, self excluded), `meta.txt` |
| `edge_stats/summary.tsv` | per graph: `algo n_sample K n_edges n_short n_long short_ratio mean_len_all mean_len_short mean_len_long p50_len_all p90_len_all` |
| `edge_stats/<graph>/` | `edges.tsv` (`sample_idx u v length_l2 is_short rank_in_topK`), `edges.bin` (same records, layout in `edge_stats/PROTOCOL.txt`), `summary.txt` |
| `search/<graph>/sweep.tsv` | per iteration count: `it itopk nq warmup_kernel_ms h2d_query_ms kernel_ms d2h_topk_ms wall_ms qps recall@10 gpu_used_mib kernel_ms_std wall_ms_std n_timed`, times are means over the three timed runs |
| `search/<graph>/it_batches.tsv` | the warm-up and the three timed runs of every iteration count |
| `contrast.tsv` | `it itopk nq recall@10_with_reverse qps_with_reverse kernel_ms_with_reverse recall@10_no_reverse qps_no_reverse kernel_ms_no_reverse d_recall@10 d_qps d_qps_frac d_kernel_ms`; Figure 16(a) plots the two recall columns at 15, 20, 32, 48 and 64 iterations |
| `contrast_edge.tsv` | `graph n_sample K n_edges n_short n_long short_ratio long_ratio mean_len_all mean_len_short mean_len_long p50_len_all p90_len_all`, one row per graph and a difference row; Figure 16(b) plots `short_ratio` and `long_ratio` |
| `contrast_summary.txt` | both tables in short text form |

## Settings

| Setting | Value |
|---|---|
| NN-descent graph degree | 64, 20 iterations, L2 |
| Graph degree after optimize | 32 |
| Edge sample | 10,000 base nodes drawn with seed 42, exact top-32 by cuVS brute force |
| Query batch | 10,000 (the whole SIFT1M query set), k = 10 |
| itopk | max(32, iterations rounded up to a multiple of 32) |
| Search width | 1 |
| Iterations | `experiments/qps_recall/it_schedule.txt`: 15 to 40 step 1, 42 to 64 step 2, 68 to 96 step 4, 104 to 160 step 8, 176 to 256 step 16, 288 to 480 step 32, and 500 |

## Notes

- An edge (u, v) of a sampled node u is short if v is one of the 32 exact nearest neighbors of u, and long
  otherwise. The threshold is this rank, not a distance. Both graphs are measured on the same sample.
- `cagra_build_csr --ab` runs NN-descent once and calls `cagra::optimize` on two copies of its graph, the second
  time with `CUVS_CAGRA_NO_REVERSE=1`. With the switch set, the patched `optimize` skips building and merging the
  reverse graph and keeps the pruned edges. The program sets and clears the variable itself.
  `build_graphs.sh` stops if the two graphs are identical, which happens when libcuvs is not patched.
- `sample_top32` takes an optional serialized cuVS brute-force index of the base set (`BF_INDEX`); without it the
  index is built from `base.fbin`. Both give the exact neighbors.
- The search is the CAGRA sweep of Figure 7 (`cagra_iter_sweep.py`) on an index that already holds the base
  vectors, so the optimized graph is searched as built. It times the first query batch only: for every iteration
  count one untimed warm-up run, then three timed runs. `kernel_ms` is the mean time of the search alone and `qps`
  is `nq` divided by the mean wall time of query upload, search and result download. Recall@10 is computed
  against `search/gt.k10.ibin`.
