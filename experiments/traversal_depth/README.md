# Common-search hops

Figure 5(b) separates the traversal depth that a graph requires from what each system's own search adds or saves.
The graphs of SONG, GGNN, GANNS, CAGRA, PilotANN, HNSW and Vamana are rebuilt with the settings of the original
manuscript on GIST, DEEP1M, DEEP10M and SIFT1M, and every graph is searched with the common searcher
(`src/unified_search`) from the same entry points. The figure shows the mean hops per query at Recall@10 = 0.9.
BANG searches the Vamana graph, so the two share one bar.

## Prerequisites

- Datasets `sift1M`, `deep1M`, `deep10M` (first 10M vectors of Deep1B), `gist` in `$DATA_DIR` (`datasets/README.md`).
- Conda environments `build`, `cagra`, `pilot`, and the programs listed in `src/build/README.md`.
- Any GPU of sm_70 or sm_80 (or set `GENCODE`). Only counts are measured, so the machine need not be exclusive.

## Commands

```bash
source env.sh
for ds in sift1M deep1M deep10M gist; do "$ENV_BUILD/bin/python" src/unified_search/entry_points.py "$DATA_DIR/$ds"; done
GPUS="0 1 2 3" bash experiments/traversal_depth/run.sh
```

`run.sh` runs all SONG builds at the same time and deals the other configurations to the devices in `GPUS`
(default `$GPU`), each build with `THREADS` (16) CPU threads. Finished configurations are skipped on a second run.
At the end it calls `collect.py`, which can also be run alone:
`"$ENV_BUILD/bin/python" experiments/traversal_depth/collect.py "$INDEX_DIR" "$OUT_DIR/traversal_depth/hops.tsv"`.
In our run the slowest builds were SONG on DEEP10M and GIST, about 6 and 5 hours.

## Build settings

All builders on a dataset get the same maximum out-degree D: 128 on GIST, 64 on DEEP10M, 32 on DEEP1M and SIFT1M.
The queue length Lq of SONG, GANNS and HNSW follows D: 60, 120 and 180 for D = 32, 64 and 128. GGNN uses
tau_build = 0.3. All other parameters keep the defaults of each implementation.

| System | Parameters |
|---|---|
| SONG | degree `SD = D/2 - 1` (stored at most `2 SD + 1 = D - 1` neighbors), `SHIFT` the smallest that fits, insertion budget `Lq` |
| GANNS | NSW, `K = D/2` (degree up to 2K = D), `M` = Lq rounded up to a power of two (64, 128, 256) |
| GGNN | `K = D`, `TAU = 0.3`, `REFINE = 2` |
| CAGRA | `GD = D`, `IGD = 128`, `NITER = 20` |
| HNSW | `M = D/2` (bottom-layer degree up to 2M = D), `EFC = Lq` |
| PilotANN | NSG, `NN = D`, `EFC = 80`; on GIST `NN = 64` |
| Vamana | `R = D`, `L = 100`, `ALPHA = 1.2` |

PilotANN uses NN = 64 on GIST because its own GPU search accepts only 32 or 64 neighbors (`4 * n_neighbors <= 256`
and `n_neighbors % 32 == 0` in `extension/traverse_cuda.cu`); this keeps the degree at which its own search ran
in Figure 5(a). `configs.tsv` lists the 28 configurations (`dataset max_degree system build_args`).

## Output

Graphs are kept in `$INDEX_DIR/<dataset>/<system>/<tag>_depth/` (`src/build/README.md`), logs in
`$OUT_DIR/traversal_depth/logs/`. `$OUT_DIR/traversal_depth/hops.tsv` has one row per configuration:

| Column | Meaning |
|---|---|
| `dataset max_degree system build_args` | configuration |
| `uniq_out_degree weak_cc` | average out-degree without duplicate edges, weakly connected components |
| `nq` | queries searched (1,000 on GIST, 10,000 elsewhere) |
| `L_at_r90` | the two L of the sweep whose Recall@10 brackets 0.9 |
| `hops_at_r90` | mean hops per query at Recall@10 = 0.9 (plotted in Figure 5(b)) |
| `ndist_at_r90` | mean distance computations per query at Recall@10 = 0.9 |
| `max_recall` | highest Recall@10 of the sweep |

## Notes

- The sweep uses L = 10, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048. Hops and distance
  computations at Recall@10 = 0.9 are interpolated linearly between the two L around it; NA means the sweep never
  reached 0.9.
- An equal maximum degree is not an equal actual degree: pruning builders (HNSW, PilotANN and Vamana on GIST) stay
  far below D, so `uniq_out_degree` belongs next to the hops.
- The suffix `_depth` keeps these graphs apart from graphs of the construction experiment with the same parameters
  (DEEP1M GANNS K 16, M 64 occurs in both).
