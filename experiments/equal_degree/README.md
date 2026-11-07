# Equal-Degree Construction and Search

Paper: Table 4, Figures 6(b), 8, and 13, and the HNSW and DiskANN curves of Figure 7.

## Settings

We evaluate the construction efficiency of seven representative algorithms across eight diverse datasets ranging from medium to large scales. The evaluation focuses on four key metrics: Construction Time (CT), Average Degree (AD), Connected Components (CC), and Peak Memory Footprint (PMF) on both CPU and GPU. BANG is omitted from the tables as it adopts the same construction mechanism as DiskANN.

For graph-based algorithms, we use maximum degrees d in {32, 64, 96, 128} and construction candidate-list sizes L in {60, 120, 150, 180} for the eight datasets to build graphs. See `parameters/main_settings.md`.

Figure 8 breaks search time into candidate locating (CL), distance calculation (DC), and data structure maintenance (DM) on SIFT1M, DEEP1M, DEEP10M, and GIST. For BANG and PilotANN, it also separates the CPU and GPU portions. Figure 6(b) uses each algorithm's own search.

| Algorithm | Build parameters |
|---|---|
| SONG | FIXED_DEGREE D, construction L |
| GGNN | KBuild D, tau_build 0.3, Layer 4 |
| GANNS | NSW, d_min D/2, candidate list L |
| CAGRA | graph degree D, intermediate degree 2D, NN-descent; IVF-PQ graph construction on DEEP100M |
| BANG | Same construction settings as DiskANN: R = D, construction L, 80 GB budget |
| PilotANN | NSG with D neighbors, sample_ratio 0.25, d_principle 96 |
| HNSW | Faiss IndexHNSWFlat, M = D/2, efConstruction L |
| DiskANN disk index | `build_disk_index`, R = D, construction L, 80 GB budget (B = 80, M = 80) |
| IVF-PQ | nlist 2,048 to 131,072, m 16 to 48; per dataset and library the combination with the highest Recall@10 |

In the scripts, every index is built 5 times, and the construction time is the mean of these builds.

## Run

Run from the repository root after `source env.sh`.

```bash
E=experiments/equal_degree
bash $E/build_indexes.sh song sift1M         # song ggnn ganns cagra vamana pilotann; 5 builds each
bash $E/build_hnsw.sh                        # 5 builds each
for s in song ggnn ganns cagra bang pilotann vamana hnsw; do bash $E/run_$s.sh; done
for t in curves stages iterations build; do $ENV_BUILD/bin/python $E/tools/parse_logs.py $t --out $t.csv; done
```
