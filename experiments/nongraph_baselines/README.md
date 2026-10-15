# Non-graph GPU baselines (Figures 1 and 7)

IVF-PQ in Faiss and in cuVS and cuVS brute force, the non-graph GPU curves and points of Figure 7 and, on SIFT1M, of
Figure 1. For each dataset and library one IVF-PQ configuration (nlist, m) is chosen from k = 100 nprobe sweeps of
several configurations; that configuration is then swept again over nprobe with k = 10, which gives the curves.
Brute force gives one point per dataset.

## Prerequisites

- Datasets in `$DATA_DIR` (`datasets/README.md`): `sift1M`, `deep1M-annb`, `deep10M-annb`, `deep100M`, `gist`,
  `glove200`, `nytimes-full`, `mnist8M`. In Figure 7 they are the panels SIFT1M, DEEP1M, DEEP10M, DEEP100M, GIST,
  GloVe200, NYTimes and MNIST8M.
- Faiss and cuVS built by `third_party/build/faiss.sh` and `third_party/build/cuvs.sh`, then
  `bash src/baselines/build.sh all` (programs `faiss_ivf_pq`, `cuvs_ivf_pq`, `cuvs_brute_force` in `$BIN_DIR`).
- Environments `$ENV_FAISS` and `$ENV_CUVS` (runtime libraries), `$ENV_BUILD` (Python for the table scripts).
- One A800 80GB GPU was used. DEEP100M needs the 38.4 GB base set on the GPU for brute force.

## Commands

```bash
source env.sh
bash src/baselines/build.sh all
E=experiments/nongraph_baselines
R=$OUT_DIR/nongraph_baselines

# 1. Build the candidate indexes and sweep nprobe with k = 100 (all datasets, or name some)
bash $E/run_ivf_sweep.sh
# 2. Brute force, one search per dataset
bash $E/run_bf.sh
# 3. Tables of all sweep points, then the configuration selection
$ENV_BUILD/bin/python $E/collect_points.py --ivf-root $R/ivf_sweep --bf-root $R/bf --out $R/points
$ENV_BUILD/bin/python $E/select_configs.py --points $R/points/qps_recall_curve.csv --out $R/selection
# 4. Final runs, k = 10, on the selected configurations (indexes of step 1 are reused)
bash $E/run_ivf_k10.sh                                       # selected_configs.csv of this folder (the paper's)
SELECTED_CSV=$R/selection/selected_configs.csv bash $E/run_ivf_k10.sh   # or the regenerated selection
# 5. One table per dataset and the points of Figures 1 and 7
$ENV_BUILD/bin/python $E/make_tables.py --k10-root $R/ivf_k10 --bf-root $R/bf --out $R/tables
```

## Outputs

| File | Content |
|---|---|
| `ivf_sweep/<dataset>/<algo>/nlist<N>_m<M>/` | `build/`, `search/nprobe_<n>/survey_metrics_search.json`, `logs/`, `curve/qps_recall.csv` |
| `ivf_k10/<dataset>/<algo>/nlist<N>_m<M>/` | the same for the final k = 10 runs |
| `bf/<dataset>/cuvs_bf/` | `build/`, `search/survey_metrics_search.json`, `logs/` |
| `curve/qps_recall.csv` | algo, config, nprobe, recall_at_1, recall_at_10, recall_at_100, search_qps, nlist, m, k, search_wall_ms_std, n_timed |
| `points/qps_recall_curve.csv` | every search: family, dataset, algo, config, nlist, m, nprobe, k, nq, nb, dim, kernel_phase, kernel_ms, ann_kernel_ms, refine_kernel_ms, qps_kernel, search_qps, recall_at_1, recall_at_10, recall_at_100, kernel_ms_std, n_timed |
| `points/all_points.csv`, `ivf_all.csv`, `bf_all.csv` | the same columns with `source_json` before `kernel_ms_std` |
| `selection/selected_configs.csv` | 2 rows per dataset: config, nlist, m, operating point (recalls, nprobe, qps_kernel, kernel_ms) |
| `selection/all_candidates_ranked.csv` | every configuration, ranked within its dataset and library |
| `tables/<dataset>_qps_recall.csv` | nprobe, cuvs_config, cuvs_qps, cuvs_recall, faiss_config, faiss_qps, faiss_recall |
| `tables/bf.csv` | brute force, one row per dataset, columns of `points/all_points.csv` |
| `tables/nongraph_points.csv` | dataset (figure panel), data_dir, system (cuvs_ivf_pq, faiss_ivf_pq, cuvs_bf), param, recall_at_10, qps |

## Settings

Candidate configurations of the k = 100 sweep (`run_ivf_sweep.sh`; m must divide the dimension):

| Dataset | (nlist, m) |
|---|---|
| sift1M | (2048, 8) (2048, 16) (4096, 8) (4096, 16) (4096, 32) (8192, 16) (16384, 16) |
| deep1M-annb | (2048, 8) (2048, 16) (4096, 8) (4096, 12) (4096, 16) (8192, 16) (16384, 16) |
| deep10M-annb | (8192, 16) (16384, 8) (16384, 16) (32768, 16) |
| deep100M | (32768, 8) (32768, 16) (65536, 8) (65536, 12) (65536, 16) (131072, 16) |
| gist | (2048, 16) (2048, 32) (4096, 16) (4096, 32) (4096, 48) (8192, 32) |
| glove200 | (2048, 8) (2048, 20) (4096, 8) (4096, 20) (4096, 40) (8192, 8) (8192, 20) |
| nytimes-full | (1024, 8) (1024, 16) (2048, 8) (2048, 16) (2048, 32) (4096, 16) (4096, 32) |
| mnist8M | (8192, 16) (8192, 28) (16384, 8) (16384, 16) (16384, 28) (32768, 16) (32768, 28) |

nprobe takes the values 1, 2, 3, 4, 5, 6, 8, 10, ..., 32768 of `NPROBES` up to nlist, plus nlist/4, nlist/2 and nlist;
Faiss stops at 2048, its GPU limit, so its cells above 2048 are empty in the tables.

Selection (`select_configs.py`): for every configuration the operating point is its sweep point with the highest
recall_at_10, ties broken by recall_at_1 and then qps_kernel; the configuration with the best operating point by the
same order is selected. Run on the paper's sweeps, the script gives exactly `selected_configs.csv` of this folder:

| Dataset | cuVS | Faiss |
|---|---|---|
| sift1M | nlist4096_m32 | nlist4096_m32 |
| deep1M-annb | nlist2048_m16 | nlist8192_m16 |
| deep10M-annb | nlist32768_m16 | nlist32768_m16 |
| deep100M | nlist131072_m16 | nlist131072_m16 |
| gist | nlist4096_m48 | nlist4096_m48 |
| glove200 | nlist4096_m40 | nlist4096_m40 |
| nytimes-full | nlist2048_m32 | nlist2048_m32 |
| mnist8M | nlist8192_m28 | nlist8192_m28 |

MNIST8M was not selected by the rule: the paper uses the configuration nlist 8192, m 28 for both libraries
(`--fixed mnist8M:8192:28` of `select_configs.py`).

Notes:

- Recall. `recall_at_10` of these programs is the fraction of queries whose nearest ground-truth neighbor is among
  the first 10 returned ids (likewise `recall_at_1`, `recall_at_100`). This is the value plotted for IVF-PQ and brute
  force.
- Timing. The queries are already on the GPU. For every setting (one nprobe, or the brute-force search) the
  program searches all queries once untimed as warm-up and then three times timed, with a device synchronization
  after each search (`SEARCH_REPEATS`, default 3, for IVF-PQ; fixed at 3 for brute force). Times are the means of the
  three timed searches (`kernel_ms` the CUDA-event time, the wall time behind `search_qps`), `kernel_ms_std` and
  `search_wall_ms_std` their sample standard deviations, `n_timed` their number. Recall comes from the results of
  the last timed search.
- QPS. For IVF-PQ the tables use `search_qps` = nq / mean wall time. For brute force they use `qps_kernel` =
  nq / mean CUDA-event time. The two differ little (median 0.005%, at most 4.4% over the paper's k = 100 sweeps).
- cuVS IVF-PQ refines ceil(2 k) candidates with exact distances (`REFINE_RATIO=2`) and uses float32 lookup tables;
  Faiss uses float16 lookup tables and no refinement.
- glove200 and nytimes-full use cosine: the programs normalize the vectors and search with the inner product.
- Training: the first 10% of the base vectors, or `learn.fbin` (the first 100,000 base vectors) for MNIST8M.
- Brute force uses k = 100 except on mnist8M (k = 10), as in the paper's runs; with exact search the top 10 is the
  same for both.
- Failed builds or searches (for example out of GPU memory at a large nprobe) are listed in `failures.log` of each run
  directory and leave a gap in the curve; the scripts continue and exit with status 2.
