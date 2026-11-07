# Non-Graph GPU Baselines

Paper: IVF-PQ and the Brute-force Algorithm in Figures 1 and 7.

## Settings

For cuVS and Faiss IVF-PQ, we evaluate configurations with nlist in [2,048, 131,072] and m in [16, 48], using 8 bits per subquantizer. For each configuration, we sweep nprobe in [1, min(nlist, 32,768)] for cuVS and nprobe in [1, min(nlist, 2,048)] for Faiss.

In the scripts, `select_configs.py` selects the configuration for each dataset and library.

## Run

Run from the repository root after `source env.sh`.

```bash
E=experiments/nongraph_baselines; R=$OUT_DIR/nongraph_baselines
bash $E/run_ivf_sweep.sh                     # candidate indexes, nprobe sweep with k = 100
bash $E/run_bf.sh
$ENV_BUILD/bin/python $E/collect_points.py --ivf-root $R/ivf_sweep --bf-root $R/bf --out $R/points
$ENV_BUILD/bin/python $E/select_configs.py --points $R/points/qps_recall_curve.csv --out $R/selection
SELECTED_CSV=$R/selection/selected_configs.csv bash $E/run_ivf_k10.sh
$ENV_BUILD/bin/python $E/make_tables.py --k10-root $R/ivf_k10 --bf-root $R/bf --out $R/tables
```
