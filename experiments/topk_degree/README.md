# Top-K and Graph Degree

Paper: Figures 12 and 13.

## Settings

We further investigate the effect of result cardinality on the SIFT1M dataset. We set Top-K in {10, 40, 70, 100} and plot the QPS-Recall curve for each configuration under the same index and query workload. At every K we report HNSW with 1, 10, 20, and 40 threads. The CPU runs are limited to 14 cores, so 20 and 40 threads gain little over 10.

We compare index construction with maximum degree d in {64, 128} on SIFT1M under the same query workload and recall target. PilotANN caps the maximum degree at 64 by default, and we raise this limit to build its graph at d = 128.

In the scripts, the K = 10 curves of Figure 12 are those of Figure 7.

## Run

Run from the repository root after `source env.sh`.

```bash
E=experiments/topk_degree
bash $E/run_gpu_topk.sh
$ENV_BUILD/bin/python $E/summarize_gpu_topk.py --root $OUT_DIR/topk_degree/gpu
bash $E/run_hnsw_topk.sh
for k in 40 70 100; do K=$k bash experiments/equal_degree/run_vamana.sh sift1M; done
for s in ggnn song ganns cagra bang vamana hnsw pilotann; do bash $E/run_${s}_degree.sh; done
```
