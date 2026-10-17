# Plots

One script per figure. Each reads one CSV from `results/` (or `--results DIR`) and writes PDFs to `plots/out/`
(or `--out DIR`), sized so that text is 9 pt and tick labels 8 pt at the width the figure has in the paper.

```bash
bash plots/plot_all.sh                       # every figure
python plots/fig07_qps_recall.py --results my_results --out my_pdfs
```

QPS-recall curves are drawn from measured points. A point is not drawn when a point with higher recall is at least
as fast (each curve is the Pareto frontier of its settings), and markers are kept at least 0.05 apart in normalized
plot coordinates while the line passes through every remaining point. Values are never changed.

## Input files

The summary scripts of the experiments (see `experiments/*/README.md`) produce these tables. Columns not listed are
ignored. A row with a non-empty `excluded` column is not plotted.

| Script | Figure | Input | Columns |
|---|---|---|---|
| `fig01_intro.py` | 1 | `fig01_intro_sift1m.csv` | system, recall_at_10, qps |
| `fig05_traversal_depth.py` | 5 | `fig05_traversal_depth.csv` | panel (a: search iterations, b: common-search hops), dataset, system, hops_per_query |
| `fig06_amortization.py` | 6 | `fig06_amortization.csv` | dataset, system, kind (amortized or resident), queries_served, ms_per_query |
| `fig07_qps_recall.py` | 7 | `fig07_qps_recall.csv` | dataset, system, recall_at_10, qps, excluded |
| `fig08_batch_size.py` | 8 | `fig08_batch_size.csv` | system, batch_size, qps |
| `fig09_search_components.py` | 9 | `fig09_search_components.csv` | dataset, system, gpu_cl, cpu_cl, gpu_dc, cpu_dc, gpu_dm, cpu_dm (shares), total_time_us |
| `fig10_scalability.py` | 10 | `fig10_scalability.csv` | system, dataset, time_us_per_10k (empty: out of memory) |
| `fig11_topk.py` | 11 | `fig11_topk.csv` | k, system, recall, qps |
| `fig12_degree.py` | 12 | `fig12_degree.csv` | degree, system, recall, qps |
| `fig13_cagra_parameters.py` | 13 | `fig13_cagra_parameters.csv` | parameter, value, recall_at_10, time_us (kernel time of 10,000 queries) |
| `fig14_fp16.py` | 14 | `fig14_fp16.csv` | dataset, system, precision, recall_at_10, qps, excluded |
| `fig16_reverse_edges.py` | 16 | `fig16_reverse_edges.csv` | panel, build (R or NR), it, recall_at_10, short_edge_share, long_edge_share |

System names: SONG, GGNN, GANNS, CAGRA, BANG, PilotANN, HNSW_<threads>, Vamana, and for the non-graph baselines
cuvs_ivf_pq, faiss_ivf_pq, cuvs_bf. Dataset names follow `datasets/README.md`, except in Figure 5, which uses the
display names (GIST, DEEP1M, DEEP10M, SIFT1M); in its panel (b) BANG and Vamana share one bar, BANG/Vamana, because
BANG searches the Vamana graph.

`style.py` holds the shared sizes, colors, markers and the curve rules.
