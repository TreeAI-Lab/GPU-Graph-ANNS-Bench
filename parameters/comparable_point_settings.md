# Comparable-Point Construction Settings

Selected construction parameters of each builder at the comparable recall-throughput point: 90% Recall@10 within
the bound B on distance computations per query, under the common search procedure. Parameter names follow each
implementation. The tuned grid and the selection are in `experiments/construction/` (`sweep_grid.tsv`,
`tuning_log.tsv`, `selected_configs.tsv`, `cagra_qps_selection.tsv`).

## NYTimes (B = 6000)

| Algorithm | Selected parameters |
| --- | --- |
| SONG | degree = 31, insertion budget = 177 |
| GGNN | KBuild = 28, tau_build = 0.5, refinement_iterations = 1 |
| GANNS | k = 64, M = 128 |
| CAGRA | graph_degree = 13, intermediate_graph_degree = 26, max_iterations = 20 |
| HNSW | M = 12, efConstruction = 150 |
| DiskANN | R = 36, L = 100, alpha = 1.2 |

## SIFT1M (B = 1000)

| Algorithm | Selected parameters |
| --- | --- |
| SONG | degree = 15, insertion budget = 87 |
| GGNN | KBuild = 16, tau_build = 0.5, refinement_iterations = 2 |
| GANNS | k = 32, M = 128 |
| CAGRA | graph_degree = 10, intermediate_graph_degree = 20, max_iterations = 5 |
| PilotANN | NSG, R = 10, GK = 20 |
| HNSW | M = 10, efConstruction = 35 |
| DiskANN | R = 10, L = 100, alpha = 1.2 |

## DEEP1M (B = 1000)

| Algorithm | Selected parameters |
| --- | --- |
| SONG | degree = 15, insertion budget = 75 |
| GGNN | KBuild = 16, tau_build = 0.5, refinement_iterations = 2 |
| GANNS | k = 16, M = 64 |
| CAGRA | graph_degree = 10, intermediate_graph_degree = 20, max_iterations = 5 |
| PilotANN | NSG, R = 12, GK = 20 |
| HNSW | M = 11, efConstruction = 40 |
| DiskANN | R = 12, L = 75, alpha = 1.2 |

## GloVe200 (B = 14500)

| Algorithm | Selected parameters |
| --- | --- |
| SONG | degree = 31, insertion budget = 250 |
| GGNN | KBuild = 48, tau_build = 0.5, refinement_iterations = 2 |
| GANNS | k = 64, M = 128 |
| CAGRA | graph_degree = 32, intermediate_graph_degree = 64, max_iterations = 10 |
| HNSW | M = 16, efConstruction = 200 |
| DiskANN | R = 34, L = 100, alpha = 1.2 |
