# Search Settings at Target Recall

The tables below record the search settings for the standard graph comparisons at 90% and 95% Recall@10 on SIFT1M, DEEP1M, DEEP10M, and GIST. For each method and dataset, the search effort is selected within [10, 500] to reach the target recall. Index construction and the remaining search configuration follow [main_settings.md](main_settings.md).

The graph-quality comparison (each system's own search) and the search-time breakdown use the 90% Recall@10 points. The data-size scalability comparison and the batch-size and latency comparison use the 95% Recall@10 points. The settings are listed below.

## 95% Recall@10

| Dataset | SONG | GGNN | GANNS | CAGRA | BANG | DiskANN | PilotANN | HNSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIFT1M | 120 | 58 | 122 | 44 | 46 | 50 | 36 | 65 |
| DEEP1M | 110 | 68 | 107 | 44 | 50 | 55 | 50 | 75 |
| DEEP10M | 80 | 57 | 83 | 39 | 45 | 48 | 34 | 75 |
| GIST | 80 | 131 | 78 | 57 | 52 | 56 | 74 | 127 |

## 90% Recall@10

| Dataset | SONG | GGNN | GANNS | CAGRA | BANG | DiskANN | PilotANN | HNSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIFT1M | 70 | 37 | 68 | 32 | 34 | 33 | 16 | 42 |
| DEEP1M | 60 | 39 | 58 | 30 | 32 | 35 | 20 | 44 |
| DEEP10M | 50 | 34 | 45 | 28 | 26 | 31 | 20 | 45 |
| GIST | 50 | 63 | 49 | 38 | 33 | 37 | 37 | 73 |
