# Search iteration

For all experimental results reported at 90% or 95% Recall@10, the target recall was achieved by adjusting the number of GPU iterations. The corresponding iteration settings used during search are listed below.

## 95% Recall@10

| Dataset | SONG | GGNN | GANNS | CAGRA | BANG | DiskANN | PilotANN | HNSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIFT1M | 120 | 58 | 122 | 44 | 46 | 50 | 36 | 65 |
| DEEP1M | 110 | 68 | 107 | 44 | 50 | 55 | 50 | 75 |
| DEEP10M | 80 | 57 | 83 | 39 | 45 | 48 | 34 | 75 |
| GIST | 80 | 131 | 78 | 57 | 52 | 56 | 74 | 127 |

## 90% Recall@10

The path-length table is the total over the query set. GIST has 1,000 queries and the other three datasets have 10,000, so the iteration run per query is that total divided by the query count, rounded to an integer.

| Dataset | SONG | GGNN | GANNS | CAGRA | BANG | DiskANN | PilotANN | HNSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIFT1M | 70 | 37 | 68 | 32 | 34 | 33 | 16 | 42 |
| DEEP1M | 60 | 39 | 58 | 30 | 32 | 35 | 20 | 44 |
| DEEP10M | 50 | 34 | 45 | 28 | 26 | 31 | 20 | 45 |
| GIST | 50 | 63 | 49 | 38 | 33 | 37 | 37 | 73 |
