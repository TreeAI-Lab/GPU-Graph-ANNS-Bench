# Equal-Degree Hyperparameter Settings

These settings follow the VLDB Journal manuscript, Section 5, Experimental Settings
(`vldb-journal/section/sec5-experiments.tex` in the manuscript repository). They apply to the
equal-degree comparison; the comparable recall-throughput construction experiment and parameter
sensitivity experiments specify their own configurations.

- **Shared construction constraint:** Enforce the same maximum graph degree for all algorithms, including those whose degree is unfixed by design. 

| Dataset | Maximum graph degree |
| --- | --- |
| SIFT1M | 32 |
| GIST | 128 |
| GloVe200 | 96 |
| NYTimes | 64 |
| MNIST8M | 128 |
| DEEP1M | 32 |
| DEEP10M | 64 |
| DEEP100M | 32 |

- **SONG:** Set the priority queue size to L = 60, 120, 150, and 180 for graph degrees of 32, 64, 96, and 128, respectively.
- **GGNN:** Fix the construction slack at τ = 0.3 and use Layer = 4.
- **GANNS:** Use NSW construction and the same L schedule as SONG.
- **CAGRA:** Use PQ-based construction for DEEP100M and NN-Descent construction for all other datasets.
- **BANG:** Use the same construction settings as DiskANN, including the degree-dependent L schedule and the 80 GB budget.
- **PilotANN:** Use NSG construction with `sample_ratio = 0.25` and `d_principle = 96`.
- **HNSW:** Use the same L schedule as SONG.
- **DiskANN:** Use the same L schedule as SONG and an 80 GB memory budget (`B = 80`, `M = 80`).
- **Other parameters:** Unless otherwise specified, use the defaults provided by the original implementations.

- **Search target:** Retrieve the top-10 nearest neighbors.
- **Search evaluation (graph indexes):** Vary only the number of search iterations in [10, 500] to measure QPS and Recall@10; keep all other search parameters fixed. Set the search width to 1. For the GPU algorithms, set the batch size to the number of queries (10,000, except 1,000 for GIST). For CAGRA warp splitting, set the team size to 8. The manuscript does not specify a uniform iteration step; `qps_recall/run_graph_systems.sh` uses a step of 2, while the target-recall settings are listed separately in `subsec_settings.md`. IVF-PQ uses the nprobe sweep specified below instead.
- **Target recall:** For the standard graph comparisons at 0.90 or 0.95 Recall@10, select the iteration setting to reach the target while keeping the other search settings unchanged. Loading, top-K, and batch-size experiments use the workload variations described in the manuscript.
- **Final iteration range:** From that sweep, keep the points at which Recall@10 is at least 0.6. The table lists the first iteration count at which each system reaches this recall; reported points run from that iteration through 500. MNIST8M is the exception: every measured point already has Recall@10 above 0.8, so its curves start at 0.85 instead. An em dash means the system is not reported on that dataset.

| Dataset | SONG | GGNN | GANNS | CAGRA | BANG | DiskANN | PilotANN | HNSW |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| SIFT1M | 30 | 15 | 10 | 12 | 10 | 10 | 10 | 10 |
| DEEP1M | 20 | 14 | 12 | 12 | 10 | 10 | 10 | 10 |
| DEEP10M | 18 | 20 | 10 | 12 | 10 | 10 | 10 | 10 |
| DEEP100M | — | 20 | — | 30 | 12 | 12 | — | — |
| GIST | 20 | 15 | 12 | 16 | 12 | 10 | 10 | 12 |
| GLOVE200 | 50 | 25 | 20 | 40 | 30 | 10 | — | 12 |
| MNIST8M | 26 | 20 | 20 | 10 | 10 | 10 | 10 | 10 |
| NYTIMES | 38 | 20 | 30 | 20 | 20 | 70 | — | 10 |

# IVF-PQ Build and Search Parameters

## Build

Test configurations with `nlist` from 2,048 to 131,072 and `m` from 16 to 48, using 8 bits per subquantizer. The selected configurations for cuVS IVF-PQ and Faiss IVF-PQ are listed below. `nlist` is the number of coarse clusters, and `m` is the number of PQ subquantizers. MNIST8M uses the same configuration as DEEP1M for each library.

| Dataset | cuVS nlist | cuVS m | Faiss nlist | Faiss m |
| --- | --- | --- | --- | --- |
| SIFT1M | 4096 | 32 | 4096 | 32 |
| GIST | 4096 | 48 | 4096 | 48 |
| GLOVE200 | 4096 | 40 | 4096 | 40 |
| NYTIMES | 2048 | 32 | 2048 | 32 |
| MNIST8M | 2048 | 16 | 8192 | 16 |
| DEEP1M | 2048 | 16 | 8192 | 16 |
| DEEP10M | 32768 | 16 | 32768 | 16 |
| DEEP100M | 131072 | 16 | 131072 | 16 |

- **PQ code size:** Use 8 bits per subquantizer in both libraries.
- **Training data:** Use the first 10% of the base vectors (`LEARN_RATIO = 0.1`) to train the coarse quantizer and PQ codebooks, then add all base vectors. cuVS uses `kmeans_trainset_fraction = 1.0` on this training subset and `add_data_on_build = false`, followed by a separate extend step.
- **Distance metric:** Use cosine similarity for GLOVE200 and NYTIMES by L2-normalizing vectors and using inner product; use L2 distance for all other datasets.
- **Configuration selection:** Select one `(nlist, m)` per dataset and library using the preliminary `k = 100` nprobe sweeps. Prefer the configuration with the highest Recall@10.
- **Other build parameters:** Use the defaults of the corresponding implementation.

## Search

- **Search target:** Set `k = 10`; reuse the selected indexes without rebuilding them.
- **Sweep parameter:** Vary only `nprobe`, keeping the selected `(nlist, m)` and other search parameters fixed. Use the following candidate values:

  ```text
  1 2 3 4 5 6 8 10 12 14 16 20 24 28 32 40 48 56 64
  80 96 112 128 160 192 224 256 320 384 448 512 640 768 896
  1024 1280 1536 1792 2048 2560 3072 3584 4096 5120 6144
  8192 10240 12288 16384 20480 24576 32768
  ```

- **Sweep limits:** Retain values up to `min(nlist, 2048)` for Faiss and `min(nlist, 32768)` for cuVS. Also include `nlist / 4`, `nlist / 2`, and `nlist` when within the corresponding limit; sort and deduplicate all values.
- **cuVS precision and refinement:** Use float32 lookup tables and internal distances (`CUVS_FP16_LUT = 0`, the default). Set `REFINE_RATIO = 2`: retrieve 20 candidates and rerank them using exact distances to return the top 10.
- **Faiss precision:** Enable float16 lookup tables at index construction (`useFloat16LookupTables = true`). The separate search mode loads the serialized index with `index_cpu_to_gpu` using its default clone options; no additional exact refinement step is applied.
- **Measurement:** Perform one untimed warm-up and three timed searches per setting (`SEARCH_REPEATS = 3`). Plot Recall@10 against QPS computed as the number of queries divided by mean CUDA-event search time. For cuVS, include both ANN search and refinement in this time (`search_e2e`); for Faiss, use the `search` phase.
- **Final curves:** Use the `k = 10` nprobe sweep of each selected configuration. Plot measured points in Recall@10 order without Pareto-envelope selection.
