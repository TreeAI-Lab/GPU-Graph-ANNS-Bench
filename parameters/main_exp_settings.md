# Equal-Degree Hyperparameter Settings

- **Shared construction constraint:** Enforce the same maximum graph degree for all algorithms, including those whose degree is unfixed by design. 

| Dataset | Maximum graph degree |
| --- | --- |
| SIFT | 32 |
| GIST | 128 |
| GLOVE | 96 |
| NYTIMES | 64 |
| MNIST | 128 |
| DEEP1M | 32 |
| DEEP10M | 64 |
| DEEP100M | 32 |

- **SONG:** Set the priority queue size to L = 60, 120, 150, and 180 for graph degrees of 32, 64, 96, and 128, respectively.
- **GGNN:** Fix the threshold parameter at τ = 0.3.
- **GANNS:** Use NSW construction and the same L schedule as SONG.
- **CAGRA:** Use PQ-based construction for DEEP100M and NN-Descent construction for all other datasets.
- **BANG:** Use the default DiskANN settings.
- **PilotANN:** Use NSG construction.
- **HNSW:** Use the same L schedule as SONG.
- **DiskANN:** Use the same L as SONG, B = 80, M = 80
- **Other parameters:** Unless otherwise specified, use the defaults provided by the original implementations.

- **Search target:** Retrieve the top-10 nearest neighbors.
- **Search evaluation (graph indexes):** Vary only the number of search iterations from 10 to 300, inclusive, in increments of 2 to measure QPS and recall; keep all other search parameters fixed. IVF-PQ uses the nprobe sweep specified below instead.

## IVF-PQ Build and Search Parameters

### Build

The selected configurations for cuVS IVF-PQ and Faiss IVF-PQ are listed below. `nlist` is the number of coarse clusters, and `m` is the number of PQ subquantizers. MNIST8M uses the same configuration as DEEP1M for each library.

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

### Search

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
