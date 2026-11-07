# GPU Graph Vector Search Benchmark

Code for the experiments of *An In-depth Survey and Experimental Study of GPU Graph Vector Search Algorithms*.
The study evaluates six GPU graph-based approximate nearest neighbor search algorithms (SONG, GGNN, GANNS, CAGRA, and
the CPU-GPU hybrids BANG and PilotANN) against CPU graph indexes (HNSW in Faiss, the DiskANN disk index) and GPU
non-graph indexes (IVF-PQ in Faiss and cuVS, the Brute-force Algorithm in cuVS), on eight datasets of up to 100 million vectors.

## Layout

| Path | Content |
|---|---|
| `env.sh` | paths used by every script (work directory, datasets, indexes, outputs, conda environments) |
| `environment/` | conda environments, with lock files of the exact packages used |
| `third_party/` | pinned upstream commits of every algorithm (`versions.tsv`), our patches, fetch and build scripts |
| `datasets/` | download, conversion and ground truth of the eight datasets |
| `parameters/` | build and search settings of the experiments, including the selected parameters at the comparable point |
| `src/` | our programs: search drivers, builders, the common searcher, baselines, FP16 variants, graph tools |
| `experiments/` | one folder per experiment with its run and summary scripts |
| `plots/` | one script per data figure; each reads one table and writes the PDFs |

Every script starts with a comment that gives its usage, inputs, parameters and outputs.

## Setup

```bash
source env.sh                       # set W, DATA_DIR, INDEX_DIR, OUT_DIR first to place them elsewhere
bash environment/create_envs.sh     # or --locked for the exact package builds
bash third_party/fetch.sh           # sources into $W/src, patches applied
bash third_party/build/all.sh       # third-party builds into $W/build
bash datasets/download.sh sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M
$ENV_BUILD/bin/python datasets/prepare.py sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M --gpu
```

`create_envs.sh` creates five conda environments under `$W/envs`: `build` (CUDA 12.4, gcc 12; SONG, GGNN, GANNS,
DiskANN, BANG and our drivers), `cagra` (cuVS 25.02.01 packages), `pilot` (PilotANN, faiss-cpu 1.15.1, PyTorch
2.5.1), `faiss-build` (Faiss with GPU support from source) and `cuvs-build` (cuVS 25.02 from source).

Our programs are compiled into `$BIN_DIR`:

```bash
bash src/search/build.sh all            # search drivers of CAGRA, GGNN, GANNS, SONG
bash src/build/build_ggnn_driver.sh     # GGNN graph builder
bash src/unified_search/build.sh        # common searcher, CPU and GPU
bash src/baselines/build.sh all         # Faiss HNSW, Faiss and cuVS IVF-PQ, cuVS Brute-force Algorithm
bash src/bench/build.sh                 # host-to-device copy benchmark
bash src/fp16/build.sh                  # FP32 and FP16 search programs
bash src/graph_tools/build.sh           # edge statistics, CAGRA graph conversion
E=experiments/equal_degree
bash $E/build_ggnn.sh; bash $E/build_cagra.sh   # GGNN and CAGRA drivers of the equal-degree runs
# SONG and GANNS fix the dimension and the degree at compile time: one program per dataset, per dimension and distance
for ds in sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M; do bash $E/build_song.sh $ds; done
for d in 96 128 784 960; do bash $E/build_ganns.sh $d l2; done; for d in 200 256; do bash $E/build_ganns.sh $d cos; done
```

## Algorithms

`third_party/versions.tsv` pins every algorithm to an upstream commit; `fetch.sh` downloads the sources and applies
our patches from `third_party/patches/`.

| Name | Upstream | Commit |
|---|---|---|
| song | github.com/sunbelbd/song | 9cb1f486fd72 |
| ggnn | github.com/cgtuebingen/ggnn | 2d03b36362bc |
| ganns | github.com/yuyuanhang/GANNS | 855e2cad3390 |
| cuvs | github.com/rapidsai/cuvs | 20b6775d4d0e (v25.02.01) |
| bang | github.com/karthik86248/BANG-Billion-Scale-ANN | e5a7c4d768fe |
| pilotann | github.com/ytgui/PilotANN | 0aee5f6c5d34 |
| faiss | github.com/facebookresearch/faiss | b0d9e80288da |
| faiss-hnsw | github.com/facebookresearch/faiss | 75c755c5a1fc (v1.15.1) |
| diskann | github.com/microsoft/DiskANN | 78256bbab468 |
| song-instrumented | github.com/sunbelbd/song | 9cb1f486fd72 |
| ggnn-instrumented | github.com/cgtuebingen/ggnn | 2d03b36362bc |
| ganns-instrumented | github.com/yuyuanhang/GANNS | 855e2cad3390 |
| diskann-instrumented | github.com/microsoft/DiskANN | 78256bbab468 |

An `-instrumented` build is the same commit with stage timers and counters added. These builds are used by
`experiments/equal_degree/` and build the SONG graphs and the DEEP100M GGNN graph of `experiments/qps_recall/`.

## Datasets

`download.sh` puts the raw files in `$DATA_DIR/raw`; `prepare.py` writes one directory per dataset with
`base.fbin`, `query.fbin` (int32 n, int32 d, then float32 vectors), the same vectors as fvecs, and `gt.ibin` (the
ids of the 100 nearest base vectors of every query).

| Name | Base | Queries | Dim | Source | Processing |
|---|---|---|---|---|---|
| `sift1M` | 1,000,000 | 10,000 | 128 | ann-benchmarks `sift-128-euclidean` | none |
| `gist` | 1,000,000 | 1,000 | 960 | ann-benchmarks `gist-960-euclidean` | none |
| `glove200` | 1,183,514 | 10,000 | 200 | ann-benchmarks `glove-200-angular` | L2-normalized, cosine |
| `nytimes` | 290,000 | 10,000 | 256 | ann-benchmarks `nytimes-256-angular` | L2-normalized, cosine |
| `deep1M` | 1,000,000 | 10,000 | 96 | first 1M vectors of Deep1B, official queries | ground truth computed |
| `deep10M` | 10,000,000 | 10,000 | 96 | first 10M vectors of Deep1B, official queries | ground truth computed |
| `deep100M` | 100,000,000 | 10,000 | 96 | first 100M vectors of Deep1B, official queries | ground truth computed |
| `mnist8M` | 8,090,000 | 10,000 | 784 | LIBSVM `mnist8m` (`MNIST8M_data.h5` on Hugging Face) | pixel values / 255, the last 10,000 vectors are the queries |

## Experiments

| Paper | Content | Folder | Plot |
|---|---|---|---|
| Table 3 | datasets | [`datasets/`](datasets/) | |
| Table 4 | construction with an equal maximum degree | [`equal_degree/`](experiments/equal_degree/) | |
| Table 5 | construction at a comparable recall-throughput point | [`construction/`](experiments/construction/) | |
| Figure 1 | SIFT1M, CAGRA and the non-graph indexes | [`qps_recall/`](experiments/qps_recall/), [`nongraph_baselines/`](experiments/nongraph_baselines/) | `fig01_intro.py` |
| Figure 6 | search iterations: (a) one common search, (b) each algorithm's own search | [`equal_degree/`](experiments/equal_degree/), [`traversal_depth/`](experiments/traversal_depth/) | `fig06_traversal_depth.py` |
| Figure 7 | QPS and Recall@10 on eight datasets | [`qps_recall/`](experiments/qps_recall/), [`nongraph_baselines/`](experiments/nongraph_baselines/), [`equal_degree/`](experiments/equal_degree/) | `fig07_qps_recall.py` |
| Figure 8 | time per search stage | [`equal_degree/`](experiments/equal_degree/) | `fig08_search_components.py` |
| Figure 9, Table 6 | load once, then serve batches; composition of the load | [`loading/`](experiments/loading/) | `fig09_amortization.py` |
| Figure 10 | batch size | [`batch_size/`](experiments/batch_size/) | `fig10_batch_size.py` |
| Figure 11 | QPS near 95% Recall@10 on DEEP1M, DEEP10M, DEEP100M | [`qps_recall/`](experiments/qps_recall/), [`equal_degree/`](experiments/equal_degree/) | `fig11_scalability.py` |
| Figure 12 | Recall@K for K = 10, 40, 70, 100 | [`topk_degree/`](experiments/topk_degree/), [`qps_recall/`](experiments/qps_recall/), [`equal_degree/`](experiments/equal_degree/) | `fig12_topk.py` |
| Figure 13 | maximum degree 64 and 128 | [`equal_degree/`](experiments/equal_degree/), [`topk_degree/`](experiments/topk_degree/) | `fig13_degree.py` |
| Figure 14 | CAGRA parameters | [`cagra_parameters/`](experiments/cagra_parameters/) | `fig14_cagra_parameters.py` |
| Tables 7 and 8, Figure 15 | hardware counters, FP16 vectors | [`profiling/`](experiments/profiling/) | `fig15_fp16.py` |
| Figure 17 | CAGRA with and without reverse edges | [`reverse_edges/`](experiments/reverse_edges/) | `fig17_reverse_edges.py` |
| Repository links and commits | code versions | `third_party/versions.tsv` | |

Each folder under `experiments/` has a `README.md` with its settings and commands.

Plots: `bash plots/plot_all.sh` draws every figure from the tables in `results/` (or `--results DIR`) into
`plots/out/` (or `--out DIR`). `results/` holds one CSV per figure, assembled from the summary tables of the
experiments above; the header of each plot script names its input file.

## Measurement

All experiments run on one server with Ubuntu 22.04 LTS, an Intel Xeon Gold 6348 CPU at 2.60 GHz, 120 GB of host
memory, and a single NVIDIA A800 GPU with 80 GB HBM2e memory. All algorithms are compiled at `-O3` with CUDA 12.4 and
run by benchmark drivers in C++17 and CUDA. HNSW and the DiskANN disk index run on the host CPU, and BANG and PilotANN
split their work between the CPU and the GPU as designed. No other job runs on the machine during the experiments.
The base vectors and the index are copied from pageable host memory, and query batches from pinned memory. Hardware
counters require driver administrator rights that are unavailable on the A800 server, so Tables 7 and 8 and Figure 15
run on a virtual machine with an NVIDIA A10 and 24 GB of GDDR6.

For each dataset, all graph-based algorithms perform searches on graphs constructed with the same maximum degree.
All searches return the top-10 nearest neighbors, and all queries of a dataset form one batch, except in the loading,
top-K, and batch-size experiments. Each search process starts with an empty host page cache
(`src/common/clear_caches.sh`) and empty device memory. Each setting is run once to warm up and then timed three
times, and we report the mean. For the pure GPU indexes, QPS uses the search kernel time on the resident index,
measured with CUDA events. BANG, PilotANN, HNSW, and the DiskANN disk index also work on the host, so their QPS uses
the wall-clock time of the whole search call. Query processing time starts after a batch has been formed and excludes
the waiting time needed to collect its queries.

The `README.md` of each experiment states where it differs from this protocol.

## License

The code in this repository is released under the MIT License (`LICENSE`). The patches in `third_party/patches/`
modify the listed upstream projects and are covered by their licenses.
