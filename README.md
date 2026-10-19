# GPU Graph Vector Search Benchmark

Code for the experiments of *A Survey and Experimental Study of GPU-Accelerated Graph Vector Search Algorithms*.
The study evaluates six GPU graph-based approximate nearest neighbor search systems (SONG, GGNN, GANNS, CAGRA, and
the CPU-GPU hybrids BANG and PilotANN) against CPU graph indexes (HNSW, Vamana) and GPU non-graph indexes (IVF-PQ in
Faiss and cuVS, brute force in cuVS), on eight datasets of up to 100 million vectors.

## Layout

| Path | Content |
|---|---|
| `env.sh` | paths used by every script (work directory, datasets, indexes, outputs, conda environments) |
| `environment/` | conda environments, with lock files of the exact packages used |
| `third_party/` | pinned upstream commits of every system, our patches, fetch and build scripts |
| `datasets/` | download, conversion and ground truth of the eight datasets |
| `src/` | our programs: search drivers, builders, the common searcher, baselines, FP16 variants, graph tools |
| `experiments/` | one folder per experiment with run scripts, summary scripts and a README |
| `plots/` | one script per figure; each reads one table and writes the PDFs |

## Setup

```bash
source env.sh                       # set W, DATA_DIR, INDEX_DIR, OUT_DIR first to place them elsewhere
bash environment/create_envs.sh     # or --locked for the exact package builds
bash third_party/fetch.sh           # sources into $W/src, patches applied
bash third_party/build/all.sh       # third-party builds into $W/build
bash datasets/download.sh sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M
python datasets/prepare.py sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M --gpu
```

Our programs are compiled into `$BIN_DIR` by the build scripts in `src/`; the experiment READMEs say which ones
they need:

```bash
bash src/search/build.sh all            # search drivers of CAGRA, GGNN, GANNS, SONG
bash src/build/build_ggnn_driver.sh     # GGNN graph builder
bash src/unified_search/build.sh        # common searcher, CPU and GPU
bash src/baselines/build.sh all         # Faiss HNSW, Faiss and cuVS IVF-PQ, cuVS brute force
bash src/bench/build.sh                 # host-to-device copy benchmark
bash src/fp16/build.sh                  # FP32 and FP16 search programs
bash src/graph_tools/build.sh           # edge statistics, CAGRA graph conversion
```

## Paper items

| Paper | Content | Experiment | Plot |
|---|---|---|---|
| Table 3 | code versions | `third_party/versions.tsv` | |
| Table 4 | datasets | `datasets/` | |
| Tables 5, 6 | construction with an equal maximum degree | `original/` | |
| Table 7 | construction at a comparable recall-throughput point | `construction/` | |
| Figure 1 | SIFT1M, CAGRA and the non-graph indexes | `qps_recall/`, `nongraph_baselines/` | `fig01_intro.py` |
| Figure 5 | (a) search iterations, (b) hops of the common searcher | `original/`, `traversal_depth/` | `fig05_traversal_depth.py` |
| Figure 6 | load once, then serve batches | `loading/` | `fig06_amortization.py` |
| Table 8 | composition of the loading payload | measured for the original manuscript, no script | |
| Figure 7 | QPS and Recall@10 on eight datasets | `qps_recall/`, `nongraph_baselines/`, `original/` | `fig07_qps_recall.py` |
| Figure 8 | batch size | `batch_size/` | `fig08_batch_size.py` |
| Figure 9 | time per search stage | `original/` | `fig09_search_components.py` |
| Figure 10 | QPS at 95% Recall@10 on DEEP1M, DEEP10M, DEEP100M | `qps_recall/`, `original/` | `fig10_scalability.py` |
| Figure 11 | Recall@K for K = 10, 40, 70, 100 | `topk_degree/`, `qps_recall/`, `original/` | `fig11_topk.py` |
| Figure 12 | maximum degree 64 and 128 | `original/`, `topk_degree/` | `fig12_degree.py` |
| Figure 13 | CAGRA parameters | `cagra_parameters/` | `fig13_cagra_parameters.py` |
| Tables 9, 10, Figure 14 | hardware counters, FP16 vectors | `profiling/` | `fig14_fp16.py` |
| Figure 16 | CAGRA with and without reverse edges | `reverse_edges/` | `fig16_reverse_edges.py` |

Experiment folders are under `experiments/`. Some curves come from the measurements of the original manuscript
(`experiments/original/`); the README there lists them and says which of their steps have scripts.

## Measurement

All experiments except the hardware-counter ones ran on one server with Ubuntu 22.04, an Intel Xeon Gold 6348 CPU
and an NVIDIA A800 80GB PCIe GPU. Tables 9 and 10 and Figure 14 ran on an NVIDIA A10 24GB, because reading
performance counters requires administrator rights on the GPU.

Every search returns the top 10 neighbors, and Recall@10 is the fraction of the 10 true nearest neighbors found.
Each setting (iteration count, L, ef, nprobe, batch size, operating point) is searched once as an untimed warm-up
and then timed three times; the mean of the three timed runs is reported, and recall is that of the timed runs. For
the pure GPU systems QPS is the number of queries divided by the mean search kernel time; for BANG, PilotANN and the
CPU indexes it uses the mean time of the whole search call. Figure 6 is a serving measurement instead: every batch of
the stream is timed once and the first batch is left out as warm-up. The experiment READMEs give the details.

## License

The code in this repository is released under the MIT License (`LICENSE`). The patches in `third_party/patches/`
modify the listed upstream projects and are covered by their licenses.
