# Graph builders

`build.sh` builds one graph with one of seven builders, records its build time and peak memory, converts it to the
CSR format of the common searcher (`src/unified_search`), and sweeps the searcher over L. Table 7 and Figure 5(b)
use it for every graph.

```bash
bash src/build/build.sh <system> <dataset> [KEY=VALUE ...]
bash src/build/build.sh cagra sift1M GD=8 IGD=16 NITER=5
```

## Prerequisites

| Item | How |
|---|---|
| dataset in `$DATA_DIR/<dataset>` with `entry_m8.ibin` | `datasets/README.md`, then `python src/unified_search/entry_points.py $DATA_DIR/<dataset>` |
| `$BIN_DIR/unified_search` | `bash src/unified_search/build.sh cpu` |
| SONG, GANNS | compiled on first use by `third_party/build/song.sh` and `ganns.sh`, one binary per parameter set (SONG) or dimension (GANNS) |
| GGNN | `bash third_party/build/ggnn.sh`, then `bash src/build/build_ggnn_driver.sh` (driver `$BIN_DIR/ggnn_build`) |
| Vamana | `bash third_party/build/diskann.sh` (`build_memory_index`) |
| PilotANN | `bash third_party/build/pilotann.sh` (installs into `$ENV_PILOT`) |
| CAGRA, HNSW | conda packages in `$ENV_CAGRA` (cuVS 25.02.01) and `$ENV_PILOT` (Faiss 1.15.1) |

## Systems and parameters

| System | Builder | Parameters (defaults) | Graph given to the searcher |
|---|---|---|---|
| `song` | SONG `song build`, single-threaded CPU insertion | `SD=15` degree, `SHIFT=5` log2 of the adjacency stride, `BUDGET=150` insertion search budget | `bfsg.graph` |
| `ganns` | GANNS, NSW | `K=16` neighbors per point (degree up to 2K), `M=64` candidate list size | NSW graph |
| `ggnn` | `ggnn_build.cpp` | `K=24` KBuild, `TAU=0.5` tau_build, `REFINE=2` refinement iterations | bottom layer |
| `cagra` | `cagra_build.py`, NN-descent | `GD=64` graph_degree, `IGD=128` intermediate_graph_degree, `NITER=20` NN-descent iterations | final graph |
| `pilotann` | `pilot_build.py` | `TYPE=nsg`, `NN=32` n_neighbors, `EFC=80` ef_construct, `DP=192` d_principle, `RATIO=0.25` sample_ratio | full graph (Faiss NSG), not the sampled GPU subgraph |
| `hnsw` | `hnsw_build.py`, Faiss IndexHNSWFlat | `M=32`, `EFC=40` efConstruction | layer 0 |
| `vamana` | DiskANN `build_memory_index` | `R=64` max degree, `L=100` build list size, `ALPHA=1.2` | output graph |

SONG requires `2*SD+1 <= 2^SHIFT - 1`. GANNS rounds K and M up to powers of two; in our runs it crashed with M = K
and worked with M of 2K or more. GGNN requires an even K. All builders use squared L2; the angular datasets are normalized beforehand (see `datasets/README.md`).

Other settings, as `KEY=VALUE` or in the environment: `GPU` (device), `THREADS` (CPU threads, default `nproc`),
`LS` (L list, default 10 to 512), `NQ` (queries searched, default 10000), `NOSWEEP=1` (no search), `SEARCH_REPS` (timed passes per L after the warm-up, default 3; 1 where only the distance counts are used), `CLEAN=1` (delete
the graph files, keep logs and tables), `GRAPHS` (output root, default `$INDEX_DIR`), `TAGSUFFIX` (suffix of the
output directory).

## Output

`$GRAPHS/<dataset>/<system>/<tag><TAGSUFFIX>/`, where the tag is made of the parameters (`tags.py`), for example
`gd8_igd16_it5`:

| File | Content |
|---|---|
| `params.txt` | system, dataset, parameters, start and end time, `wall_s` and `peak_rss_kb` of the build |
| `build.log` | builder output, the self-reported build time, and the GNU `time -v` report |
| `gpu_mem.txt`, `gpu_mem_baseline.txt` | GPU memory used on the device every 100 ms during the build, and before it (CAGRA, GANNS, GGNN, PilotANN) |
| `graph.csr` | the graph for the common searcher |
| `stats.tsv` | `unified_search stats` |
| `search.tsv` | `unified_search search` over `LS` |

## Build time and memory

| System | Self-reported build time in `build.log` |
|---|---|
| CAGRA | `CAGRA_BUILD_SECONDS`: from copying the base vectors to GPU memory until the cuVS build returns |
| GGNN | `GGNN_BUILD_SECONDS`: the GGNN build call, which uploads the data and copies the graph back |
| GANNS | `Running time: ... seconds`, printed by GANNS |
| HNSW | `HNSW_BUILD_SECONDS`: the Faiss `add` call |
| PilotANN | `PILOT_BUILD_SECONDS`: the whole `IndexStaged.train` (SVD, full graph, sampled subgraph) |
| Vamana | `Indexing time: ...`, printed by DiskANN |
| SONG | none; the wall time of the process is used (it includes parsing the text input) |

Apart from SONG, none of them includes file I/O or CUDA initialization. CAGRA does not include copying the graph back to the host,
because the Python interface of cuVS 25.02 does not expose the graph. The process wall time and the peak host memory
(Maximum resident set size) come from GNU `time -v`. The peak GPU memory is the largest sample minus the value
before the build. PilotANN builds on the CPU, so its peak GPU memory is 0.

## Other files

| File | Use |
|---|---|
| `to_csr.py` | converts native graph files to `graph.csr`; formats `song ganns ggnn cagra faiss_hnsw faiss_nsg diskann npz pilotann_pt bang_disk` |
| `tags.py` | the directory tag of a configuration, used by the experiment scripts |
| `build_ggnn_driver.sh` | compiles `ggnn_build.cpp` into `$BIN_DIR/ggnn_build` |
