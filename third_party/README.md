# Third-party systems

The benchmarked systems are not copied into this repository. `versions.tsv` pins each one to an
upstream commit, `patches/` holds our changes to it, `fetch.sh` downloads the sources into
`$W/src/<name>` and applies the patches, and `build/` compiles them with the conda environments
of `environment/`. All locations come from `env.sh`.

## Systems

Commits are shortened here; `versions.tsv` has the full hashes.

| Name | Upstream | Commit | Patch | What the patch changes |
|---|---|---|---|---|
| song | github.com/sunbelbd/song | 9cb1f486fd72 | none | |
| ggnn | github.com/cgtuebingen/ggnn | 2d03b36362bc | none | |
| ganns | github.com/yuyuanhang/GANNS | 855e2cad3390 | none | |
| cuvs | github.com/rapidsai/cuvs | fac323dae6ae | cuvs_no_reverse.patch | With `CUVS_CAGRA_NO_REVERSE=1`, CAGRA graph optimization skips building and merging the reverse graph and keeps the pruned edges (Figure 16) |
| bang | github.com/karthik86248/BANG-Billion-Scale-ANN | e5a7c4d768fe | bang.patch | Turns on the stage timers (`_TIMERS`) and host-side bounds checks (`_DBG_BOUNDS`), adds counters (distance computations, path length), progress output and CUDA error checks, indexes the sort kernel by R+1 instead of MAX_R+1, sets MAX_R to 128; the driver sweeps L or runs one L, with one untimed warm-up run and three timed runs per L, and prints their mean wall time; adds the converters from the DiskANN disk index to BANG's input, `bang_preprocess.py` (nodes that fit in one 4096-byte sector) and `bang_preprocess_mul.py` (nodes that span several sectors, such as GIST at R=128) |
| pilotann | github.com/ytgui/PilotANN | 0aee5f6c5d34 | pilotann.patch | `BUFFER_SIZE` 512 so that graphs of degree 128 can be searched, counters and timers in the traversal kernels, a `DataLoader` mode that reads `base.fvecs`, `query.fvecs` and `groundtruth.ivecs` from a directory, compile flags |
| faiss | github.com/facebookresearch/faiss | b0d9e80288da | none | |
| diskann | github.com/microsoft/DiskANN | 78256bbab468 | none | |
| song-orig | github.com/sunbelbd/song | fa2d7555190b | song_orig.patch | Search limit `max_iterations` as a run-time argument (queue of max_iterations - 1 entries), `path_length` and `calculate_count` counters, search and build timers, graph layout taken from `FIXED_DEGREE` and `FIXED_DEGREE_SHIFT` in `config.h`, graph and vector files at a path given on the command line, binary `.fvecs`/`.fbin` loader in place of the libsvm text parser |
| ggnn-orig | github.com/cgtuebingen/ggnn | adfcf96ab681 | ggnn_orig.patch | Search counters and per-stage clock timers (fetch, filter, distance, update), a build/query driver `ggnn_main_gpu_data` with `.fvecs`/`.fbin` input that builds in shards of 20,000,000 vectors (built as `ggnn_main_gpu_data_sharded`), sm_80 only |
| ganns-orig | github.com/yuyuanhang/GANNS | dd296b766f35 | ganns_orig.patch | A search kernel with `cal_count` and `path_length` counters next to the six stage clocks, load, query and transfer times, graph analysis (average degree, connected components), 64-bit vector offsets for MNIST8M, sm_80 in the Makefile |
| diskann-orig | github.com/microsoft/DiskANN | 1f08328e25f2 | diskann_orig.patch | `search_disk_index` reports fetch, distance and update times, hops, distance computations, path length and calc count per L; `build_disk_index` also writes the PQ pivots, centroid and chunk offsets as three separate files (`_pq_pivots.bin_myself*`) |

The first eight rows are used by the revision experiments. The four `-orig` rows are the versions
of the original manuscript, with the stage timers, counters and data loaders that its component-time
and path-length results were measured with; song-orig also builds the SONG graphs of the QPS-recall
figures, and ggnn-orig the DEEP100M GGNN graph.

SONG and GANNS compile a generated copy of their sources: SONG's `generate_template.sh` writes
`build_template/` with placeholders that `fill_parameters.sh` fills, and GANNS's
`generate_template.sh` writes `template/` with the macro files of `macro/` expanded into the kernel
headers. The instrumentation of the original runs had been edited into those generated copies.
`song_orig.patch` and `ganns_orig.patch` carry the same edits as changes to the files the generate
scripts read, so that the scripts reproduce the instrumented copies. Apart from comments and
whitespace, the only difference is that SONG reads its vectors from `<graph_dir>/bfsg.data`
instead of a fixed path.

Notes on the pins:

- BANG and PilotANN: the commits are the upstream versions that our patched copies were based on.
- cuVS: fac323dae6ae is on branch-25.02 (version 25.02.00). The tag v25.02.01 is 15 commits later
  and differs only in version numbers and packaging files. cuVS is built from source because the
  CAGRA search driver, the IVF-PQ program and the brute-force program link against it, and because
  Figure 16 needs the patched CAGRA graph optimization. CAGRA graph construction in Python uses
  the conda package libcuvs 25.02.01 (`environment/cagra.yml`).
- DiskANN: the commit is on the cpp_main branch. Its gperftools submodule is not needed; tcmalloc
  comes from the build environment.
- diskann-orig: the DiskANN version of the original manuscript's Vamana runs, commit 1f08328e25f2 on the
  cpp_main branch. The patch instruments the disk-index search and build (`search_disk_index.cpp`,
  `pq_flash_index.cpp`, `percentile_stats.h`, `pq.cpp`).

## Fetch

```bash
bash third_party/fetch.sh                 # all rows of versions.tsv
bash third_party/fetch.sh bang pilotann   # selected rows
```

Each source is fetched at its pinned commit only (shallow), checked out detached, and patched with
`git apply`. Running the script again keeps existing checkouts and skips patches that are already
applied. It stops if a checkout has other local changes.

## Build

`bash third_party/build/all.sh` fetches all sources and runs the scripts below in order, with logs
in `$W/build/logs/`. Each script can also be run on its own.

| Script | Arguments | Output | Environment |
|---|---|---|---|
| `build/diskann.sh` | | `$W/build/diskann/apps/build_memory_index`, `build_disk_index` | `$ENV_BUILD` |
| `build/ggnn.sh` | | `$W/build/ggnn/libGGNN{Base,Construction,Query}.so` | `$ENV_BUILD` |
| `build/ganns.sh` | `<DIM>` | `$W/build/ganns/dim<DIM>/GANNS` | `$ENV_BUILD` |
| `build/song.sh` | `<SD> <SHIFT> <BUDGET> <DIM>` | `$W/build/song/d<SD>_s<SHIFT>_b<BUDGET>_dim<DIM>/build_template/song` | `$ENV_BUILD` |
| `build/bang.sh` | `<R>` | `$W/build/bang/r<R>/bang_search`, `libbang.so` | `$ENV_BUILD` |
| `build/pilotann.sh` | | editable install of `$W/src/pilotann` | `$ENV_PILOT` |
| `build/faiss.sh` | | `$W/build/faiss/install` (`lib/libfaiss.so`, `include/`) | `$ENV_FAISS` |
| `build/cuvs.sh` | | `$W/build/cuvs/install` (`lib/libcuvs.so`, `include/`), dependency sources in `$W/build/cuvs/build/_deps` | `$ENV_CUVS` |
| `build/song_orig.sh` | `<FD> <SHIFT> <DIM> [TOPK] [METRIC]` | `$W/build/song-orig/fd<FD>_s<SHIFT>_dim<DIM>/song`; `_k<TOPK>` and `_<METRIC>` are appended when TOPK is not 64 or METRIC not l2 | `$ENV_BUILD` |
| `build/ggnn_orig.sh` | | `$W/build/ggnn-orig/libGGNN{Base,Construction,Query}.so`, `ggnn_main_gpu_data_sharded` | `$ENV_BUILD` |
| `build/diskann_orig.sh` | | `$W/build/diskann-orig/apps/search_disk_index`, `build_disk_index` | `$ENV_BUILD` |

Settings that matter:

- GPU architectures. SONG and GGNN are compiled for sm_70 and sm_80 (`CUDA_ARCHS` in `ggnn.sh`),
  GANNS for the 70, 75 and 80 of its CMakeLists, and PilotANN for `TORCH_CUDA_ARCH_LIST`
  (default `7.0;8.0`). BANG, Faiss and cuVS take `CUDA_ARCH` (default 80, A100 and A800).
- SONG, GANNS and BANG have compile-time parameters, so each setting gets its own build directory:
  SONG's graph degree SD, adjacency stride 2^SHIFT and insertion budget (see the comment in
  `song.sh`), GANNS's dimension, and BANG's `MAX_R`, which must equal the graph degree R.
  `all.sh` builds the lists in `GANNS_DIMS`, `BANG_RS` and `SONG_CONFIGS`; the graph-building
  scripts call the same scripts for other settings.
- The BANG search program `bang_search` sweeps L from the recall depth in steps of 8 (environment
  variable `BANG_L_STEP`) up to its maximum, or runs a single L when its eighth argument is an
  integer. For every L it runs the query batch once as an untimed warm-up and then three times
  timed, and prints a line per run and a summary line with the mean wall time of the timed runs,
  QPS = nq / mean wall time, the mean recall, the standard deviation of the wall time and the
  number of timed runs; it computes recall with `OMP_NUM_THREADS` threads (default 14).
- PilotANN's `setup.py` in the patched tree builds the C++ code with
  `-mfma -fopenmp -ffast-math -funroll-loops` (no `-mavx2`; the code uses only AVX and FMA
  intrinsics) and passes `-allow-unsupported-compiler` to nvcc.
- Faiss is built as shared libraries with the AVX2 variant, OpenBLAS, GPU support, and without
  Python bindings, C API or tests. Its GPU code includes `curand_kernel.h` and
  `cuda_profiler_api.h`, so the faiss-build environment needs the CUDA 12.6 cuRAND and profiler
  API headers.
- cuVS is configured like `./build.sh libcuvs --no-mg` without tests and benchmarks. CMake
  downloads its dependencies (RAFT, RMM, CCCL, cutlass, hnswlib and others) into
  `$W/build/cuvs/build/_deps`. If they are supplied from a local copy instead, hnswlib must be
  v0.7.0 with `cpp/cmake/patches/hnswlib.diff` applied.
- `diskann.sh` and `diskann_orig.sh` add an unversioned `libmkl_def.so` link inside
  `$ENV_BUILD/lib`, because DiskANN's CMakeLists does not find the `.so.3` file of MKL 2025 and
  later, and link with `-latomic`.
- `song_orig.sh` sets `FIXED_DEGREE` (slots per vertex; the forward degree is half of it) and
  `FIXED_DEGREE_SHIFT` (stride 2^SHIFT, which must hold FD ids and the count) in `config.h`, then runs
  `generate_template.sh` and `fill_parameters.sh <TOPK> <DIM> <METRIC>`. SONG's Makefile is left as
  it is (sm_50 to sm_75 and compute_75 PTX, which the driver compiles for newer GPUs at load time).
  The program takes `max_iterations` and the graph path as its 9th and 10th arguments; `song build`
  writes `<path>.graph`, `song test` reads `<path>/bfsg.graph` and `<path>/bfsg.data`.
- ggnn-orig has two drivers: the patched example of the GGNN tree (`ggnn_main_gpu_data_sharded`,
  which builds in shards of 20,000,000 vectors and so needs a multiple of that, as DEEP100M), and the
  driver of the original release in `experiments/original/src/ggnn`, built by
  `experiments/original/build_ggnn.sh` against the same libraries. Both are compiled with nvcc
  directly, because the example's CMake target links `gflags_nothreads_static`, which the conda
  gflags package does not provide.
- The GANNS programs of the original runs are built per dimension and distance by
  `experiments/original/build_ganns.sh` (`generate_template.sh`, `generate_build_instances.sh`,
  `generate_query_instances.sh`).
