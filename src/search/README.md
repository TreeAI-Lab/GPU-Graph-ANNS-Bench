# Search drivers

Programs that search the GPU graph indexes with the base vectors and the graph kept in GPU memory, and time each
batch of queries as query H2D, search kernel and top-k D2H with CUDA events, and the PilotANN scripts. They are used
by `experiments/qps_recall` (Figures 1, 7 and 10), `experiments/batch_size` (Figure 8), `experiments/topk_degree`
(Figure 12), `experiments/reverse_edges` (Figure 16), `experiments/loading` (Figure 6) and `experiments/profiling`.

## Build

```bash
bash src/search/build.sh            # all; or: cagra | ggnn | ganns | song
```

| Target | Needs | Binaries in `$BIN_DIR` |
|---|---|---|
| `cagra` | cuVS built from source (`third_party/build/cuvs.sh`), `$ENV_CUVS`, system `g++-11` | `cagra_resident_search`, `cagra_build_ivfpq`, `make_graph_to_cagra`, `extract_graph`, `attach_dataset` |
| `ggnn` | GGNN libraries (`third_party/build/ggnn.sh`, `$W/build/ggnn`), `$ENV_BUILD` | `ggnn_resident_search` |
| `ganns` | per-dimension GANNS copy (`third_party/build/ganns.sh <DIM>`, called if missing), `$ENV_BUILD` | `ganns_resident_search_d<DIM>` for DIM 96, 128, 200, 256, 960 |
| `song` | SONG sources `$W/src/song-orig` (commit fa2d755), `$ENV_BUILD` | `song_resident_search_d<DIM>_fd<FD>` for 96/32, 128/32, 200/64, 256/32, 960/128 |

GANNS and SONG fix the dimension (and SONG the graph layout) at compile time, hence one binary per dataset
shape. For SONG, `build.sh` assembles a workspace per binary in `$W/build/search`: eight SONG headers copied
unchanged from `$W/src/song-orig`, `astar_accelerator.h` and `config.h` from the same tree with `TOPK 64`,
`FIXED_DEGREE`, `FIXED_DEGREE_SHIFT`, `SEARCH_DEGREE 16`, `CONSTRUCT_SEARCH_BUDGET 160` and
`ACC_BATCH_SIZE 1000000`, and the one modified SONG file kept here, `song/warp_astar_accelerator.h` (iteration limit
in the kernel, counters, and the resident-search wrapper). The headers are taken from commit fa2d755 because its
`smmh2.h` differs from the later SONG commit. Compile flags: `-D__USE_L2_DIST -D__ENABLE_FIXHASH
-D__ENABLE_VISITED_DEL`, sm_80. GPU architectures can be changed with `GGNN_GENCODE`, `GANNS_GENCODE`,
`SONG_GENCODE` and, for the CAGRA tools, `make ... ARCH=`.

`common.cuh`, included by `extract_graph` and `cagra_build_ivfpq`, is the example header of cuVS
(`$W/src/cuvs/examples/cpp/src`).

## Programs

All search drivers take fbin queries and an optional ibin ground truth. With `--it-file` they search only the
first batch (`min(batch size, number of queries)` queries), and for every value in the file run the search once as
an untimed warm-up and then three times timed. A row of `sweep.tsv` holds the means over the three timed runs;
`it_batches.tsv` holds every run (phases `warmup`, `timed1` to `timed3`). Without `--it-file` they search the whole
query file batch by batch, each batch once (`batch_timing.tsv`, `recall.tsv`). Every run also writes `params.txt`,
`load_timing.tsv` and `gpu_mem_cuda.tsv`.

In `sweep.tsv`, `kernel_ms` is the mean time of the search kernel (CUDA events), `wall_ms` the mean host time around
H2D, kernel and D2H, `qps` = nq / `wall_ms`, and `recall@10` that of the timed runs. `warmup_kernel_ms` is the
kernel time of the warm-up run. Appended after the other columns: `kernel_ms_std` and `wall_ms_std` (sample standard
deviations over the timed runs) and `n_timed` (3). The figures use nq / `kernel_ms`.

| Program | Arguments (defaults) | Sweep parameter |
|---|---|---|
| `cagra/cagra_iter_sweep.py` (run with `$ENV_CAGRA/bin/python`) | `--index --base <fbin> --query --out --batch-size --it-file [--gt] [--k 10] [--search-width 1] [--gpu 0] [--cache-dir] [--graph-degree 32] [--intermediate-graph-degree 64] [--nn-descent-niter 20]`; `--itopk-size` is accepted and not used | `max_iterations`; `itopk_size` = max(32, it rounded up to a multiple of 32) |
| `cagra_resident_search` | `--index --base <fbin> --query --out [--gt] [--batch-size 10000] [--k 10] [--max-iterations 55] [--itopk-size 64] [--search-width 1] [--max-batches 0] [--gpu 0] [--it-file]` | as above |
| `ggnn_resident_search` | `--base <fvecs or fbin> --graph-dir --query --out [--gt] [--k-build 32] [--k-query 10] [--tau-query 0.5] [--max-iterations 55] [--batch-size 10000] [--gpu 0] [--max-batches 0] [--it-file] [--shard-size 0]` | `max_iterations` of the GGNN query |
| `ganns_resident_search_d<DIM>` | `--base <fvecs> --graph graph.nsw --query --out [--gt] [--k 10] [--M 64] [--n-e 55] [--batch-size 10000] [--gpu 0] [--max-batches 0] [--it-file]` | `n_e`; the candidate list per point is the smallest power of two that is at least 64, `n_e` and the adjacency slots of the graph; it must not exceed `--M` rounded up to a power of two (512 in the runs) |
| `song_resident_search_d<DIM>_fd<FD>` | `--base <fbin> --graph bfsg.graph --query --out --n-base --dim --shift [--gt] [--batch-size 10000] [--gpu 0] [--max-batches 0] [--max-iterations 55] [--recall-k 10] [--it-file] [--kernel-chunk]` (ignored) | `max_iterations`; buffers are sized for 500 iterations in a sweep |

Ground truth: GANNS reads rows of exactly `k` ids, so give it the top-10 file (`gt.k10.ibin`); the other drivers
use the first 10 ids of each row. SONG raises the CUDA device heap to 16 GB, because every query allocates its
search structures on it, and launches one kernel per batch.

CAGRA with the Python driver: cuVS 25.02 Python cannot attach a dataset to a graph-only index, so when `--index`
has no dataset the driver builds the index again with the same parameters (NN-descent) from `--base`, saves it with
the dataset as `<cache-dir>/index_with_dataset.cagra`, and searches that index; later runs load the cached file.
`cagra_resident_search` (C++) attaches the base to the graph-only index in memory instead.

Index tools (cuVS C++):

| Program | Arguments |
|---|---|
| `cagra_build_ivfpq` | `build <base.fbin or fvecs> <index_file> <degree> [mem_tsv]`: graph-only CAGRA index with the IVF-PQ graph builder (default IVF-PQ parameters of cuVS, intermediate degree 2 x degree), GPU and host memory sampled every 2 s to `mem_tsv` |
| `make_graph_to_cagra` | `<graph.bin> <index_out>`: graph-only CAGRA index from uint32 n, uint32 log2(degree), n x degree uint32 ids |
| `extract_graph` | `<index> <graph.bin>`: the inverse |
| `attach_dataset` | `<index_in> <base.fvecs> <index_out>`: index with the base vectors attached |

PilotANN (Python, run with `$ENV_PILOT/bin/python`, PilotANN installed by `third_party/build/pilotann.sh`). All take
explicit fbin query and ibin ground-truth files.

| Program | Arguments (defaults) | Timing |
|---|---|---|
| `pilotann/pilotann_build.py` | `--base <fbin> --ckpt <out.pt> --nn [--efc 80] [--dp 192] [--ratio 0.25] [--gtype nsg]`: IndexStaged (SVD, full graph, sampled GPU subgraph), d_principle capped at the dimension | build time printed as `PILOT_BUILD_SECONDS` |
| `pilotann/pilotann_sweep.py` | `--dataset --ckpt --query --gt --out [--batch 10000] [--k 10]`: ef = 16 to 64 step 4, 68 to 128 step 8, 144 to 256 step 16, 288 to 512 step 32 on the first `--batch` queries | per ef 1 untimed warm-up run, then 3 timed runs; `sweep.tsv` holds their means, with `pipeline_ms_std`, `wall_ms_std` and `n_timed` appended |
| `pilotann/pilotann_batch.py` | `--ckpt --query --gt --it-file --out [--batches 1 10 100 1000 10000]`: ef from `--it-file` for every batch size | per ef and batch size 1 untimed warm-up run, then 3 timed runs; means, with `pipeline_ms_std`, `wall_ms_std` and `n_timed` appended |

PilotANN QPS is nq / mean wall time of the whole search: query projection on the CPU, H2D of the principal
dimensions, entry search on the GPU and the overlapped GPU subgraph / CPU refinement pipeline.
