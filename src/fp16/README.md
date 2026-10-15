# FP32 and FP16 search programs (Tables 9 and 10, Figure 14)

GPU-resident search programs for SONG, GGNN, GANNS and CAGRA in which the base and query vectors can be stored in
GPU memory as FP16 (`__half`). Each element is converted to float when it is read, and distances are computed and
accumulated in FP32. The graph is the one built for Figure 7 and is not changed. The queries are stored as FP16 as
well, because cuVS requires queries and data of the same type; the other three systems follow the same rule. The
host converts with `__float2half_rn` (round to nearest); the query conversion is outside the timed region.

Every change is behind a compile switch. With the switch off, the program is the FP32 search, and its recall at the
operating points is identical to that of the FP32 programs used for Figure 7. The FP32 and FP16 rows of Tables 9
and 10 and the curves of Figure 14 all come from these programs.

## Build

```bash
bash third_party/fetch.sh song ganns ggnn        # pinned sources in $W/src
bash src/fp16/build.sh                           # or: bash src/fp16/build.sh song ganns ggnn cagra
```

Prerequisites: `$ENV_BUILD` (CUDA 12.4 nvcc, gcc 12, cmake, glog, gflags) and `$ENV_CAGRA` (libcuvs 25.02.01 with
its nvcc). `build.sh` copies `$W/src/song`, `$W/src/ganns` and `$W/src/ggnn` to `$W/build/fp16/`, applies the
patches below, and writes these programs to `$BIN_DIR`:

| Program | System | Dataset | Built with |
|---|---|---|---|
| `song_deep1M_fp32`, `song_deep1M_fp16` | SONG | DEEP1M | `-DDIM=96 -DFIXED_DEGREE=32 -DFIXED_DEGREE_SHIFT=6` |
| `song_gist_fp32`, `song_gist_fp16` | SONG | GIST | `-DDIM=960 -DFIXED_DEGREE=128 -DFIXED_DEGREE_SHIFT=8` |
| `ganns_deep1M_fp32`, `ganns_deep1M_fp16` | GANNS | DEEP1M | `-DPLACE_HOLDER_DIM=96` |
| `ganns_gist_fp32`, `ganns_gist_fp16` | GANNS | GIST | `-DPLACE_HOLDER_DIM=960` |
| `ggnn_fp32`, `ggnn_fp16` (links `ggnn_deep1M_*`, `ggnn_gist_*`) | GGNN | both | links the patched GGNN library in `$W/build/fp16/ggnn_build` |
| `cagra_fp16` | CAGRA | both | cuVS through CMake `find_package(cuvs)`; `--precision fp32` or `fp16` |

The FP16 programs add `-DSONG_FP16`, `-DGANNS_FP16` or `-DGGNN_FP16`. SONG and GGNN are compiled for sm_70 and
sm_80, GANNS for sm_70, sm_75 and sm_80, CAGRA for sm_80; the A10 (sm_86) runs the sm_80 code.

## Files

| File | Content |
|---|---|
| `song/song_resident_search.cu` | SONG driver: base vectors and graph stay on the GPU; per batch the queries are copied in, the kernel runs, the results are copied back |
| `song/song_fp16.patch` | against SONG `9cb1f48`: see below |
| `ganns/ganns_resident_search.cu` | GANNS driver, same structure |
| `ganns/ganns_fp16.patch` | against GANNS `855e2ca`: `SearchK` reads `vec_t` vectors (`__half` with `GANNS_FP16`), `compute_dist_q` becomes a template on the vector type; `SearchK_H` keeps float |
| `ggnn/ggnn_resident_search.cpp` | GGNN driver; with `GGNN_FP16` it converts base and queries to `Dataset<__half>` |
| `ggnn/ggnn_fp16.patch` | against GGNN `2d03b36`: see below |
| `cagra/cagra_resident_fp16.cu`, `cagra/CMakeLists.txt` | CAGRA driver on the cuVS C++ API |

SONG (`song_fp16.patch`): the search kernel takes `vec_t` pointers for `d_data` and `d_query` (`__half` with
`SONG_FP16`) and reads them through `vec_load`; the query in shared memory and the accumulation stay float.
`DIM`, `FIXED_DEGREE` and `FIXED_DEGREE_SHIFT` can be set on the command line, so the GIST programs come from the
same source. The patch also contains the parts of the resident search that every SONG run of the paper uses: the
search takes `max_iterations` as an argument, the result heap holds `max_iterations - 1` entries (also the result
width), and a query stops when that heap is full or after `max_iterations` expansions (the upstream test on the
distance of the worst result is not used); counters of distance computations and expansions are added; the
`SongResidentSearch` wrapper keeps base and graph on the GPU; `TOPK`, `SEARCH_DEGREE`, `CONSTRUCT_SEARCH_BUDGET`,
`BLOOM_FILTER_BIT64` and `BLOOM_FILTER_BIT_SHIFT` are 64, 16, 160, 128 and 7. `smmh2.h` keeps the `deletion` of
commit `fa2d755` (`Y = adjust_sibling(...)`), which upstream changed in `9f42836`; the resident search was written
and run with that version.

GGNN (`ggnn_fp16.patch`): the distance code already casts the vector elements to float (`ValueT`) before it
subtracts, squares and accumulates, so `__half` vectors mean FP16 storage and FP32 accumulation without kernel
changes. The patch adds the type: `DataType::FLOAT16` and its `DataTypeAssignment` (`data.cuh`), the size and name
in `data.cu`, `__half` in `GGNN_BASES` (`lib.h`, which instantiates query, GPU instance and construction),
`Dataset<__half>` (`dataset.cu`), and the dispatch in `setBase`, `query` and `bfQuery` (`ggnn.cu`). In
`distance.cuh` and `simple_knn_sym_cache.cuh` the `? x : 0` when reading vectors becomes `BaseT(0)` and the mixed
float and half expressions get explicit casts to `ValueT`; for float these are the same values. The float and uint8
paths are unchanged.

CAGRA: cuVS supports half natively, but the Python API of cuVS 25.02 cannot save or load a half index, and building
a new index from half data would change the graph. `cagra_fp16` therefore deserializes the float index that the
CAGRA QPS sweep of Figure 7 searches (the index with its dataset, `index_with_dataset.cagra`), converts its dataset
to half on the GPU, attaches it to an `index<half>` with `update_dataset`, and passes the graph of the float index
to `update_graph`. The program checks that the graph is referenced, not copied. With `--precision fp32` it searches
the float index directly; Tables 9 and 10 use this program for both precisions, because the timed region of the
Python driver includes the Python call overhead (kernel time 5.5 ms against 4.0 ms on DEEP1M).

## Command line

All four programs take an iteration file (`--it-file`): only the first query batch is searched, and per iteration
count the program runs one untimed warm-up search and then three timed searches. `<out>/sweep.tsv` gets one row per
iteration count (`it`, `nq`, `warmup_kernel_ms`, `h2d_query_ms`, `kernel_ms`, `d2h_topk_ms`, `wall_ms`, `qps`,
`recall@10`, then `kernel_ms_std`, `wall_ms_std`, `n_timed`); the times are the means over the three timed searches,
`qps` is nq / mean `wall_ms`, and the standard deviations are sample standard deviations. GANNS adds `M` after `it`,
CAGRA `itopk`. `<out>/it_batches.tsv` has one row per search (phase `warmup`, `timed1`, `timed2`, `timed3`).
QPS in the paper is `nq / kernel_ms`. `IT_SLEEP_MS` makes the program sleep before each iteration count, that is
before its warm-up search. `experiments/profiling/run_point.sh` has the full command lines.

```bash
song_deep1M_fp16 --base base.fbin --graph bfsg.graph --query query.fbin --gt gt_k10.ibin --out OUT \
  --batch-size 10000 --gpu 0 --n-base 1000000 --dim 96 --shift 6 --max-iterations 500 --kernel-chunk 0 --it-file its.txt
ggnn_deep1M_fp16 --base base.fvecs --graph-dir <GGNN graph dir> --query query.fbin --gt gt_k10.ibin --out OUT \
  --k-build 32 --k-query 10 --tau-query 0.5 --batch-size 10000 --gpu 0 --it-file its.txt
ganns_deep1M_fp16 --base base.fvecs --graph graph.nsw --query query.fbin --gt gt_k10.ibin --out OUT \
  --k 10 --M 512 --n-e 15 --batch-size 10000 --gpu 0 --it-file its.txt
cagra_fp16 --index index_with_dataset.cagra --query query.fbin --gt gt_k10.ibin --out OUT \
  --batch-size 10000 --it-file its.txt --k 10 --precision fp16
```

GANNS computes the recall against a ground truth with exactly k columns, so the experiments pass a top-10 file
(`experiments/profiling/make_gt_k10.py`).
