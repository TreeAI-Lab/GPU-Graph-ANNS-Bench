# Graph tools

Programs that sample graphs, measure edge lengths and build the CAGRA graphs with and without reverse edges
(Figure 16), and scripts that convert DiskANN disk indexes into the input files of BANG (Table 9).

## Edge statistics and the reverse-edge A/B build (Figure 16)

Build: `bash src/graph_tools/build.sh` writes `sample_top32`, `edge_length_stats` and `cagra_build_csr` to
`$BIN_DIR`. `sample_top32` and `cagra_build_csr` link the libcuvs built by `third_party/build/cuvs.sh`, which
carries `third_party/patches/cuvs_no_reverse.patch`; the RAPIDS headers come from that cuVS build tree (Makefile
variables `CUVS_INSTALL`, `DEPS`, `RAFT_SRC`, `CUTLASS`, `CUVS_SRC`). `edge_length_stats` needs only a C++17
compiler with OpenMP.

| Program | Usage | Output |
|---|---|---|
| `sample_top32` | `sample_top32 <data_dir> <out_dir> [S=10000] [K=32] [seed=42] [bf_index]` | `sample_ids.ibin` (S x 1, sorted ids drawn with the seed), `gt_top<K>.ibin` and `gt_top<K>_dist.fbin` (exact K nearest neighbors by cuVS brute force, L2, self excluded), `meta.txt` |
| `edge_length_stats` | `edge_length_stats <data_dir> <sample_dir> <out_dir> <name=graph.csr> [...]` | `summary.tsv` (`algo n_sample K n_edges n_short n_long short_ratio mean_len_all mean_len_short mean_len_long p50_len_all p90_len_all`), per graph `<name>/edges.tsv` (`sample_idx u v length_l2 is_short rank_in_topK`), `edges.bin`, `summary.txt`, and `PROTOCOL.txt` |
| `cagra_build_csr` | `cagra_build_csr [--ab] <base.fbin or data_dir> <out_dir> [gd=32] [igd=64] [niter=20]` | `graph.csr`, `graph.abin`, `params.txt`; with `--ab` the same files in `with_reverse/` and `no_reverse/` plus `compare.txt` |

An edge (u, v) of a sampled node u is short if v is among the K exact nearest neighbors of u in `gt_top<K>.ibin`
(K = 32), and long otherwise; the threshold is this rank, not a distance. `length_l2` is the Euclidean distance
between the two base vectors. The CSR format is int64 n, int64 nnz, int64 offsets[n+1], int32 neighbors[nnz].

`cagra_build_csr` runs cuVS NN-descent (degree igd, niter iterations, L2) and `cagra::optimize` down to degree
gd. With `--ab` it runs NN-descent once and optimizes two copies of that graph, the second with
`CUVS_CAGRA_NO_REVERSE=1`, which the patched `optimize` reads to skip building and merging the reverse edges.
`compare.txt` reports how many edges the two graphs share (`edges_identical_position`, `edges_same_as_set`,
`mean_neighbor_jaccard`, `edges_changed_set`). Without `--ab` the program honors `CUVS_CAGRA_NO_REVERSE` from the
environment.

## BANG input files (Table 9)

BANG searches a Vamana graph built by DiskANN `build_disk_index`. It reads, under one prefix, `<prefix>_disk.bin`
and `<prefix>_disk_metadata.bin` (vectors and adjacency lists), `<prefix>_pq_compressed.bin`, and the PQ pivots as
three files `<prefix>_pq_pivots.bin`, `<prefix>_pq_pivots.bin_centroid.bin` and
`<prefix>_pq_pivots.bin_chunk_offsets.bin`. `experiments/profiling/build_bang_graphs.sh` runs the steps below.
The scripts need Python 3 with numpy (`$ENV_BUILD/bin/python`).

| Script | Usage | Use |
|---|---|---|
| `disk_index_to_bang.py` | `disk_index_to_bang.py <prefix_disk.index> <prefix_disk.bin> <DIM> <DATATYPE> <DEGREE>` | writes `<prefix>_disk.bin` and `<prefix>_disk_metadata.bin`; DATATYPE 0 int8, 1 uint8, 2 float |
| `split_pq_pivots.py` | `split_pq_pivots.py <DiskANN _pq_pivots.bin> <output prefix>` | writes the three PQ pivot files in the layout BANG reads |

`disk_index_to_bang.py` writes the same format as `bang_preprocess.py` of the BANG repository: per node the vector,
the degree d, the first d neighbors in ascending order, and the other R - d slots unchanged; the metadata file holds
medoid and node length (uint64) and data type, dimension, degree and node count (uint32). On DEEP1M its output is
byte-identical to that of `bang_preprocess.py` at BANG commit `f1ec2a35`. `bang_preprocess.py` reads a fixed number
of nodes from each 4096-byte sector, which is zero when a node does not fit into one sector, as on GIST with R = 128
(960 x 4 + 4 + 128 x 4 = 4356 bytes). The converter locates nodes as DiskANN's `get_node_sector` does, so it also
handles nodes that span several sectors.

DiskANN at the pinned commit stores pivots, centroid and chunk offsets in one `_pq_pivots.bin` behind a table of
offsets. BANG reads the pivots from byte 8 of `_pq_pivots.bin` and the other two blocks from separate files.
`split_pq_pivots.py` writes each block unchanged to its own file under a new prefix and leaves the DiskANN file
as it is; the other three BANG files are linked into the new prefix.
