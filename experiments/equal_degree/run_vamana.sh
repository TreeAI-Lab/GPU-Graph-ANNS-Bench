#!/bin/bash
# Vamana search (Figures 5(a), 7): DiskANN search_disk_index on the disk index that BANG also searches,
# 48 threads, 1,000 cached nodes, one process per list size L. The process searches all queries with the same L
# WARMUP_RUNS + TIMED_RUNS times (default 1 + 3): L is given that many times to -L, and the result table has one
# row per search. tools/parse_logs.py drops the warm-up rows, whose number the first line of the log records, and
# averages the timed rows. The search program is search_disk_index of diskann-instrumented (built by
# build_search_disk_index.sh); its result table has seven more columns (fetch, calc and update time, hops,
# distance computations, path length, calc count).
#
# Usage:  bash run_vamana.sh [dataset ...]    (default: all eight datasets)
# Needs:  build_search_disk_index.sh, build_indexes.sh vamana <dataset> [degree]
# Output: $OUT_DIR/equal_degree/vamana/<run>/k<K>/L<L>.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

K=${K:-10}
THREADS=${THREADS:-48}
CACHE_NODES=${CACHE_NODES:-1000}
L_LIST=${L_LIST:-$(seq 10 2 500)}
DISKANN_APPS=${DISKANN_APPS:-$W/build/diskann-instrumented/apps}
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M)

for ds in "${DATASETS[@]}"; do
  deg=${DEGREE:-$(degree_of "$ds")}
  case $ds in glove200|nytimes) dist=cosine ;; *) dist=l2 ;; esac
  run=$ds; [ -z "${DEGREE:-}" ] || run=$ds-d$deg
  prefix=$INDEX_DIR/$ds/vamana/eqdeg-d$deg/vamana
  out=$EQDEG_OUT/vamana/$run/k$K
  write_meta "$out" vamana "$ds" "$K"
  for L in $L_LIST; do
    Ls=()
    for ((i = 0; i < WARMUP_RUNS + TIMED_RUNS; i++)); do Ls+=("$L"); done
    clear_caches
    {
      echo "### runs per L: warmup $WARMUP_RUNS, timed $TIMED_RUNS"
      "$DISKANN_APPS/search_disk_index" --data_type float --dist_fn "$dist" --index_path_prefix "$prefix" \
        --query_file "$DATA_DIR/$ds/query.fbin" --gt_file "$DATA_DIR/$ds/gt.ibin" -K "$K" -T "$THREADS" \
        -L "${Ls[@]}" --result_path "$out/L$L" --num_nodes_to_cache "$CACHE_NODES"
    } > "$out/L$L.log" 2>&1
  done
done

echo "All Vamana experiments finished."
