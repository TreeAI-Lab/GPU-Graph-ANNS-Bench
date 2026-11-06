#!/bin/bash
# HNSW search of the equal-degree runs (HNSW curves of Figure 7).
# Usage:  bash run_hnsw.sh [dataset ...]    (default: all eight datasets)
# Needs:  build_hnsw.sh
# Output: $OUT_DIR/equal_degree/hnsw/<run>/k<K>/t<threads>.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

K=${K:-10}
THREAD_LIST=${THREAD_LIST:-"1 10 20 40"}
if [[ -z ${EF_FILE:-} ]]; then
  EF_FILE=$EQDEG_OUT/hnsw/ef_10_500_s2.txt
  mkdir -p "$(dirname "$EF_FILE")"
  seq 10 2 500 > "$EF_FILE"
fi
BIN=${BIN:-$BIN_DIR/faiss_hnsw}
export SEARCH_REPEATS=${TIMED_RUNS:-3}
export LD_LIBRARY_PATH=$W/build/faiss-hnsw/install/lib:$ENV_FAISS/lib:${LD_LIBRARY_PATH:-}
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M)

metric_of() {
  case $1 in glove200|nytimes) echo cosine ;; *) echo l2 ;; esac
}

for ds in "${DATASETS[@]}"; do
  deg=${DEGREE:-$(degree_of "$ds")}
  run=$ds; [ -z "${DEGREE:-}" ] || run=$ds-d$deg
  index=$INDEX_DIR/$ds/hnsw/eqdeg-d$deg/hnsw.faissindex
  out=$EQDEG_OUT/hnsw/$run/k$K
  write_meta "$out" hnsw "$ds" "$K"
  cp "$EF_FILE" "$out/ef_schedule.txt"
  for t in $THREAD_LIST; do
    clear_caches
    if [[ $(metric_of "$ds") == cosine ]]; then
      HNSW_METRIC=cosine "$BIN" search "$DATA_DIR/$ds" "$index" "$K" "$K" \
        --threads "$t" --ef-file "$out/ef_schedule.txt" > "$out/t${t}.log" 2>&1
    else
      "$BIN" search "$DATA_DIR/$ds" "$index" "$K" "$K" \
        --threads "$t" --ef-file "$out/ef_schedule.txt" > "$out/t${t}.log" 2>&1
    fi
  done
done

echo "All HNSW experiments finished."
