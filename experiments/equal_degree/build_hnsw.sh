#!/bin/bash
# Usage:  bash build_hnsw.sh [dataset ...]    (default: all eight datasets)
# Needs:  $BIN_DIR/faiss_hnsw (bash src/baselines/build.sh faiss)
# Builds each index BUILD_RUNS times (default 5), M = degree / 2 and efConstruction = the construction L of the degree
# (60, 120, 150, 180 for D = 32, 64, 96, 128).
# Output: $INDEX_DIR/<dataset>/hnsw/eqdeg-d<degree>/hnsw.faissindex,
#         $OUT_DIR/equal_degree/build/hnsw/<dataset>-d<degree>.b<i>.log for build i
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

BUILD_THREADS=${BUILD_THREADS:-10}
BUILD_RUNS=${BUILD_RUNS:-5}
BIN=${BIN:-$BIN_DIR/faiss_hnsw}
export LD_LIBRARY_PATH=$W/build/faiss-hnsw/install/lib:$ENV_FAISS/lib:${LD_LIBRARY_PATH:-}
DATASETS=("$@")
[ ${#DATASETS[@]} -gt 0 ] || DATASETS=(sift1M gist glove200 nytimes deep1M deep10M deep100M mnist8M)

metric_of() {
  case $1 in glove200|nytimes) echo cosine ;; *) echo l2 ;; esac
}

for ds in "${DATASETS[@]}"; do
  deg=${DEGREE:-$(degree_of "$ds")}
  m=$((deg / 2))
  efc=${EFC:-$(build_l_of "$deg")}
  index=$INDEX_DIR/$ds/hnsw/eqdeg-d$deg/hnsw.faissindex
  mkdir -p "$(dirname "$index")" "$EQDEG_OUT/build/hnsw"
  read -r -a TIME <<< "$(time_cmd)"
  for ((run = 1; run <= BUILD_RUNS; run++)); do
    log=$EQDEG_OUT/build/hnsw/$ds-d$deg.b$run.log
    if [[ $(metric_of "$ds") == cosine ]]; then
      HNSW_METRIC=cosine "${TIME[@]}" "$BIN" build "$DATA_DIR/$ds" "$m" "$efc" "$BUILD_THREADS" "$index" > "$log" 2>&1
    else
      "${TIME[@]}" "$BIN" build "$DATA_DIR/$ds" "$m" "$efc" "$BUILD_THREADS" "$index" > "$log" 2>&1
    fi
  done
done
