#!/bin/bash
# Figure 12, HNSW curves (one thread): SIFT1M with maximum degree 64 and 128.
# Usage:  bash experiments/topk_degree/run_hnsw_degree.sh
# Needs:  experiments/equal_degree/build_hnsw.sh, $BIN_DIR/faiss_hnsw
# Output: $OUT_DIR/equal_degree/hnsw/sift1M-d<degree>/k10/t1.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

EQDEG=$BENCH/experiments/equal_degree
DEGREES=${DEGREES:-64 128}
EF=$OUT_DIR/topk_degree/ef_10_300_s2.txt
mkdir -p "$(dirname "$EF")"
seq 10 2 300 > "$EF"

for D in $DEGREES; do
  index=$INDEX_DIR/sift1M/hnsw/eqdeg-d$D/hnsw.faissindex
  [[ -f $index ]] || BUILD_RUNS=1 DEGREE=$D bash "$EQDEG/build_hnsw.sh" sift1M
  THREAD_LIST=1 DEGREE=$D K=10 TIMED_RUNS=3 EF_FILE=$EF bash "$EQDEG/run_hnsw.sh" sift1M
done
