#!/bin/bash
# Figure 13, DiskANN disk-index curves: SIFT1M with maximum degree 64 and 128.
# Usage:  bash experiments/topk_degree/run_vamana_degree.sh
# Needs:  experiments/equal_degree/build_search_disk_index.sh, build_indexes.sh vamana sift1M <degree>
# Output: $OUT_DIR/equal_degree/vamana/sift1M-d<degree>/k10/
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

EQDEG=$BENCH/experiments/equal_degree
DEGREES=${DEGREES:-64 128}

for D in $DEGREES; do
  index=$INDEX_DIR/sift1M/vamana/eqdeg-d$D/vamana_disk.index
  [[ -f $index ]] || bash "$EQDEG/build_indexes.sh" vamana sift1M "$D"
  DEGREE=$D K=10 WARMUP_RUNS=1 TIMED_RUNS=3 L_LIST="$(seq 10 2 500)" bash "$EQDEG/run_vamana.sh" sift1M
done
