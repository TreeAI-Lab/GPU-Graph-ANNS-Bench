#!/bin/bash
# Figure 12, BANG curves: SIFT1M with maximum degree 64 and 128.
# Usage:  bash experiments/topk_degree/run_bang_degree.sh
# Needs:  third_party/build/bang.sh <degree>, build_indexes.sh vamana sift1M <degree>
# Output: $OUT_DIR/equal_degree/bang/sift1M-d<degree>/k10/L<L>.log
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

EQDEG=$BENCH/experiments/equal_degree
DEGREES=${DEGREES:-64 128}

for D in $DEGREES; do
  bang=$W/build/bang/r$D/bang_search
  index=$INDEX_DIR/sift1M/vamana/eqdeg-d$D/vamana_disk.index
  [[ -x $bang ]] || bash "$BENCH/third_party/build/bang.sh" "$D"
  [[ -f $index ]] || bash "$EQDEG/build_indexes.sh" vamana sift1M "$D"
  DEGREE=$D K=10 L_LIST="$(seq 10 2 500)" bash "$EQDEG/run_bang.sh" sift1M
done
