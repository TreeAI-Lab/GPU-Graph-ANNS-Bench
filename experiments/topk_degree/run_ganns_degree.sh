#!/bin/bash
# Figure 13, GANNS curves: SIFT1M with maximum degree 64 and 128.
# Usage:  bash experiments/topk_degree/run_ganns_degree.sh
# Needs:  experiments/equal_degree/build_ganns.sh 128 l2 (128 = SIFT1M dimension), build_indexes.sh ganns sift1M <degree>
# Output: $OUT_DIR/equal_degree/ganns/sift1M-d<degree>/k10/
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

EQDEG=$BENCH/experiments/equal_degree
DEGREES=${DEGREES:-64 128}

for D in $DEGREES; do
  graph=$INDEX_DIR/sift1M/ganns/eqdeg-d$D/graph.nsw
  [[ -f $graph ]] || bash "$EQDEG/build_indexes.sh" ganns sift1M "$D"
  DEGREE=$D K=10 WARMUP_RUNS=1 TIMED_RUNS=3 E_LIST="$(seq 10 2 500)" bash "$EQDEG/run_ganns.sh" sift1M
done
