#!/bin/bash
# Figure 13, GGNN curves: SIFT1M with maximum degree 64 and 128.
# Usage:  bash experiments/topk_degree/run_ggnn_degree.sh
# Needs:  experiments/equal_degree/build_ggnn.sh, build_indexes.sh ggnn sift1M <degree>
# Output: $OUT_DIR/equal_degree/ggnn/sift1M-d<degree>/k10/
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

EQDEG=$BENCH/experiments/equal_degree
DEGREES=${DEGREES:-64 128}

for D in $DEGREES; do
  graph=$INDEX_DIR/sift1M/ggnn/eqdeg-d$D
  if [[ ! -d $graph ]] || [[ -z $(find "$graph" -mindepth 1 -print -quit) ]]; then
    bash "$EQDEG/build_indexes.sh" ggnn sift1M "$D"
  fi
  DEGREE=$D K=10 WARMUP_RUNS=1 TIMED_RUNS=3 IT_LIST="$(seq 10 2 500)" bash "$EQDEG/run_ggnn.sh" sift1M
done
