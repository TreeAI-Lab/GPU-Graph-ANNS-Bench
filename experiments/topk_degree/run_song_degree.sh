#!/bin/bash
# Figure 12, SONG curves: SIFT1M with maximum degree 64 and 128.
# Usage:  bash experiments/topk_degree/run_song_degree.sh
# Needs:  experiments/equal_degree/build_song.sh sift1M <degree>, build_indexes.sh song sift1M <degree>
# Output: $OUT_DIR/equal_degree/song/sift1M-d<degree>/k10/
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

EQDEG=$BENCH/experiments/equal_degree
DEGREES=${DEGREES:-64 128}

for D in $DEGREES; do
  bin=$BIN_DIR/equal_degree/song/sift1M-d$D/song
  graph=$INDEX_DIR/sift1M/song/eqdeg-d$D/bfsg.graph
  [[ -x $bin ]] || bash "$EQDEG/build_song.sh" sift1M "$D"
  [[ -f $graph ]] || bash "$EQDEG/build_indexes.sh" song sift1M "$D"
  DEGREE=$D K=10 WARMUP_RUNS=1 TIMED_RUNS=3 IT_LIST="$(seq 10 2 500)" bash "$EQDEG/run_song.sh" sift1M
done
