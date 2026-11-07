#!/bin/bash
# Build the SONG program of the equal-degree runs (song-instrumented with song_instrumented.patch) for one dataset.
#
# Dimension, distance, queue size (pq_size) and the graph degree are compile-time constants, so every
# dataset gets its own build, made by third_party/build/song_instrumented.sh. The degree is FIXED_DEGREE in
# config.h; the adjacency stride 2^FIXED_DEGREE_SHIFT must hold the degree plus one count slot.
# The same program builds the graph (song build) and searches it (song test); the search takes
# max_iterations and the graph directory on the command line and prints path_length and
# calculate_count (Figure 6(b)).
#
# Usage:  bash build_song.sh <dataset> [degree]
# Output: $BIN_DIR/equal_degree/song/<dataset>-d<degree>/song; $W/build/equal_degree/song/<dataset>-d<degree>
#         links to the build directory (its config.h gives the shift to build_indexes.sh)
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

DS=${1:?usage: build_song.sh <dataset> [degree]}
DEG=${2:-$(degree_of "$DS")}
PQ_SIZE=${PQ_SIZE:-60}   # pq_size of the SONG search queue

case $DS in
  glove200|nytimes) METRIC=cos ;;
  *) METRIC=l2 ;;
esac
DIM=$(dim_of "$DS")
SHIFT=0
while [ $(( 1 << SHIFT )) -lt $(( DEG + 1 )) ]; do SHIFT=$(( SHIFT + 1 )); done

bash "$BENCH/third_party/build/song_instrumented.sh" "$DEG" "$SHIFT" "$DIM" "$PQ_SIZE" "$METRIC"
NAME=fd${DEG}_s${SHIFT}_dim${DIM}
[ "$PQ_SIZE" = 64 ] || NAME=${NAME}_k$PQ_SIZE
[ "$METRIC" = l2 ] || NAME=${NAME}_$METRIC
BUILD=$W/build/song-instrumented/$NAME

LINK=$W/build/equal_degree/song/$DS-d$DEG
mkdir -p "$(dirname "$LINK")"
rm -rf "$LINK"
ln -s "$BUILD" "$LINK"
mkdir -p "$EQDEG_BIN/song/$DS-d$DEG"
cp "$BUILD/song" "$EQDEG_BIN/song/$DS-d$DEG/song"
echo "$EQDEG_BIN/song/$DS-d$DEG/song (dim $DIM, $METRIC, degree $DEG, shift $SHIFT, pq_size $PQ_SIZE)"
