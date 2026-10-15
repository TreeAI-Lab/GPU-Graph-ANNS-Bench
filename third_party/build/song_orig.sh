#!/bin/bash
# Build the SONG program of the original manuscript ($W/src/song-orig with song_orig.patch) for one
# graph layout, dimension, queue capacity and distance.
#
# The patched kernelgraph.h takes the graph layout from config.h: FIXED_DEGREE slots per vertex
# (forward degree FIXED_DEGREE/2) and an adjacency stride of 2^FIXED_DEGREE_SHIFT entries, which
# must hold FIXED_DEGREE ids plus the count. The program is generated as in the SONG README:
# generate_template.sh, then fill_parameters.sh <pq_size> <dim> <metric>, which compiles
# build_template/tmp and copies the program to the top of the copy. The search limit
# (max_iterations) is a run-time argument of the patched program.
#
# Usage:  bash third_party/build/song_orig.sh <FD> <SHIFT> <DIM> [TOPK] [METRIC]
#         TOPK is the pq_size of fill_parameters.sh (default 64), METRIC l2, cos or ip (default l2)
# Output: $W/build/song-orig/fd<FD>_s<SHIFT>_dim<DIM>/song, with _k<TOPK> and _<METRIC> appended
#         to the directory name when they differ from the defaults
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

FD=${1:?usage: song_orig.sh <FD> <SHIFT> <DIM> [TOPK] [METRIC]}
SHIFT=${2:?usage: song_orig.sh <FD> <SHIFT> <DIM> [TOPK] [METRIC]}
DIM=${3:?usage: song_orig.sh <FD> <SHIFT> <DIM> [TOPK] [METRIC]}
TOPK=${4:-64}
METRIC=${5:-l2}

[ $(( FD + 1 )) -le $(( 1 << SHIFT )) ] || { echo "FD=$FD needs FD+1 <= 2^SHIFT" >&2; exit 1; }
case $METRIC in l2|cos|ip) ;; *) echo "METRIC must be l2, cos or ip" >&2; exit 1 ;; esac

NAME=fd${FD}_s${SHIFT}_dim${DIM}
[ "$TOPK" = 64 ] || NAME=${NAME}_k$TOPK
[ "$METRIC" = l2 ] || NAME=${NAME}_$METRIC
OUT=$W/build/song-orig/$NAME

export PATH=$ENV_BUILD/bin:$PATH
rm -rf "$OUT" && mkdir -p "$OUT" && cp -r "$W/src/song-orig/." "$OUT/" && rm -rf "$OUT/.git"
cd "$OUT"
grep -q 'max_iterations' main.cu || { echo "$W/src/song-orig lacks song_orig.patch" >&2; exit 1; }
sed -i -e "s/^#define FIXED_DEGREE .*/#define FIXED_DEGREE $FD/" \
       -e "s/^#define FIXED_DEGREE_SHIFT .*/#define FIXED_DEGREE_SHIFT $SHIFT/" config.h
grep -n '^#define' config.h
bash generate_template.sh > generate_template.log 2>&1
bash fill_parameters.sh "$TOPK" "$DIM" "$METRIC" > fill_parameters.log 2>&1 && [ -x "$OUT/song" ] \
  || { echo "build failed, see $OUT/fill_parameters.log" >&2; exit 1; }
ls -la "$OUT/song"
