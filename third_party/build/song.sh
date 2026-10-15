#!/bin/bash
# Build SONG ($W/src/song) for one set of graph parameters and one data dimension, L2 distance.
#
# The GPU graph construction (song build, called by build_graph.sh) uses KernelFixedDegreeGraph in
# kernelgraph.h, where the parameters are constants: degree (line 27, default 15),
# flexible_degree = 2*degree+1 (line 28, storage limit), vertex_offset_shift (line 29, default 5,
# adjacency stride 2^shift) and the insertion search budget degree*10 (line 152).
# SEARCH_DEGREE, FIXED_DEGREE and CONSTRUCT_SEARCH_BUDGET in config.h only affect the CPU program
# song.cpu. This script edits the constants in a copy of the source (the fetched tree is not
# changed) and sets the config.h macros to the same values. It requires 2*SD+1 <= 2^SHIFT - 1.
#
# Usage:  bash third_party/build/song.sh <SD> <SHIFT> <BUDGET> <DIM>
# Output: $W/build/song/d<SD>_s<SHIFT>_b<BUDGET>_dim<DIM>/build_template/song
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

SD=${1:-15} SHIFT=${2:-5} BUDGET=${3:-150} DIM=${4:-128}
PQ_SIZE=${PQ_SIZE:-100}   # <pq_size> argument of fill_parameters.sh
JOBS=${JOBS:-8}

export PATH=$ENV_BUILD/bin:$PATH
OUT=$W/build/song/d${SD}_s${SHIFT}_b${BUDGET}_dim${DIM}
FD=$(( (1 << SHIFT) - 1 ))
[ $(( 2 * SD + 1 )) -le $FD ] || { echo "SD=$SD needs 2*SD+1 <= 2^SHIFT-1 = $FD" >&2; exit 1; }

rm -rf "$OUT" && mkdir -p "$OUT" && cp -r "$W/src/song/." "$OUT/" && cd "$OUT"
sed -i -e "s/const int degree = 15;/const int degree = $SD;/" \
       -e "s/const int vertex_offset_shift = 5;/const int vertex_offset_shift = $SHIFT;/" \
       -e "s/search_top_k(point,degree\*10,neighbor);/search_top_k(point,$BUDGET,neighbor);/" kernelgraph.h
grep -c -E "const int degree = $SD;|const int vertex_offset_shift = $SHIFT;|search_top_k\(point,$BUDGET,neighbor\);" kernelgraph.h \
  | grep -q '^3$' || { echo "kernelgraph.h substitution failed" >&2; exit 1; }
sed -i -e "s/^#define SEARCH_DEGREE .*/#define SEARCH_DEGREE $SD/" \
       -e "s/^#define FIXED_DEGREE .*/#define FIXED_DEGREE $FD/" \
       -e "s/^#define FIXED_DEGREE_SHIFT .*/#define FIXED_DEGREE_SHIFT $SHIFT/" \
       -e "s/^#define CONSTRUCT_SEARCH_BUDGET .*/#define CONSTRUCT_SEARCH_BUDGET $BUDGET/" config.h
# Compile for sm_70 and sm_80 only; CUDA 12.4 warns that the older architectures are deprecated.
sed -i -e 's/-arch=sm_50 \\/-arch=sm_70 \\/' -e '/compute_5[02]\|compute_6[01]\|compute_75/d' Makefile
sed -i -e 's/-gencode=arch=compute_70,code=sm_70 \\/-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 \\/' Makefile

bash generate_template.sh > gen.log 2>&1
cd build_template && bash fill_parameters.sh "$PQ_SIZE" "$DIM" l2 > fill.log 2>&1 && make -j "$JOBS" song > make.log 2>&1
ls -la "$OUT/build_template/song"
grep -n -E 'const int degree|vertex_offset_shift = |search_top_k\(point,' "$OUT/build_template/kernelgraph.h"
