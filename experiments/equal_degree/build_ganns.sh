#!/bin/bash
# Build the GANNS program of the equal-degree runs (ganns-instrumented with ganns_instrumented.patch) for one dimension and
# distance. As in third_party/build/ganns.sh, the dimension and the distance are compile-time constants set in
# CMakeLists.txt, so each pair gets its own copy of the source.
#
# The program builds the graph (-b). With -s it prints the average degree and the connected components of the
# graph, then searches it and prints the per-stage clocks, cal_count and path_length. The -s mode and the 64-bit
# vector offsets needed for MNIST8M come from ganns_instrumented.patch.
#
# Usage:  bash build_ganns.sh <dim> <l2|cos>
# Output: $BIN_DIR/equal_degree/ganns/GANNS_<dim>_<metric>
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

DIM=${1:?usage: build_ganns.sh <dim> <l2|cos>}
METRIC=${2:?usage: build_ganns.sh <dim> <l2|cos>}
SRC=${GANNS_SRC:-$W/src/ganns-instrumented}
JOBS=${JOBS:-16}
case $METRIC in l2) dist=L2 ;; cos) dist=CS ;; *) echo "metric must be l2 or cos" >&2; exit 1 ;; esac

E=$ENV_BUILD
export PATH=$E/bin:$PATH CUDACXX=$E/bin/nvcc
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
WORK=$W/build/equal_degree/ganns/${DIM}_$METRIC
S=$WORK/src
rm -rf "$WORK" && mkdir -p "$S"
cp -r "$SRC/." "$S/" && rm -rf "$S/.git"
grep -q "SearchK_Stats" "$S/src/kernels.h" || { echo "$SRC lacks ganns_instrumented.patch" >&2; exit 1; }
sed -i -e "s/PLACE_HOLDER_DIM=[0-9]*/PLACE_HOLDER_DIM=${DIM}/" \
       -e "s/PRIVATE USE_DIST_[A-Z0-9]*_)/PRIVATE USE_DIST_${dist}_)/" "$S/CMakeLists.txt"
grep -n -E 'PLACE_HOLDER_DIM|USE_DIST' "$S/CMakeLists.txt"
cmake -S "$S" -B "$WORK/b" -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT="$E" > /dev/null
cmake --build "$WORK/b" -j "$JOBS" > /dev/null

mkdir -p "$EQDEG_BIN/ganns"
cp "$WORK/b/GANNS" "$EQDEG_BIN/ganns/GANNS_${DIM}_$METRIC"
ls -l "$EQDEG_BIN/ganns/GANNS_${DIM}_$METRIC"
