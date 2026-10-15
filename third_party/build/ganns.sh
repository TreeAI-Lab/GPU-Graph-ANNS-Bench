#!/bin/bash
# Build GANNS ($W/src/ganns) for one data dimension, L2 distance.
#
# The dimension and the distance are compile-time constants: DIM in src/device_func.h:8, set
# through lines 23-24 of CMakeLists.txt (README, section "Usage"). Following the README, each
# dimension gets its own copy of the source in which only these two lines are changed.
#
# Usage:  bash third_party/build/ganns.sh <DIM>
# Output: $W/build/ganns/dim<DIM>/GANNS
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

DIM=${1:?usage: ganns.sh <dim>}
JOBS=${JOBS:-16}

E=$ENV_BUILD
export PATH=$E/bin:$PATH CUDACXX=$E/bin/nvcc
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
OUT=$W/build/ganns/dim${DIM}
S=$OUT/src
rm -rf "$OUT" && mkdir -p "$S" && cp -r "$W/src/ganns/." "$S/"
sed -i -e "s/PLACE_HOLDER_DIM=[0-9]*/PLACE_HOLDER_DIM=${DIM}/" \
       -e "s/PRIVATE USE_DIST_[A-Z0-9]*_)/PRIVATE USE_DIST_L2_)/" "$S/CMakeLists.txt"
grep -n -E 'PLACE_HOLDER_DIM|USE_DIST' "$S/CMakeLists.txt"
cmake -S "$S" -B "$OUT/b" -DCMAKE_BUILD_TYPE=Release -DCUDAToolkit_ROOT="$E" > /dev/null
cmake --build "$OUT/b" -j "$JOBS" > /dev/null
cp "$OUT/b/GANNS" "$OUT/GANNS" && ls -la "$OUT/GANNS"
