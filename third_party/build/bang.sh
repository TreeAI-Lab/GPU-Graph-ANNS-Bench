#!/bin/bash
# Build BANG ($W/src/bang/BANG_Base, patched) for graphs of degree R.
#
# bang_search.cu sizes its shared-memory buffers with the compile-time constant MAX_R, which must
# equal the degree R of the graph that is searched. Each degree gets its own copy of the source
# with MAX_R set to R (the patch sets 128 as the default).
#
# Usage:  bash third_party/build/bang.sh <R>
# Output: $W/build/bang/r<R>/bang_search and $W/build/bang/r<R>/libbang.so
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

R=${1:?usage: bang.sh <R>}
CUDA_ARCH=${CUDA_ARCH:-80}
JOBS=${JOBS:-$(nproc)}

E=$ENV_BUILD
export PATH=$E/bin:$PATH CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
OUT=$W/build/bang/r$R
rm -rf "$OUT" && mkdir -p "$OUT/src"
cp -r "$W/src/bang/BANG_Base/." "$OUT/src/"
sed -i -E "s/^#define MAX_R [0-9]+/#define MAX_R $R/" "$OUT/src/bang_search.cu"
grep -n '^#define MAX_R' "$OUT/src/bang_search.cu"
# CMakeLists.txt uses the FindCUDA module, which expects the headers and libraries directly under the toolkit
# root; cuda_root.sh gives the conda environment that layout. CUDA_ROOT may point to a system CUDA 12.4 instead.
CUDA_ROOT=${CUDA_ROOT:-$(bash "$(cd "$(dirname "$0")" && pwd)/cuda_root.sh" "$E")}
CUDA_ARGS=(-DCUDA_TOOLKIT_ROOT_DIR="$CUDA_ROOT")
cmake -S "$OUT/src" -B "$OUT" "${CUDA_ARGS[@]}" \
  -DCUDA_NVCC_FLAGS="-gencode=arch=compute_${CUDA_ARCH},code=sm_${CUDA_ARCH}" > "$OUT/build_cmake.log" 2>&1 \
  || { tail -20 "$OUT/build_cmake.log"; exit 1; }
cmake --build "$OUT" -j "$JOBS" > "$OUT/build_make.log" 2>&1 || { tail -30 "$OUT/build_make.log"; exit 1; }
ls -la "$OUT/bang_search" "$OUT/libbang.so"
