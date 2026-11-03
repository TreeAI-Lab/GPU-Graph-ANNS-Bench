#!/bin/bash
# Build cagra_build_search (src/cagra) against libcuvs of the cagra environment, with the CUDA 12.4 toolchain of the
# build environment.
#
# Usage:  bash build_cagra.sh
# Output: $BIN_DIR/equal_degree/cagra_build_search
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

JOBS=${JOBS:-8}
BUILD=$W/build/equal_degree/cagra

E=$ENV_BUILD
export PATH=$E/bin:$PATH CUDACXX=$E/bin/nvcc
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
# The cuVS headers include cuSOLVER and cuSPARSE headers, which come with the cagra environment (CUDA 12.4)
cmake -S "$EQDEG_HERE/src/cagra" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ENV_CAGRA" \
      -DCUDAToolkit_ROOT="$E" \
      -DCMAKE_CUDA_FLAGS="-I$ENV_CAGRA/targets/x86_64-linux/include" \
      -DCMAKE_CXX_FLAGS="-I$ENV_CAGRA/targets/x86_64-linux/include"
cmake --build "$BUILD" -j "$JOBS" --target cagra_build_search
mkdir -p "$EQDEG_BIN"
cp "$BUILD/cagra_build_search" "$EQDEG_BIN/cagra_build_search"
ls -l "$EQDEG_BIN/cagra_build_search"
