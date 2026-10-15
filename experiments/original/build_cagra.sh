#!/bin/bash
# Build my_cagra (src/cagra) against libcuvs of the cagra environment, with the CUDA 12.4 toolchain of the
# build environment.
#
# Usage:  bash build_cagra.sh
# Output: $BIN_DIR/original/my_cagra
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

JOBS=${JOBS:-8}
BUILD=$W/build/original/cagra

E=$ENV_BUILD
export PATH=$E/bin:$PATH CUDACXX=$E/bin/nvcc
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
# The cuVS headers include cuSOLVER and cuSPARSE headers, which come with the cagra environment (CUDA 12.4)
cmake -S "$ORIG_HERE/src/cagra" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ENV_CAGRA" \
      -DCUDAToolkit_ROOT="$E" \
      -DCMAKE_CUDA_FLAGS="-I$ENV_CAGRA/targets/x86_64-linux/include" \
      -DCMAKE_CXX_FLAGS="-I$ENV_CAGRA/targets/x86_64-linux/include"
cmake --build "$BUILD" -j "$JOBS" --target my_cagra
mkdir -p "$ORIG_BIN"
cp "$BUILD/my_cagra" "$ORIG_BIN/my_cagra"
ls -l "$ORIG_BIN/my_cagra"
