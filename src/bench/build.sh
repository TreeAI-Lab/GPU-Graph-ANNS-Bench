#!/bin/bash
# Build the host-to-device copy benchmark into $BIN_DIR/h2d_copy_bench (CUDA 12.4 toolchain of $ENV_BUILD).
# Usage: bash src/bench/build.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

HERE=$(cd "$(dirname "$0")" && pwd)
# A800 / A100: sm_80
GENCODE=${GENCODE:--gencode=arch=compute_80,code=sm_80}

"$ENV_BUILD/bin/nvcc" -O3 -std=c++17 $GENCODE \
  -ccbin "$ENV_BUILD/bin/x86_64-conda-linux-gnu-g++" \
  "$HERE/h2d_copy_bench.cu" -o "$BIN_DIR/h2d_copy_bench"
echo "built $BIN_DIR/h2d_copy_bench"
