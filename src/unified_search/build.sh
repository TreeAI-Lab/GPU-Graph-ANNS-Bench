#!/bin/bash
# Build the common searcher into $BIN_DIR: unified_search (CPU, OpenMP) and unified_search_gpu (CUDA).
# Usage: bash src/unified_search/build.sh [cpu] [gpu]      (both by default)
# -O3 -march=native change the speed only, not the counts.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)

# GPU architectures: sm_70 (V100) and sm_80 (A100, A800)
GENCODE=${GENCODE:-"-gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80"}
CXX=$ENV_BUILD/bin/x86_64-conda-linux-gnu-g++

[ $# -gt 0 ] || set -- cpu gpu
for target in "$@"; do
  case $target in
    cpu)
      "$CXX" -O3 -march=native -fopenmp -std=c++17 -o "$BIN_DIR/unified_search" "$HERE/unified_search.cpp"
      echo "built $BIN_DIR/unified_search" ;;
    gpu)
      "$ENV_BUILD/bin/nvcc" -O3 -std=c++17 $GENCODE -ccbin "$CXX" \
        -o "$BIN_DIR/unified_search_gpu" "$HERE/unified_search_gpu.cu" \
        -Xlinker -rpath -Xlinker "$ENV_BUILD/lib"
      echo "built $BIN_DIR/unified_search_gpu" ;;
    *)
      echo "unknown target $target (cpu or gpu)"; exit 1 ;;
  esac
done
