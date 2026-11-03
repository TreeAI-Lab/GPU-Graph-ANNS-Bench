#!/bin/bash
# Build the GGNN libraries ($W/src/ggnn) for sm_70 and sm_80, without the Python bindings.
#
# Usage:  bash third_party/build/ggnn.sh
# Output: $W/build/ggnn/libGGNNBase.so, libGGNNConstruction.so, libGGNNQuery.so
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

CUDA_ARCHS=${CUDA_ARCHS:-"70-real;80-real"}
JOBS=${JOBS:-16}

E=$ENV_BUILD
export PATH=$E/bin:$PATH CUDACXX=$E/bin/nvcc
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
# Python_EXECUTABLE points to a missing file so that CMake skips the nanobind module.
cmake -S "$W/src/ggnn" -B "$W/build/ggnn" -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_PREFIX_PATH="$E" -DCUDAToolkit_ROOT="$E" \
      -DGGNN_CUDA_ARCHITECTURES="$CUDA_ARCHS" -DPython_EXECUTABLE=/nonexistent
# Only the three libraries; the bundled example programs are not used and fail to link on some
# systems because of how cuRAND is linked.
cmake --build "$W/build/ggnn" -j "$JOBS" --target GGNNBase GGNNConstruction GGNNQuery
ls -la "$W"/build/ggnn/*.so
