#!/bin/bash
# Build the GGNN of the original manuscript ($W/src/ggnn-orig with ggnn_orig.patch): the three GGNN
# libraries, whose query() returns the counters and stage clocks, and the patched example driver,
# which builds DEEP100M in shards of 20,000,000 vectors.
#
# The patched CMakeLists.txt compiles the libraries for sm_80 only. The example is compiled with
# nvcc against those libraries with the flags of the CMake Release build, because its CMake target
# links gflags_nothreads_static, which the conda gflags package does not provide (the driver does
# not use gflags). The glog definitions are those of the conda glog::glog target.
#
# Usage:  bash third_party/build/ggnn_orig.sh
# Output: $W/build/ggnn-orig/libGGNN{Base,Construction,Query}.so,
#         $W/build/ggnn-orig/ggnn_main_gpu_data_sharded
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

JOBS=${JOBS:-16}
E=$ENV_BUILD
SRC=$W/src/ggnn-orig
B=$W/build/ggnn-orig
GENCODE="--generate-code=arch=compute_80,code=[compute_80,sm_80]"

grep -q 'd_path_count' "$SRC/include/ggnn/base/ggnn.cuh" || { echo "$SRC lacks ggnn_orig.patch" >&2; exit 1; }
export PATH=$E/bin:$PATH CUDACXX=$E/bin/nvcc
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
# Python_EXECUTABLE points to a missing file so that CMake skips the nanobind module.
cmake -S "$SRC" -B "$B" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$E" -DCUDAToolkit_ROOT="$E" \
      -DPython_EXECUTABLE=/nonexistent
cmake --build "$B" -j "$JOBS" --target GGNNBase GGNNConstruction GGNNQuery
"$E/bin/nvcc" -O3 -DNDEBUG -std=c++20 --expt-relaxed-constexpr $GENCODE -ccbin "$CXX" \
  -DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS -I"$SRC/include" -I"$E/include" \
  -x cu "$SRC/examples/cpp-and-cuda/ggnn_main_gpu_data.cu" -o "$B/ggnn_main_gpu_data_sharded" \
  -L"$B" -L"$E/lib" -lGGNNBase -lGGNNConstruction -lGGNNQuery -lglog -lgflags -lcurand \
  -Xlinker -rpath -Xlinker "$B" -Xlinker -rpath -Xlinker "$E/lib"
ls -la "$B"/libGGNN*.so "$B/ggnn_main_gpu_data_sharded"
