#!/bin/bash
# Build ggnn_main_gpu_data (src/ggnn, the build and query driver of the equal-degree runs) against the GGNN
# libraries that third_party/build/ggnn_instrumented.sh builds from ggnn-instrumented with ggnn_instrumented.patch
# (query() returns the counters and stage clocks; sm_80). The driver is compiled with nvcc and the flags
# of a CMake Release build, the same way ggnn_instrumented.sh compiles the sharded example driver.
#
# Usage:  bash build_ggnn.sh
# Output: $BIN_DIR/equal_degree/ggnn_main_gpu_data
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

SRC=$W/src/ggnn-instrumented
LIBS=$W/build/ggnn-instrumented
GENCODE="--generate-code=arch=compute_80,code=[compute_80,sm_80]"

[ -f "$LIBS/libGGNNQuery.so" ] || bash "$BENCH/third_party/build/ggnn_instrumented.sh"

E=$ENV_BUILD
CXX=$E/bin/x86_64-conda-linux-gnu-g++
mkdir -p "$EQDEG_BIN"
"$E/bin/nvcc" -O3 -DNDEBUG -std=c++20 --expt-relaxed-constexpr $GENCODE -ccbin "$CXX" \
  -DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS -I"$SRC/include" -I"$E/include" \
  -x cu "$EQDEG_HERE/src/ggnn/ggnn_main_gpu_data.cu" -o "$EQDEG_BIN/ggnn_main_gpu_data" \
  -L"$LIBS" -L"$E/lib" -lGGNNBase -lGGNNConstruction -lGGNNQuery -lglog -lgflags -lcurand \
  -Xlinker -rpath -Xlinker "$LIBS" -Xlinker -rpath -Xlinker "$E/lib"
ls -l "$EQDEG_BIN/ggnn_main_gpu_data"
