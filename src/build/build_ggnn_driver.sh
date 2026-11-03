#!/bin/bash
# Compile the GGNN build driver (ggnn_build.cpp) into $BIN_DIR/ggnn_build. It links the GGNN libraries that
# third_party/build/ggnn.sh builds in $W/build/ggnn, with the headers of $W/src/ggnn.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)

# GPU architectures: sm_70 (V100) and sm_80 (A100, A800)
GENCODE=${GENCODE:-"-arch=sm_70 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80"}
E=$ENV_BUILD

"$E/bin/nvcc" -O3 -std=c++20 $GENCODE -ccbin "$E/bin/x86_64-conda-linux-gnu-g++" \
  -x cu "$HERE/ggnn_build.cpp" -o "$BIN_DIR/ggnn_build" \
  -I"$W/src/ggnn/include" -I"$E/include" -L"$W/build/ggnn" -L"$E/lib" \
  -lGGNNBase -lGGNNConstruction -lGGNNQuery -lglog -lgflags -lcurand \
  -Xlinker -rpath -Xlinker "$W/build/ggnn" -Xlinker -rpath -Xlinker "$E/lib"
echo "built $BIN_DIR/ggnn_build"
