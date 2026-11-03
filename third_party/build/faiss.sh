#!/bin/bash
# Build Faiss with GPU support from $W/src/<name> in the faiss-build environment and install it
# under $W/build/<name>/install: shared libfaiss and libfaiss_avx2, OpenBLAS, no Python bindings,
# no C API, no tests. <name> is faiss (the IVF-PQ program links against it) or faiss-hnsw (v1.15.1,
# the HNSW program), see third_party/versions.tsv.
#
# Usage:  bash third_party/build/faiss.sh [faiss|faiss-hnsw]    (default: faiss)
# Output: $W/build/<name>/install/lib/libfaiss.so, libfaiss_avx2.so, $W/build/<name>/install/include
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

CUDA_ARCH=${CUDA_ARCH:-80}   # CMAKE_CUDA_ARCHITECTURES
JOBS=${JOBS:-16}
NAME=${1:-faiss}
BUILD_DIR=$W/build/$NAME/build
PREFIX=$W/build/$NAME/install

E=$ENV_FAISS
export PATH=$E/bin:$PATH
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
# The GPU code includes curand_kernel.h (impl/IcmEncoder.cu) and cuda_profiler_api.h
# (utils/DeviceUtils.cu). Use the headers of the same CUDA version as nvcc; mixing in the
# headers of another toolkit gives type errors.
for h in curand_kernel.h cuda_profiler_api.h; do
  [ -f "$E/targets/x86_64-linux/include/$h" ] || [ -f "$E/include/$h" ] || {
    echo "$h not found in $E; install libcurand-dev and cuda-profiler-api of the CUDA version of nvcc" >&2; exit 1; }
done

cmake -S "$W/src/$NAME" -B "$BUILD_DIR" -G Ninja \
      -DCMAKE_BUILD_TYPE=Release \
      -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=OFF -DFAISS_ENABLE_C_API=OFF \
      -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DFAISS_OPT_LEVEL=avx2 -DBLA_VENDOR=OpenBLAS \
      -DCMAKE_PREFIX_PATH="$E" -DCUDAToolkit_ROOT="$E" -DCMAKE_CUDA_COMPILER="$E/bin/nvcc" \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib
cmake --build "$BUILD_DIR" -j "$JOBS" --target faiss faiss_avx2
cmake --install "$BUILD_DIR"
ls -la "$PREFIX/lib/libfaiss.so" "$PREFIX/lib/libfaiss_avx2.so"
