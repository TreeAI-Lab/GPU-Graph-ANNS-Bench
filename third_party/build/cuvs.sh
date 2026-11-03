#!/bin/bash
# Build libcuvs from $W/src/cuvs (cuVS branch-25.02 with cuvs_no_reverse.patch) in the cuvs-build
# environment and install it under $W/build/cuvs/install. The configuration matches
# `./build.sh libcuvs --no-mg` (shared library and C API, no multi-GPU algorithms) without tests
# and benchmarks; cmake is called directly because build.sh needs literal quotes in --cmake-args.
# CMake (rapids-cmake/CPM) downloads RAFT, RMM, CCCL, cutlass, hnswlib and the other dependencies
# into $W/build/cuvs/build/_deps. The programs in src/ that link libcuvs take their headers from
# there and from the install prefix.
#
# If the dependencies are supplied from a local copy instead of being downloaded, hnswlib must be
# v0.7.0 with cpp/cmake/patches/hnswlib.diff applied; cuVS 25.02 does not compile against 0.8.
#
# Usage:  bash third_party/build/cuvs.sh
# Output: $W/build/cuvs/install/lib/libcuvs.so and libcuvs_c.so, $W/build/cuvs/install/include,
#         dependency sources in $W/build/cuvs/build/_deps
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

CUDA_ARCH=${CUDA_ARCH:-80}   # CMAKE_CUDA_ARCHITECTURES; NATIVE builds for the GPUs in this machine
JOBS=${JOBS:-8}
BUILD_DIR=$W/build/cuvs/build
PREFIX=$W/build/cuvs/install

E=$ENV_CUVS
export PATH=$E/bin:$PATH CONDA_PREFIX=$E CUDAToolkit_ROOT=$E CUDACXX=$E/bin/nvcc
export CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CXX
# conda keeps the CUDA library headers (cuFile among them) under targets/x86_64-linux/include.
# cuVS compiles with -Werror, and gcc 12 reports a false maybe-uninitialized warning in
# src/neighbors/brute_force_serialize.cu; -Wno-error keeps that one a warning (the code is unchanged).
NO_ERR=-Wno-error=maybe-uninitialized
export CXXFLAGS="-isystem $E/targets/x86_64-linux/include $NO_ERR ${CXXFLAGS:-}"

cmake -S "$W/src/cuvs/cpp" -B "$BUILD_DIR" -G Ninja \
      -DCMAKE_CXX_FLAGS="$CXXFLAGS" -DCMAKE_CUDA_FLAGS="-Xcompiler=$NO_ERR" \
      -DCMAKE_INSTALL_PREFIX="$PREFIX" -DCMAKE_INSTALL_LIBDIR=lib -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCH" \
      -DBUILD_C_LIBRARY=ON -DBUILD_MG_ALGOS=OFF \
      -DBUILD_TESTS=OFF -DBUILD_C_TESTS=OFF -DBUILD_CUVS_BENCH=OFF \
      -DCUDAToolkit_ROOT="$E" -DCMAKE_CUDA_COMPILER="$E/bin/nvcc"
cmake --build "$BUILD_DIR" -j "$JOBS" --target cuvs install
ls -la "$PREFIX/lib/libcuvs.so" "$PREFIX/lib/libcuvs_c.so"
