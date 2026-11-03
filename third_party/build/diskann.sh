#!/bin/bash
# Build the DiskANN tools ($W/src/diskann, cpp_main branch) with MKL and Intel OpenMP from the
# build environment (mkl-devel). build_memory_index builds the Vamana graphs; build_disk_index
# builds the disk index from which the BANG input files are converted.
#
# Usage:  bash third_party/build/diskann.sh
# Output: $W/build/diskann/apps/build_memory_index, $W/build/diskann/apps/build_disk_index
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

JOBS=${JOBS:-32}
E=$ENV_BUILD

export PATH=$E/bin:$PATH CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
# DiskANN's CMakeLists only looks for libmkl_def.so.2 or libmkl_def.so; conda-forge MKL 2025 and
# later ships libmkl_def.so.3, so add an unversioned link next to it.
[ -e "$E/lib/libmkl_def.so" ] || ln -s libmkl_def.so.3 "$E/lib/libmkl_def.so"
# libmkl_intel_thread of MKL 2025 and later needs libatomic.
cmake -S "$W/src/diskann" -B "$W/build/diskann" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$E" \
      -DOMP_PATH="$E/lib" -DMKL_PATH="$E/lib" -DMKL_INCLUDE_PATH="$E/include" \
      -DCMAKE_EXE_LINKER_FLAGS="-latomic"
cmake --build "$W/build/diskann" -j "$JOBS" --target build_memory_index build_disk_index
ls -la "$W/build/diskann/apps/build_memory_index" "$W/build/diskann/apps/build_disk_index"
