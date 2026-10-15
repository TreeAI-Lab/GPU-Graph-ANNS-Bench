#!/bin/bash
# Build the DiskANN tools of the original manuscript ($W/src/diskann-orig with diskann_orig.patch):
# search_disk_index with per-stage timers and counters, and build_disk_index. Same toolchain and
# MKL handling as diskann.sh.
#
# Usage:  bash third_party/build/diskann_orig.sh
# Output: $W/build/diskann-orig/apps/search_disk_index, $W/build/diskann-orig/apps/build_disk_index
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

JOBS=${JOBS:-32}
E=$ENV_BUILD
SRC=$W/src/diskann-orig
BUILD=$W/build/diskann-orig

grep -q 'path_length' "$SRC/include/percentile_stats.h" || { echo "$SRC lacks diskann_orig.patch" >&2; exit 1; }
export PATH=$E/bin:$PATH CC=$E/bin/x86_64-conda-linux-gnu-gcc CXX=$E/bin/x86_64-conda-linux-gnu-g++
# DiskANN's CMakeLists only looks for libmkl_def.so.2 or libmkl_def.so; conda-forge MKL 2025 and
# later ships libmkl_def.so.3, so add an unversioned link next to it.
[ -e "$E/lib/libmkl_def.so" ] || ln -s libmkl_def.so.3 "$E/lib/libmkl_def.so"
# libmkl_intel_thread of MKL 2025 and later needs libatomic.
cmake -S "$SRC" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$E" \
      -DOMP_PATH="$E/lib" -DMKL_PATH="$E/lib" -DMKL_INCLUDE_PATH="$E/include" \
      -DCMAKE_EXE_LINKER_FLAGS="-latomic"
cmake --build "$BUILD" -j "$JOBS" --target search_disk_index build_disk_index
ls -la "$BUILD/apps/search_disk_index" "$BUILD/apps/build_disk_index"
