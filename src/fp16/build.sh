#!/bin/bash
# Build the FP32 and FP16 search programs used for Tables 9 and 10 and Figure 14 into $BIN_DIR.
# The FP32 and FP16 programs of a system come from the same sources and differ only in one compile switch.
# SONG, GANNS and GGNN are built from fresh copies of the pinned sources in $W/src with our patches applied.
#
# usage: bash src/fp16/build.sh [song] [ganns] [ggnn] [cagra]     (no argument: all four)
# output in $BIN_DIR:
#   song_deep1M_{fp32,fp16}  song_gist_{fp32,fp16}  ganns_deep1M_{fp32,fp16}  ganns_gist_{fp32,fp16}
#   ggnn_{fp32,fp16} with the links ggnn_deep1M_* and ggnn_gist_*, cagra_fp16 (both precisions, --precision)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"

BUILD=${FP16_BUILD:-$W/build/fp16}   # patched source copies and the GGNN and CAGRA build trees
JOBS=${JOBS:-12}
NVCC=$ENV_BUILD/bin/nvcc
HOSTCXX=$ENV_BUILD/bin/x86_64-conda-linux-gnu-g++
export PATH=$ENV_BUILD/bin:$PATH     # cmake
SYSTEMS=("$@")
[[ ${#SYSTEMS[@]} -gt 0 ]] || SYSTEMS=(song ganns ggnn cagra)
mkdir -p "$BUILD" "$BIN_DIR"

# Fresh copy of a pinned source tree with one patch applied
patched_copy() {  # <source dir> <destination> <patch>
  [[ -d "$1" ]] || { echo "missing $1 (third_party/versions.tsv)" >&2; exit 1; }
  rm -rf "$2"
  cp -r "$1" "$2"
  patch -p1 -s -d "$2" < "$3"
}

# SONG: dimension, fixed degree and shift are macros (DEEP1M 96 32 6, GIST 960 128 8)
build_song() {  # <dim> <fixed degree> <shift> <fp32|fp16> <output>
  local extra=()
  [[ "$4" == fp16 ]] && extra+=(-DSONG_FP16)
  "$NVCC" -arch=sm_70 \
    -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 \
    -std=c++11 "$HERE/song/song_resident_search.cu" -lcublas -O3 -ccbin="$HOSTCXX" \
    -o "$5" -Xptxas -O3 \
    -D__USE_L2_DIST -D__ENABLE_FIXHASH -D__ENABLE_VISITED_DEL \
    -DDIM="$1" -DFIXED_DEGREE="$2" -DFIXED_DEGREE_SHIFT="$3" ${extra[@]+"${extra[@]}"} \
    -I"$BUILD/song_src"
  echo "built $5 (dim=$1 fd=$2 shift=$3 $4)"
}

# GANNS: the dimension is the macro PLACE_HOLDER_DIM
build_ganns() {  # <dim> <fp32|fp16> <output>
  local extra=()
  [[ "$2" == fp16 ]] && extra+=(-DGANNS_FP16)
  "$NVCC" -O3 -std=c++14 -DPLACE_HOLDER_DIM="$1" -DUSE_DIST_L2_ ${extra[@]+"${extra[@]}"} \
    "--generate-code=arch=compute_70,code=[compute_70,sm_70]" \
    "--generate-code=arch=compute_75,code=[compute_75,sm_75]" \
    "--generate-code=arch=compute_80,code=[compute_80,sm_80]" \
    -I"$BUILD/ganns_src/src" -ccbin "$HOSTCXX" \
    "$HERE/ganns/ganns_resident_search.cu" -o "$3"
  echo "built $3 (dim=$1 $2)"
}

# GGNN library with the __half type, in its own build tree; both drivers link it
build_ggnn_lib() {
  patched_copy "$W/src/ggnn" "$BUILD/ggnn_src" "$HERE/ggnn/ggnn_fp16.patch"
  cmake -S "$BUILD/ggnn_src" -B "$BUILD/ggnn_build" -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_COMPILER="$NVCC" -DCMAKE_CXX_COMPILER="$ENV_BUILD/bin/c++" \
    -DGGNN_CUDA_ARCHITECTURES="70-real;80-real" -DCMAKE_CUDA_ARCHITECTURES="70-real;80-real" \
    -DCMAKE_PREFIX_PATH="$ENV_BUILD" -DCMAKE_DISABLE_FIND_PACKAGE_Python=TRUE \
    > "$BUILD/ggnn_cmake.log" 2>&1 || { tail -30 "$BUILD/ggnn_cmake.log"; exit 1; }
  cmake --build "$BUILD/ggnn_build" -j "$JOBS" --target GGNNBase GGNNConstruction GGNNQuery \
    > "$BUILD/ggnn_make.log" 2>&1 || { grep -n -i -B2 -A8 'error' "$BUILD/ggnn_make.log" | head -80; exit 1; }
}

build_ggnn() {  # <fp32|fp16> <output>
  local extra=()
  [[ "$1" == fp16 ]] && extra+=(-DGGNN_FP16)
  "$NVCC" -O3 -std=c++20 -arch=sm_70 \
    -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80 \
    -ccbin "$HOSTCXX" -x cu ${extra[@]+"${extra[@]}"} \
    "$HERE/ggnn/ggnn_resident_search.cpp" -o "$2" \
    -I"$BUILD/ggnn_src/include" -I"$ENV_BUILD/include" \
    -L"$BUILD/ggnn_build" -L"$ENV_BUILD/lib" \
    -lGGNNBase -lGGNNConstruction -lGGNNQuery -lglog -lgflags -lcurand \
    -Xlinker -rpath -Xlinker "$BUILD/ggnn_build" \
    -Xlinker -rpath -Xlinker "$ENV_BUILD/lib"
  echo "built $2 ($1)"
}

# CAGRA: cuVS 25.02 from $ENV_CAGRA through its CMake package. A plain nvcc command does not work here:
# RMM needs a newer CCCL than the one shipped with CUDA 12.4.
build_cagra() {
  cmake -S "$HERE/cagra" -B "$BUILD/cagra_build" -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$ENV_CAGRA" \
    -DCMAKE_CUDA_COMPILER="$ENV_CAGRA/bin/nvcc" -DCMAKE_CXX_COMPILER="$HOSTCXX" -DCMAKE_CUDA_HOST_COMPILER="$HOSTCXX"
  cmake --build "$BUILD/cagra_build" -j 8
  cp "$BUILD/cagra_build/cagra_resident_fp16" "$BIN_DIR/cagra_fp16"
  echo "built $BIN_DIR/cagra_fp16"
}

for s in "${SYSTEMS[@]}"; do
  case "$s" in
    song)
      patched_copy "$W/src/song" "$BUILD/song_src" "$HERE/song/song_fp16.patch"
      for p in fp32 fp16; do
        build_song 96 32 6 "$p" "$BIN_DIR/song_deep1M_$p"
        build_song 960 128 8 "$p" "$BIN_DIR/song_gist_$p"
      done ;;
    ganns)
      patched_copy "$W/src/ganns" "$BUILD/ganns_src" "$HERE/ganns/ganns_fp16.patch"
      for p in fp32 fp16; do
        build_ganns 96 "$p" "$BIN_DIR/ganns_deep1M_$p"
        build_ganns 960 "$p" "$BIN_DIR/ganns_gist_$p"
      done ;;
    ggnn)
      build_ggnn_lib
      for p in fp32 fp16; do
        build_ggnn "$p" "$BIN_DIR/ggnn_$p"
        ln -sfn "ggnn_$p" "$BIN_DIR/ggnn_deep1M_$p"
        ln -sfn "ggnn_$p" "$BIN_DIR/ggnn_gist_$p"
      done ;;
    cagra) build_cagra ;;
    *) echo "unknown system $s (song ganns ggnn cagra)" >&2; exit 1 ;;
  esac
done
