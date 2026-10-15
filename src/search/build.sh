#!/bin/bash
# Build the search drivers into $BIN_DIR.
# Usage: bash src/search/build.sh [cagra|ggnn|ganns|song|all] ...
#   cagra  CAGRA tools (cagra/Makefile), cuVS built by third_party/build/cuvs.sh, environment $ENV_CUVS
#   ggnn   ggnn_resident_search, GGNN libraries built by third_party/build/ggnn.sh in $W/build/ggnn
#   ganns  ganns_resident_search_d<DIM>, one per dimension, against the per-dimension GANNS copy that
#          third_party/build/ganns.sh <DIM> makes in $W/build/ganns/dim<DIM>/src
#   song   song_resident_search_d<DIM>_fd<FD>, one per dimension and graph degree, see build_song below
# GGNN, GANNS and SONG use the CUDA 12.4 toolchain of $ENV_BUILD.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)

# Dimensions of the datasets: 96 (DEEP), 128 (SIFT1M), 200 (GloVe200), 256 (NYTimes), 960 (GIST)
GANNS_DIMS=${GANNS_DIMS:-"96 128 200 256 960"}
# SONG binaries as DIM:FIXED_DEGREE:FIXED_DEGREE_SHIFT. FIXED_DEGREE is the maximum degree of the graph
# (forward degree FIXED_DEGREE/2), each vertex has 2^SHIFT slots in bfsg.graph.
SONG_CONFIGS=${SONG_CONFIGS:-"96:32:6 128:32:6 200:64:7 256:32:6 960:128:8"}
# SONG sources the kernel header was derived from (commit fa2d755, name song-orig in third_party/versions.tsv).
# Its smmh2.h differs from the later SONG commit, so the headers are taken from this tree.
SONG_SRC=${SONG_SRC:-$W/src/song-orig}
WORK=${WORK:-$W/build/search}
JOBS=${JOBS:-4}

NVCC=${NVCC:-$ENV_BUILD/bin/nvcc}
HOST_CXX=${HOST_CXX:-$ENV_BUILD/bin/x86_64-conda-linux-gnu-g++}
GGNN_GENCODE=${GGNN_GENCODE:-"-arch=sm_70 -gencode=arch=compute_70,code=sm_70 -gencode=arch=compute_80,code=sm_80"}
GANNS_GENCODE=${GANNS_GENCODE:-"--generate-code=arch=compute_70,code=[compute_70,sm_70] --generate-code=arch=compute_80,code=[compute_80,sm_80]"}
SONG_GENCODE=${SONG_GENCODE:-"-arch=sm_80 -gencode=arch=compute_80,code=sm_80"}

mkdir -p "$BIN_DIR" "$WORK"

build_cagra() {
  make -C "$HERE/cagra" -j"$JOBS" all W="$W" BIN_DIR="$BIN_DIR" ENV_CUVS="$ENV_CUVS"
}

build_ggnn() {
  # shellcheck disable=SC2086
  "$NVCC" -O3 -std=c++20 $GGNN_GENCODE -ccbin "$HOST_CXX" -x cu \
    "$HERE/ggnn_resident_search.cpp" -o "$BIN_DIR/ggnn_resident_search" \
    -I"$W/src/ggnn/include" -I"$ENV_BUILD/include" \
    -L"$W/build/ggnn" -L"$ENV_BUILD/lib" \
    -lGGNNBase -lGGNNConstruction -lGGNNQuery -lglog -lgflags -lcurand \
    -Xlinker -rpath -Xlinker "$W/build/ggnn" \
    -Xlinker -rpath -Xlinker "$ENV_BUILD/lib"
  echo "built $BIN_DIR/ggnn_resident_search"
}

build_ganns() {
  local dim src
  for dim in $GANNS_DIMS; do
    src=$W/build/ganns/dim$dim/src/src
    if [[ ! -f $src/kernels.h ]]; then
      bash "$BENCH/third_party/build/ganns.sh" "$dim"
    fi
    # shellcheck disable=SC2086
    "$NVCC" -O3 -std=c++14 \
      -DPLACE_HOLDER_DIM="$dim" -DUSE_DIST_L2_ \
      $GANNS_GENCODE \
      -I"$src" -ccbin "$HOST_CXX" \
      "$HERE/ganns_resident_search.cu" -o "$BIN_DIR/ganns_resident_search_d$dim"
    echo "built $BIN_DIR/ganns_resident_search_d$dim"
  done
}

# SONG fixes the dimension and the graph layout at compile time. Each binary gets a workspace with the
# SONG headers it includes: eight unchanged files of $SONG_SRC, astar_accelerator.h and config.h of $SONG_SRC
# with the macros below, and the modified kernel header song/warp_astar_accelerator.h.
song_workspace() {
  local dim=$1 fd=$2 shift=$3 ws=$4
  local f
  rm -rf "$ws"
  mkdir -p "$ws"
  for f in bin_heap.h blocked_bloomfilter.h bloomfilter.h cuckoofilter.h data.h fixhash.h smmh2.h vanilla_list.h \
           astar_accelerator.h config.h; do
    [[ -f $SONG_SRC/$f ]] || { echo "missing $SONG_SRC/$f (fetch the SONG sources first)" >&2; exit 2; }
    cp "$SONG_SRC/$f" "$ws/$f"
  done
  cp "$HERE/song/warp_astar_accelerator.h" "$HERE/song/song_resident_search.cu" "$ws/"
  # TOPK 64 is the value SONG's fill_parameters.sh wrote for the kernel header; SEARCH_DEGREE and
  # CONSTRUCT_SEARCH_BUDGET only affect SONG's CPU graph construction.
  sed -i.orig -e "s/^#define TOPK .*/#define TOPK 64/" "$ws/astar_accelerator.h"
  sed -i.orig \
    -e "s/^#define ACC_BATCH_SIZE .*/#define ACC_BATCH_SIZE 1000000/" \
    -e "s/^#define FIXED_DEGREE .*/#define FIXED_DEGREE $fd/" \
    -e "s/^#define FIXED_DEGREE_SHIFT .*/#define FIXED_DEGREE_SHIFT $shift/" \
    -e "s/^#define SEARCH_DEGREE .*/#define SEARCH_DEGREE 16/" \
    -e "s/^#define CONSTRUCT_SEARCH_BUDGET .*/#define CONSTRUCT_SEARCH_BUDGET 160/" \
    "$ws/config.h"
  sed -i.orig -e "s/#define DIM 96$/#define DIM $dim/" "$ws/warp_astar_accelerator.h"
  rm -f "$ws"/*.orig
  grep -q "^#define FIXED_DEGREE $fd$" "$ws/config.h"
  grep -q "^#define FIXED_DEGREE_SHIFT $shift$" "$ws/config.h"
  grep -q "^#define TOPK 64$" "$ws/astar_accelerator.h"
  grep -q "#define DIM $dim$" "$ws/warp_astar_accelerator.h"
}

build_song() {
  local cfg dim fd shift ws
  for cfg in $SONG_CONFIGS; do
    IFS=: read -r dim fd shift <<<"$cfg"
    ws=$WORK/song_d${dim}_fd${fd}
    song_workspace "$dim" "$fd" "$shift" "$ws"
    # shellcheck disable=SC2086
    "$NVCC" $SONG_GENCODE \
      -std=c++11 "$ws/song_resident_search.cu" -lcublas -O3 -ccbin="$HOST_CXX" \
      -o "$BIN_DIR/song_resident_search_d${dim}_fd${fd}" -Xptxas -O3 \
      -D__USE_L2_DIST \
      -D__ENABLE_FIXHASH -D__ENABLE_VISITED_DEL \
      -I"$ws"
    echo "built $BIN_DIR/song_resident_search_d${dim}_fd${fd}"
  done
}

TARGETS=${*:-all}
for t in $TARGETS; do
  case $t in
    cagra) build_cagra ;;
    ggnn) build_ggnn ;;
    ganns) build_ganns ;;
    song) build_song ;;
    all) build_cagra; build_ggnn; build_ganns; build_song ;;
    *) echo "unknown target $t (cagra|ggnn|ganns|song|all)" >&2; exit 1 ;;
  esac
done
