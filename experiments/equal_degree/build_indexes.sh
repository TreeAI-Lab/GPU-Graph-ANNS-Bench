#!/bin/bash
# Tables 5 and 6 (and the graphs searched by the run scripts): build one index BUILD_RUNS times (default 5) and
# record the construction time, peak host memory, average degree and components of every build;
# tools/parse_logs.py build reports the mean construction time. The index of the last build is kept.
#
# Usage:  bash build_indexes.sh <song|ggnn|ganns|cagra|vamana|pilotann> <dataset> [degree]
# Output: index under $INDEX_DIR/<dataset>/<system>/eqdeg-d<degree>/, log under
#         $OUT_DIR/equal_degree/build/<system>/<dataset>-d<degree>.b<i>.log for build i (read by tools/parse_logs.py build)
#
# vamana builds the DiskANN disk index searched by Vamana (search_disk_index) and converts it into the
# input files of BANG. The peak host memory comes from GNU time -v; peak GPU memory is not recorded.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

SYS=${1:?usage: build_indexes.sh <system> <dataset> [degree]}
DS=${2:?usage: build_indexes.sh <system> <dataset> [degree]}
DEG=${3:-$(degree_of "$DS")}
BUILD_RUNS=${BUILD_RUNS:-5}
GGNN_TAU=${GGNN_TAU:-0.3}                    # tau_build
SONG_L=${SONG_L:-$(build_l_of "$DEG")}       # construction L of SONG
GANNS_E=${GANNS_E:-$(build_l_of "$DEG")}     # candidate list size e of GANNS, the L of SONG
VAMANA_L=${VAMANA_L:-$(build_l_of "$DEG")}  # construction candidate list, shared with SONG
VAMANA_B=${VAMANA_B:-80}                     # search memory budget, GB
VAMANA_M=${VAMANA_M:-80}                     # build memory budget, GB
PILOT_ALGO=${PILOT_ALGO:-nsg}      # PilotANN graph type
DISKANN_APPS=${DISKANN_APPS:-$W/build/diskann-instrumented/apps}  # third_party/build/diskann_instrumented.sh

DIM=$(dim_of "$DS")
D=$DATA_DIR/$DS
IDX=$INDEX_DIR/$DS/$SYS/eqdeg-d$DEG
LOGDIR=$EQDEG_OUT/build/$SYS
mkdir -p "$IDX" "$LOGDIR"
read -r -a TIME <<< "$(time_cmd)"

# build_once <log>
build_once() {
local LOG=$1
case $SYS in
  song)
    case $DS in glove200|nytimes) metric=cos ;; *) metric=l2 ;; esac
    bin=$EQDEG_BIN/song/$DS-d$DEG/song
    # song build <data> 0 0 <rows> <dim> 0 <metric> <max_iterations> <graph prefix>; writes <prefix>.graph
    "${TIME[@]}" "$bin" build "$D/base.fvecs" 0 0 "$(nb_of "$DS")" "$DIM" 0 "$metric" "$SONG_L" "$IDX/bfsg" > "$LOG" 2>&1
    python3 "$EQDEG_TOOLS/strip_fbin_header.py" "$D/base.fbin" "$IDX/bfsg.data" >> "$LOG"
    shift=$(sed -n 's/^#define FIXED_DEGREE_SHIFT \([0-9]*\).*/\1/p' "$W/build/equal_degree/song/$DS-d$DEG/config.h")
    python3 "$EQDEG_TOOLS/song_graph_stats.py" "$IDX/bfsg.graph" --shift "$shift" >> "$LOG"
    ;;
  ggnn)
    "${TIME[@]}" "$EQDEG_BIN/ggnn_main_gpu_data" build "$D/base.fvecs" "$IDX" "$DEG" "$GGNN_TAU" > "$LOG" 2>&1
    ;;
  ganns)
    case $DS in glove200|nytimes) metric=cos ;; *) metric=l2 ;; esac
    # k (d_min) is half the degree, the candidate list M is GANNS_E
    dmin=$(( DEG / 2 ))
    "${TIME[@]}" "$EQDEG_BIN/ganns/GANNS_${DIM}_$metric" -b "$D/base.fvecs" nsw "$dmin" "$GANNS_E" "$IDX/graph.nsw" \
      > "$LOG" 2>&1
    ;;
  cagra)
    "${TIME[@]}" "$EQDEG_BIN/cagra_build_search" build "$D/base.fvecs" "$IDX/index" "$DEG" > "$LOG" 2>&1
    ;;
  vamana)
    case $DS in glove200|nytimes) dist=cosine ;; *) dist=l2 ;; esac
    "${TIME[@]}" "$DISKANN_APPS/build_disk_index" --data_type float --dist_fn "$dist" --data_path "$D/base.fbin" \
      --index_path_prefix "$IDX/vamana" -R "$DEG" -L "$VAMANA_L" -B "$VAMANA_B" -M "$VAMANA_M" > "$LOG" 2>&1
    python3 "$EQDEG_TOOLS/vamana_graph_stats.py" "$IDX/vamana_disk.index" "$DIM" 2 "$DEG" >> "$LOG"
    # BANG reads <prefix>_disk.bin and <prefix>_disk_metadata.bin next to the PQ files of the disk index
    # nodes larger than one 4096-byte sector (GIST at degree 128) need the multi-sector converter
    conv=bang_preprocess.py
    [ $((DIM * 4 + 4 + DEG * 4)) -gt 4096 ] && conv=bang_preprocess_mul.py
    python3 "$W/src/bang/BANG_Base/$conv" "$IDX/vamana_disk.index" "$IDX/vamana_disk.bin" \
      "$DIM" 2 "$DEG" > "$IDX/bang_preprocess.log" 2>&1
    ;;
  pilotann)
    data=$EQDEG_OUT/pilotann_data/$DS
    mkdir -p "$data"
    ln -sf "$D/base.fvecs" "$data/base.fvecs"
    ln -sf "$D/query.fvecs" "$data/query.fvecs"
    ln -sf "$(gt_ivecs "$DS")" "$data/groundtruth.ivecs"
    "${TIME[@]}" "$ENV_PILOT/bin/python" "$EQDEG_TOOLS/pilotann_build.py" --dataset_dir "$data" \
      --output "$IDX/$DS.ckpt" --method search-pilot --algorithm "$PILOT_ALGO" --n_neighbors "$DEG" > "$LOG" 2>&1
    "$ENV_PILOT/bin/python" "$EQDEG_TOOLS/pilotann_graph_stats.py" --graph_path "$IDX/$DS.ckpt" >> "$LOG"
    ;;
  *)
    echo "unknown system: $SYS" >&2; exit 1 ;;
esac
}

for ((run = 1; run <= BUILD_RUNS; run++)); do
  build_once "$LOGDIR/$DS-d$DEG.b$run.log"
done
echo "$SYS $DS degree $DEG: index in $IDX, $BUILD_RUNS build logs in $LOGDIR/$DS-d$DEG.b<i>.log"
