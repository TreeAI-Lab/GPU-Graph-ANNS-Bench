#!/bin/bash
# Tables 5 and 6 (and the graphs searched by the run scripts): build one index with the settings of the
# original runs and record its construction time, peak host memory, average degree and components.
#
# Usage:  bash build_indexes.sh <song|ggnn|ganns|cagra|vamana|pilotann> <dataset> [degree]
# Output: index under $INDEX_DIR/<dataset>/<system>/orig-d<degree>/, log under
#         $OUT_DIR/original/build/<system>/<dataset>-d<degree>.log (read by tools/parse_logs.py build)
#
# vamana builds the DiskANN disk index searched by Vamana (search_disk_index) and converts it into the
# input files of BANG. The peak host memory comes from GNU time -v; peak GPU memory is not recorded.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
source "$(cd "$(dirname "$0")" && pwd)/common.sh"

SYS=${1:?usage: build_indexes.sh <system> <dataset> [degree]}
DS=${2:?usage: build_indexes.sh <system> <dataset> [degree]}
DEG=${3:-$(degree_of "$DS")}
GGNN_TAU=${GGNN_TAU:-0.3}          # tau_build of the original runs
GANNS_E=${GANNS_E:-60}             # candidate list size e of the GANNS README example
VAMANA_L=${VAMANA_L:-50}           # build_disk_index -L -B -M of the BANG README of the original release
VAMANA_B=${VAMANA_B:-10}
VAMANA_M=${VAMANA_M:-10}
PILOT_ALGO=${PILOT_ALGO:-nsg}      # PilotANN graph type of the original runs
DISKANN_APPS=${DISKANN_APPS:-$W/build/diskann-orig/apps}  # third_party/build/diskann_orig.sh

DIM=$(dim_of "$DS")
D=$DATA_DIR/$DS
IDX=$INDEX_DIR/$DS/$SYS/orig-d$DEG
LOG=$ORIG_OUT/build/$SYS/$DS-d$DEG.log
mkdir -p "$IDX" "$(dirname "$LOG")"
read -r -a TIME <<< "$(time_cmd)"

case $SYS in
  song)
    case $DS in nytimes-full|nytimes|glove200) metric=cos ;; *) metric=l2 ;; esac
    bin=$ORIG_BIN/song/$DS-d$DEG/song
    # song build <data> 0 0 <rows> <dim> 0 <metric> <max_iterations> <graph prefix>; writes <prefix>.graph
    "${TIME[@]}" "$bin" build "$D/base.fvecs" 0 0 "$(nb_of "$DS")" "$DIM" 0 "$metric" 200 "$IDX/bfsg" > "$LOG" 2>&1
    python3 "$ORIG_TOOLS/strip_fbin_header.py" "$D/base.fbin" "$IDX/bfsg.data" >> "$LOG"
    shift=$(sed -n 's/^#define FIXED_DEGREE_SHIFT \([0-9]*\).*/\1/p' "$W/build/original/song/$DS-d$DEG/config.h")
    python3 "$ORIG_TOOLS/song_graph_stats.py" "$IDX/bfsg.graph" --shift "$shift" >> "$LOG"
    ;;
  ggnn)
    "${TIME[@]}" "$ORIG_BIN/ggnn_main_gpu_data" build "$D/base.fvecs" "$IDX" "$DEG" "$GGNN_TAU" > "$LOG" 2>&1
    ;;
  ganns)
    case $DS in nytimes-full|nytimes|glove200) metric=cos ;; *) metric=l2 ;; esac
    # the builder writes <base path>_<e>_<d_min>.nsw next to the base file; d_min is half the degree
    ln -sf "$D/base.fvecs" "$IDX/base.fvecs"
    dmin=$(( DEG / 2 ))
    "${TIME[@]}" "$ORIG_BIN/ganns/build_${DIM}_$metric" "$IDX/base.fvecs" nsw "$GANNS_E" "$dmin" > "$LOG" 2>&1
    mv "$IDX/base.fvecs_${GANNS_E}_$dmin.nsw" "$IDX/graph.nsw"
    rm -f "$IDX/base.fvecs"
    ;;
  cagra)
    "${TIME[@]}" "$ORIG_BIN/my_cagra" build "$D/base.fvecs" "$IDX/index" "$DEG" > "$LOG" 2>&1
    ;;
  vamana)
    "${TIME[@]}" "$DISKANN_APPS/build_disk_index" --data_type float --dist_fn l2 --data_path "$D/base.fbin" \
      --index_path_prefix "$IDX/vamana" -R "$DEG" -L "$VAMANA_L" -B "$VAMANA_B" -M "$VAMANA_M" > "$LOG" 2>&1
    python3 "$ORIG_TOOLS/vamana_graph_stats.py" "$IDX/vamana_disk.index" "$DIM" 2 "$DEG" >> "$LOG"
    # BANG reads <prefix>_disk.bin and <prefix>_disk_metadata.bin next to the PQ files of the disk index
    # nodes larger than one 4096-byte sector (GIST at degree 128) need the multi-sector converter
    conv=bang_preprocess.py
    [ $((DIM * 4 + 4 + DEG * 4)) -gt 4096 ] && conv=bang_preprocess_mul.py
    python3 "$W/src/bang/BANG_Base/$conv" "$IDX/vamana_disk.index" "$IDX/vamana_disk.bin" \
      "$DIM" 2 "$DEG" > "$IDX/bang_preprocess.log" 2>&1
    ;;
  pilotann)
    data=$ORIG_OUT/pilotann_data/$DS
    mkdir -p "$data"
    ln -sf "$D/base.fvecs" "$data/base.fvecs"
    ln -sf "$D/query.fvecs" "$data/query.fvecs"
    ln -sf "$(gt_ivecs "$DS")" "$data/groundtruth.ivecs"
    "${TIME[@]}" "$ENV_PILOT/bin/python" "$ORIG_TOOLS/pilotann_build.py" --dataset_dir "$data" \
      --output "$IDX/$DS.ckpt" --method search-pilot --algorithm "$PILOT_ALGO" --n_neighbors "$DEG" > "$LOG" 2>&1
    "$ENV_PILOT/bin/python" "$ORIG_TOOLS/pilotann_graph_stats.py" --graph_path "$IDX/$DS.ckpt" >> "$LOG"
    ;;
  *)
    echo "unknown system: $SYS" >&2; exit 1 ;;
esac
echo "$SYS $DS degree $DEG: index in $IDX, log in $LOG"
