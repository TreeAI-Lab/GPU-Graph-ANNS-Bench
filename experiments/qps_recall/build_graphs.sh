#!/bin/bash
# Build the graphs and indexes searched in Figures 1, 7, and 11.
# Usage: bash experiments/qps_recall/build_graphs.sh [dataset ...]      (default: all datasets of datasets.sh)
#   SYSTEMS  systems to build, default "cagra ggnn ganns song bang pilotann"
#   FORCE=1  rebuild indexes that already exist
#   BANG_CFG="R L M B QD" replaces the BANG settings of datasets.sh
# Output: $INDEX_DIR/<data>/<system>/<tag>/ with params.txt (parameters, wall time, peak_gpu_mib for CAGRA, GGNN and GANNS) and build.log, plus
#   cagra/<tag>/index.cagra (graph only)   ggnn/<tag>/part_<i>.ggnn   ganns/<tag>/graph.nsw
#   song/<tag>/bfsg.graph                   bang/<tag>/index_*          pilotann/<tag>/index.pt
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/datasets.sh"

SYSTEMS=${SYSTEMS:-"cagra ggnn ganns song bang pilotann"}
FORCE=${FORCE:-0}
THREADS=${THREADS:-14}
PY=${PY:-$ENV_BUILD/bin/python}
# GNU time reports wall time and peak resident memory
TIME_BIN=${TIME_BIN:-/usr/bin/time}; [[ -x $TIME_BIN ]] || TIME_BIN=$ENV_BUILD/bin/time
# Builders made by other parts of the repository
CAGRA_BUILD_PY=$BENCH/src/build/cagra_build.py        # CAGRA with NN-descent, cuVS 25.02 Python
GGNN_BUILD=$BIN_DIR/ggnn_build                         # src/build/build_ggnn_driver.sh
GGNN_SHARDED_BIN=${GGNN_SHARDED_BIN:-$W/build/ggnn-instrumented/ggnn_main_gpu_data_sharded}  # third_party/build/ggnn_instrumented.sh: GGNN with 20M-vector shards (DEEP100M)
DISKANN_BUILD=$W/build/diskann/apps/build_disk_index
BANG_SRC=$W/src/bang/BANG_Base                         # bang_preprocess.py, bang_preprocess_mul.py

done_or_clean() {  # done_or_clean <dir> <file that marks a finished build>
  if [[ -f $1/$2 && $FORCE != 1 ]]; then
    echo "SKIP $1 (exists)"
    return 0
  fi
  rm -rf "$1"
  mkdir -p "$1"
  return 1
}

finish() {  # finish <dir>: wall time and peak RSS from the GNU time report
  local out=$1
  echo "wall_s=$(grep -m1 'Elapsed (wall clock)' "$out/build.log" | awk '{print $NF}') peak_rss_kb=$(grep -m1 'Maximum resident set size' "$out/build.log" | awk '{print $NF}') end=$(date '+%F %T')" >>"$out/params.txt"
  echo "OK $out"
}

with_gpu() {
  local out=$1 mon base
  shift
  nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits >"$out/gpu_mem_baseline.txt" 2>/dev/null || true
  nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 >"$out/gpu_mem.txt" 2>/dev/null &
  mon=$!
  set +e
  "$@"
  local rc=$?
  set -e
  kill "$mon" 2>/dev/null || true
  wait "$mon" 2>/dev/null || true
  if [[ -s $out/gpu_mem.txt ]]; then
    base=$(awk 'NR==1{print $1; exit}' "$out/gpu_mem_baseline.txt" 2>/dev/null || echo 0)
    awk -v b="${base:-0}" 'NF{v=$1+0; if(v>m)m=v} END{printf "peak_gpu_mib=%.0f\n", (m>b?m-b:0)}' "$out/gpu_mem.txt" >>"$out/params.txt"
  fi
  return "$rc"
}

build_cagra() {
  local ds=$1 data n dim deg nq ctag rest out metric
  read -r data n dim deg nq ctag rest <<<"$(cfg_graph "$ds")"
  out=$INDEX_DIR/$data/cagra/$ctag
  done_or_clean "$out" index.cagra && return 0
  echo "system=cagra dataset=$data tag=$ctag n=$n dim=$dim gpu=$GPU start=$(date '+%F %T')" >"$out/params.txt"
  if [[ $ds == deep100M ]]; then
    echo "GD=$deg IGD=$((deg * 2)) graph_builder=ivf_pq" >>"$out/params.txt"
    with_gpu "$out" "$BIN_DIR/cagra_build_ivfpq" build "$DATA_DIR/$data/base.fbin" "$out/index.cagra" "$deg" \
      "$out/build_mem.tsv" >"$out/build.log" 2>&1
    echo "end=$(date '+%F %T')" >>"$out/params.txt"
    echo "OK $out"
    return 0
  fi
  metric=sqeuclidean
  [[ $ds == glove200 || $ds == nytimes ]] && metric=inner_product
  echo "GD=$deg IGD=$((deg * 2)) NITER=20 metric=$metric" >>"$out/params.txt"
  (cd "$out" && with_gpu "$out" "$TIME_BIN" -v "$ENV_CAGRA/bin/python" "$CAGRA_BUILD_PY" \
      "$DATA_DIR/$data/base.fbin" "$out/index.cagra" "$deg" "$((deg * 2))" 20 "$metric" >"$out/build.log" 2>&1)
  finish "$out"
}

build_ggnn() {
  local ds=$1 data n dim deg nq ctag gtag rest out
  read -r data n dim deg nq ctag gtag rest <<<"$(cfg_graph "$ds")"
  [[ $gtag == - ]] && { echo "no GGNN graph for $ds"; return 0; }
  out=$INDEX_DIR/$data/ggnn/$gtag
  done_or_clean "$out" part_0.ggnn && return 0
  echo "system=ggnn dataset=$data tag=$gtag n=$n dim=$dim gpu=$GPU start=$(date '+%F %T')" >"$out/params.txt"
  echo "K=$deg TAU=0.3 REFINE=2" >>"$out/params.txt"
  if [[ $ds == deep100M ]]; then
    # The sharded GGNN build (ggnn-instrumented): base read from fbin, 5 shards of 20,000,000 vectors
    echo "SHARD_SIZE=20000000" >>"$out/params.txt"
    [[ -x $GGNN_SHARDED_BIN ]] || bash "$BENCH/third_party/build/ggnn_instrumented.sh"
    with_gpu "$out" "$TIME_BIN" -v "$GGNN_SHARDED_BIN" build "$DATA_DIR/$data/base.fbin" "$out" "$deg" 0.3 2 >"$out/build.log" 2>&1
  else
    with_gpu "$out" "$TIME_BIN" -v "$GGNN_BUILD" "$DATA_DIR/$data/base.fvecs" "$out" "$deg" 0.3 2 0 >"$out/build.log" 2>&1
  fi
  finish "$out"
}

build_ganns() {
  local ds=$1 data n dim deg nq ctag gtag atag rest out bin k m
  read -r data n dim deg nq ctag gtag atag rest <<<"$(cfg_graph "$ds")"
  [[ $atag == - ]] && { echo "no GANNS graph for $ds"; return 0; }
  out=$INDEX_DIR/$data/ganns/$atag
  done_or_clean "$out" graph.nsw && return 0
  k=$((deg / 2)); m=$(build_l "$deg")
  bin=$W/build/ganns/dim$dim/GANNS
  [[ -x $bin ]] || bash "$BENCH/third_party/build/ganns.sh" "$dim"
  echo "system=ganns dataset=$data tag=$atag n=$n dim=$dim gpu=$GPU start=$(date '+%F %T')" >"$out/params.txt"
  echo "K=$k M=$m" >>"$out/params.txt"
  (cd "$out" && with_gpu "$out" "$TIME_BIN" -v "$bin" -b "$DATA_DIR/$data/base.fvecs" nsw "$k" "$m" graph.nsw >"$out/build.log" 2>&1)
  finish "$out"
}

# SONG fixes the graph layout at compile time: one builder per FIXED_DEGREE, shift and dimension. The builder is
# song-instrumented, which adds an iteration limit (forward degree FIXED_DEGREE/2, base read from fbin).
build_song() {
  local ds=$1 data n dim deg nq ctag gtag atag stag shift out bin metric suffix
  read -r data n dim deg nq ctag gtag atag stag shift <<<"$(cfg_graph "$ds")"
  [[ $stag == - ]] && { echo "no SONG graph for $ds"; return 0; }
  out=$INDEX_DIR/$data/song/$stag
  done_or_clean "$out" bfsg.graph && return 0
  L=$(build_l "$deg")
  metric=l2
  [[ $ds == glove200 || $ds == nytimes ]] && metric=cos
  suffix=
  [[ $metric == l2 ]] || suffix=_$metric
  bin=$W/build/song-instrumented/fd${deg}_s${shift}_dim${dim}${suffix}/song
  [[ -x $bin ]] || bash "$BENCH/third_party/build/song_instrumented.sh" "$deg" "$shift" "$dim" 64 "$metric"
  {
    echo "system=song dataset=$data tag=$stag n=$n dim=$dim start=$(date '+%F %T')"
    echo "FIXED_DEGREE=$deg FIXED_DEGREE_SHIFT=$shift degree=$((deg / 2)) L=$L metric=$metric"
  } >"$out/params.txt"
  # song build <build_data> <query_data> <search_top_k> <row> <dim> <return_top_k> <dist> <max_iter> <graph_file>
  (cd "$out" && "$TIME_BIN" -v "$bin" build "$DATA_DIR/$data/base.fbin" /dev/null 10 "$n" "$dim" 10 "$metric" "$L" bfsg \
      >"$out/build.log" 2>&1)
  finish "$out"
}

# DiskANN disk index, then the conversion to BANG's layout. A node of the disk index (vector, degree, R ids) that
# does not fit in one 4096-byte sector (GIST) needs the multi-sector converter.
build_bang() {
  local ds=$1 data n dim rest R L M B QD tag out prefix nodelen conv dist
  read -r data n dim rest <<<"$(cfg_graph "$ds")"
  read -r R L M B QD <<<"${BANG_CFG:-$(cfg_bang "$ds")}"
  tag=$(bang_tag "$R" "$L")
  out=$INDEX_DIR/$data/bang/$tag
  prefix=$out/index
  done_or_clean "$out" index_disk.bin && return 0
  dist=l2
  [[ $ds == glove200 || $ds == nytimes ]] && dist=cosine
  echo "system=bang dataset=$data tag=$tag n=$n dim=$dim R=$R L=$L M=$M B=$B QD=$QD dist=$dist start=$(date '+%F %T')" \
    >"$out/params.txt"
  LD_LIBRARY_PATH=$ENV_BUILD/lib:${LD_LIBRARY_PATH:-} OMP_NUM_THREADS=$THREADS \
    "$TIME_BIN" -v "$DISKANN_BUILD" --data_type float --dist_fn "$dist" \
      --data_path "$DATA_DIR/$data/base.fbin" --index_path_prefix "$prefix" \
      -R "$R" -L "$L" -M "$M" -B "$B" -T "$THREADS" --QD "$QD" --build_PQ_bytes 0 >"$out/build.log" 2>&1
  nodelen=$((dim * 4 + 4 + R * 4))
  conv=bang_preprocess.py
  [[ $nodelen -gt 4096 ]] && conv=bang_preprocess_mul.py
  "$PY" "$BANG_SRC/$conv" "${prefix}_disk.index" "${prefix}_disk.bin" "$dim" 2 "$R" >>"$out/build.log" 2>&1
  # rewrites <prefix>_pq_pivots.bin in place: the whole file is read before the three outputs are written
  "$PY" "$BENCH/src/graph_tools/split_pq_pivots.py" "${prefix}_pq_pivots.bin" "$prefix" >>"$out/build.log" 2>&1
  [[ -s ${prefix}_disk.bin ]] || { echo "FAILED: empty ${prefix}_disk.bin, see $out/build.log" >&2; return 1; }
  # BANG reads only _disk.bin, _disk_metadata.bin and the PQ files
  rm -f -- "${prefix}_disk.index" "${prefix}_sample_data.bin" "${prefix}_sample_ids.bin"
  finish "$out"
}

build_pilotann() {
  local ds=$1 data rest gtype nn efc dp ratio tag out
  read -r data rest <<<"$(cfg_graph "$ds")"
  read -r gtype nn efc dp ratio <<<"$(cfg_pilotann "$ds")"
  tag=$(pilotann_tag "$gtype" "$nn" "$efc" "$dp" "$ratio")
  out=$INDEX_DIR/$data/pilotann/$tag
  done_or_clean "$out" index.pt && return 0
  echo "system=pilotann dataset=$data tag=$tag start=$(date '+%F %T')" >"$out/params.txt"
  LD_LIBRARY_PATH=$ENV_PILOT/lib:${LD_LIBRARY_PATH:-} \
    "$TIME_BIN" -v "$ENV_PILOT/bin/python" -u "$BENCH/src/search/pilotann/pilotann_build.py" \
      --base "$DATA_DIR/$data/base.fbin" --ckpt "$out/index.pt" --nn "$nn" --efc "$efc" --dp "$dp" \
      --ratio "$ratio" --gtype "$gtype" >"$out/build.log" 2>&1
  finish "$out"
}

in_list() { [[ " $2 " == *" $1 "* ]]; }

DATASETS=${*:-$GRAPH_DATASETS}
for ds in $DATASETS; do
  for sys in $SYSTEMS; do
    case $sys in
      cagra|ggnn|ganns|song) in_list "$ds" "$GRAPH_DATASETS" || continue ;;
      bang) in_list "$ds" "$BANG_DATASETS" || continue ;;
      pilotann) in_list "$ds" "$PILOTANN_DATASETS" || continue ;;
      *) echo "unknown system $sys" >&2; exit 1 ;;
    esac
    echo "===== build $sys $ds $(date '+%F %T') ====="
    "build_$sys" "$ds"
  done
done
