#!/bin/bash
# Table 7, QPS column: QPS of the common searcher on the selected configurations, CPU and GPU version on the same
# graph.
#
# Stage build: rebuild the 28 graphs of selected_configs.tsv into $INDEX_DIR/<dataset>/<system>/<tag>_qps/,
#   keeping only graph.csr, logs and tables. Builds are not timed, so the SONG builds run in parallel with each other
#   and the other builds run one after another.
# Stage search: strictly one configuration at a time. On each graph the CPU version (THREADS threads) and then the
#   GPU version sweep L; per L each runs one untimed warm-up pass and REPS timed passes over all queries and reports
#   the mean time of the timed passes. The L lists are those of run_timing.sh. Nothing else may run on the machine
#   during this stage.
# Output: $OUT_DIR/construction/gpu_qps/<dataset>_<system>.cpu.tsv and .gpu.tsv, summarized by collect_gpu_qps.py
#   into $OUT_DIR/construction/gpu_qps.tsv.
# Usage: bash run_gpu_qps.sh        configurations with output are skipped; STAGE=build or STAGE=search runs one stage
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)

REPS=${REPS:-3}              # timed passes per L of the CPU and GPU versions, after one warm-up pass
THREADS=${THREADS:-14}       # CPU threads of the builders and of the CPU version
STAGE=${STAGE:-all}
CFG=${CFG:-$HERE/selected_configs.tsv}

BS=$BENCH/src/build/build.sh
PY=$ENV_BUILD/bin/python
OUT=$OUT_DIR/construction
RAW=$OUT/gpu_qps
mkdir -p "$RAW" "$OUT/logs"; LOG=$OUT/logs/gpu_qps_$(date +%Y%m%d_%H%M%S).log
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
# L lists of the sweep, per dataset
lslist(){ case $1 in sift1M|deep1M) echo 10,16,24,32,48,64,96,128,192,256,384,512;;
          nytimes) echo 10,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048;;
          *) echo 10,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096;; esac; }
tagdir(){ "$PY" "$BENCH/src/build/tags.py" "$1" "$2"; }
ROWS=$(tail -n +2 "$CFG")

build_one(){  # build_one <dataset> <system> <args>
  local ds=$1 sys=$2 args=$3 dir
  dir=$INDEX_DIR/$1/$2/$(tagdir "$2" "$3")_qps
  if [ -f "$dir/graph.csr" ]; then say "have  $ds $sys"; return; fi
  say "build $ds $sys $args"
  env GPU="$GPU" THREADS="$THREADS" NOSWEEP=1 TAGSUFFIX=_qps bash "$BS" "$sys" "$ds" $args > "$OUT/logs/qps_build_${ds}_${sys}.out" 2>&1 || true
  if [ -f "$dir/graph.csr" ]; then
    find "$dir" -maxdepth 1 -type f ! -name graph.csr ! -name '*.txt' ! -name '*.tsv' ! -name '*.log' -delete; rm -rf "$dir"/*/
    say "built $ds $sys: $(tail -1 "$dir/stats.tsv" | cut -f3,5,10)"
  else
    say "FAILED build $ds $sys, see $OUT/logs/qps_build_${ds}_${sys}.out"
  fi
}

if [ "$STAGE" = all ] || [ "$STAGE" = build ]; then
  say "stage build start"
  pids=()
  while IFS=$'\t' read -r ds B sys args nd fb; do
    if [ "$sys" = song ]; then build_one "$ds" "$sys" "$args" & pids+=($!); fi
  done <<< "$ROWS"
  while IFS=$'\t' read -r ds B sys args nd fb; do
    if [ "$sys" != song ]; then build_one "$ds" "$sys" "$args"; fi
  done <<< "$ROWS"
  [ ${#pids[@]} -eq 0 ] || wait "${pids[@]}"
  say "stage build done"
fi

if [ "$STAGE" = all ] || [ "$STAGE" = search ]; then
  busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$busy" -gt 0 ] && [ "${FORCE:-0}" != 1 ]; then say "$busy other processes on the GPU, timings would be off. Stop them first, or set FORCE=1."; exit 1; fi
  say "stage search start: threads=$THREADS reps=$REPS gpu=$GPU"
  while IFS=$'\t' read -r ds B sys args nd fb; do
    g=$INDEX_DIR/$ds/$sys/$(tagdir "$sys" "$args")_qps/graph.csr; o=$RAW/${ds}_${sys}
    if [ ! -f "$g" ]; then say "MISSING $g"; continue; fi
    common=(--base "$DATA_DIR/$ds/base.fbin" --query "$DATA_DIR/$ds/query.fbin" --gt "$DATA_DIR/$ds/gt.ibin"
            --graph "$g" --entry "$DATA_DIR/$ds/entry_m8.ibin" --L "$(lslist "$ds")")
    if [ ! -s "$o.cpu.tsv" ]; then
      say "cpu   $ds $sys"
      "$BIN_DIR/unified_search" search "${common[@]}" --threads "$THREADS" --reps "$REPS" > "$o.cpu.part" 2>> "$LOG" && mv "$o.cpu.part" "$o.cpu.tsv" || say "FAILED cpu $ds $sys"
    fi
    if [ ! -s "$o.gpu.tsv" ]; then
      say "gpu   $ds $sys"
      CUDA_VISIBLE_DEVICES=$GPU "$BIN_DIR/unified_search_gpu" "${common[@]}" --reps "$REPS" > "$o.gpu.part" 2>> "$LOG" && mv "$o.gpu.part" "$o.gpu.tsv" || say "FAILED gpu $ds $sys"
    fi
  done <<< "$ROWS"
  "$PY" "$HERE/collect_gpu_qps.py" "$CFG" "$RAW" "$OUT/gpu_qps.tsv" | tee -a "$LOG"
  say "stage search done"
fi
