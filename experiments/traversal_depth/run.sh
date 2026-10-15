#!/bin/bash
# Figure 5(b): build the graphs of configs.tsv and sweep the common searcher over L, giving the hops at
# Recall@10 = 0.9.
#
# Only counts are used, nothing is timed, so builds run in parallel:
#   every SONG build (a single-threaded CPU program) runs on its own, all at the same time;
#   the other configurations are dealt in order to the devices in GPUS, run one after another on each device,
#   each with THREADS CPU threads.
# Graphs are kept in $INDEX_DIR/<dataset>/<system>/<tag>_depth/ (the suffix keeps them apart from the graphs of
# the construction experiment with the same parameters).
#
# Usage:
#   bash run.sh                             all 28 configurations
#   DATASETS="sift1M deep1M" bash run.sh    some datasets
#   GPUS="0 1 2 3" bash run.sh              spread the GPU builds over four devices
# Configurations that already have a search.tsv are skipped, so an interrupted run continues with the same command.
# At the end collect.py writes $OUT_DIR/traversal_depth/hops.tsv, see README.md.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)

DATASETS=${DATASETS:-"sift1M deep1M deep10M gist"}
THREADS=${THREADS:-16}
GPUS=(${GPUS:-$GPU})
LS=10,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048

BS=$BENCH/src/build/build.sh
PY=$ENV_BUILD/bin/python
OUT=$OUT_DIR/traversal_depth
mkdir -p "$OUT/logs"; LOG=$OUT/logs/run_$(date +%Y%m%d_%H%M%S).log
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
tagdir(){ "$PY" "$BENCH/src/build/tags.py" "$1" "$2"; }

run_one(){  # run_one <gpu> <dataset> <system> <args>
  local gpu=$1 ds=$2 sys=$3 args=$4 dir
  dir=$INDEX_DIR/$ds/$sys/$(tagdir "$sys" "$args")_depth
  if [ -f "$dir/search.tsv" ]; then say "skip   $ds $sys (done)"; return; fi
  say "start  $ds $sys $args gpu=$gpu"
  env GPU="$gpu" THREADS="$THREADS" LS="$LS" TAGSUFFIX=_depth SEARCH_REPS=1 bash "$BS" "$sys" "$ds" $args > "$OUT/logs/${ds}_${sys}.log" 2>&1 \
    && say "done   $ds $sys: $(tail -1 "$dir/stats.tsv" 2>/dev/null | cut -f1-4)" \
    || say "FAILED $ds $sys, see $OUT/logs/${ds}_${sys}.log"
}

say "start: datasets=[$DATASETS] threads=$THREADS gpus=[${GPUS[*]}] log=$LOG"
song=(); other=()
while IFS=$'\t' read -r ds deg sys args; do
  [[ " $DATASETS " == *" $ds "* ]] || continue
  if [ "$sys" = song ]; then song+=("$ds|$sys|$args"); else other+=("$ds|$sys|$args"); fi
done < <(tail -n +2 "$HERE/configs.tsv")

pids=()
for job in "${song[@]}"; do
  IFS='|' read -r ds sys args <<< "$job"; run_one "${GPUS[0]}" "$ds" "$sys" "$args" & pids+=($!)
done
for g in "${!GPUS[@]}"; do
  (
    for ((i = g; i < ${#other[@]}; i += ${#GPUS[@]})); do
      IFS='|' read -r ds sys args <<< "${other[$i]}"; run_one "${GPUS[$g]}" "$ds" "$sys" "$args"
    done
  ) & pids+=($!)
done
wait "${pids[@]}"
"$PY" "$HERE/collect.py" "$INDEX_DIR" "$OUT/hops.tsv" | tee -a "$LOG"
say "done: $OUT/hops.tsv"
