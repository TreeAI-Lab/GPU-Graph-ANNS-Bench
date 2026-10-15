#!/bin/bash
# Table 7: rebuild every selected configuration, verify it, and time it.
#
# For each configuration of selected_configs.tsv, strictly one build at a time:
#   1. Verify: build once (this build is also the warm-up and is not timed) and sweep the common searcher over L.
#      The configuration passes if its distance computations at Recall@10 = 0.9 do not exceed B. Otherwise the
#      fallback_args of the row are verified the same way; if neither passes, the configuration is skipped.
#   2. Time: build REPS more times with the verified parameters, without searching.
#   After every build only logs and tables are kept and the graph files are deleted.
#   3. Summarize into $OUT_DIR/construction/verify.tsv and timing.tsv (collect_timing.py).
#
# Usage:
#   bash run_timing.sh                              all 28 configurations
#   DATASETS="sift1M deep1M" bash run_timing.sh     some datasets
#   SYSTEMS="cagra ggnn" bash run_timing.sh         some systems
#   EXCLUDE="song" bash run_timing.sh               skip SONG (the slowest builder, which can run separately)
#   FORCE=1 bash run_timing.sh                      run although other processes use the GPU (timings are then off)
# Finished configurations are skipped, so an interrupted run continues with the same command.
# Nothing else may run on the machine while timing.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)

THREADS=${THREADS:-14}       # CPU threads of the builders and of the searcher (the paper: 14, the CPU quota of its machine)
REPS=${REPS:-3}              # timed builds per configuration
DATASETS=${DATASETS:-"sift1M deep1M nytimes glove200"}
SYSTEMS=${SYSTEMS:-"cagra ganns ggnn hnsw pilotann song vamana"}
EXCLUDE=${EXCLUDE:-}
CFG=${CFG:-$HERE/selected_configs.tsv}

BS=$BENCH/src/build/build.sh
PY=$ENV_BUILD/bin/python
OP=$BENCH/src/unified_search/operating_point.py
OUT=$OUT_DIR/construction
mkdir -p "$OUT/logs"; LOG=$OUT/logs/timing_$(date +%Y%m%d_%H%M%S).log
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
# L lists of the sweep, per dataset
lslist(){ case $1 in sift1M|deep1M) echo 10,16,24,32,48,64,96,128,192,256,384,512;;
          nytimes) echo 10,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048;;
          *) echo 10,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096;; esac; }
tagdir(){ "$PY" "$BENCH/src/build/tags.py" "$1" "$2"; }

# Timing requires a GPU without other processes
busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -gt 0 ] && [ "${FORCE:-0}" != 1 ]; then
  say "$busy other processes on the GPU, timings would be off. Stop them first, or set FORCE=1."; nvidia-smi; exit 1
fi
say "timing start: datasets=[$DATASETS] systems=[$SYSTEMS] exclude=[$EXCLUDE] reps=$REPS threads=$THREADS gpu=$GPU log=$LOG"

# Build failures are not fatal here: verification records them as FAIL and the summary counts only finished builds
tail -n +2 "$CFG" | while IFS=$'\t' read -r ds B sys args nd_sweep fb; do
  [[ " $DATASETS " == *" $ds "* ]] || continue
  [[ " $SYSTEMS " == *" $sys "* ]] || continue
  if [[ " $EXCLUDE " == *" $sys "* ]]; then continue; fi
  use=""
  for which in selected fallback; do
    a=$args; if [ $which = fallback ]; then a=$fb; fi
    if [ "$a" = "-" ]; then continue; fi
    vdir=$INDEX_DIR/$ds/$sys/$(tagdir "$sys" "$a")_verify
    if [ ! -f "$vdir/search.tsv" ]; then
      say "verify  $ds $sys ($which) $a"
      env GPU="$GPU" THREADS="$THREADS" LS="$(lslist "$ds")" CLEAN=1 TAGSUFFIX=_verify bash "$BS" "$sys" "$ds" $a >> "$LOG" 2>&1 || true
    fi
    res=$([ -f "$vdir/search.tsv" ] && "$PY" "$OP" --check "$vdir/search.tsv" "$B" || echo "NA FAIL")
    say "verify  $ds $sys ($which): ndist@0.9 = ${res% *} (parameter search ${nd_sweep}, B $B) -> ${res#* }"
    if [ "${res#* }" = PASS ]; then use=$a; break; fi
  done
  if [ -z "$use" ]; then say "SKIP    $ds $sys: neither the selected nor the fallback parameters pass, not timed"; continue; fi
  for r in $(seq 1 "$REPS"); do
    rdir=$INDEX_DIR/$ds/$sys/$(tagdir "$sys" "$use")_rep$r
    if [ -f "$rdir/stats.tsv" ]; then say "timing  $ds $sys rep $r done before, skipped"; continue; fi
    say "timing  $ds $sys rep $r/$REPS $use"
    env GPU="$GPU" THREADS="$THREADS" NOSWEEP=1 CLEAN=1 TAGSUFFIX=_rep$r bash "$BS" "$sys" "$ds" $use >> "$LOG" 2>&1 || true
  done
done

"$PY" "$HERE/collect_timing.py" "$CFG" "$INDEX_DIR" "$OUT" | tee -a "$LOG"
say "timing done: $OUT/verify.tsv and $OUT/timing.tsv"
