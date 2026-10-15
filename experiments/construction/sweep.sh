#!/bin/bash
# Parameter search for Table 7: build every configuration of a grid and sweep the common searcher over L.
# The search uses only distance computation counts, so builds run in parallel and the machine need not be
# exclusive.
#
# Usage: bash experiments/construction/sweep.sh [grid.tsv]      (default: sweep_grid.tsv next to this script)
# The grid has the columns dataset, system, build_args (KEY=VALUE pairs of src/build/build.sh).
# Scheduling: configurations of CAGRA, GANNS, GGNN and PilotANN are dealt to the devices in GPUS and run one after
# another on each device; HNSW and Vamana each run one after another on the CPU; every SONG build (single-threaded)
# runs on its own. All queues run at the same time. Configurations that already have a search.tsv are skipped.
# Graphs: $INDEX_DIR/<dataset>/<system>/<tag>_sweep/. Logs: $OUT_DIR/construction/logs/sweep_*.log.
# Afterwards: python experiments/construction/collect_sweep.py, see README.md.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
HERE=$(cd "$(dirname "$0")" && pwd)

GRID=${1:-$HERE/sweep_grid.tsv}
GPUS=(${GPUS:-$GPU})                    # devices for the GPU builders, e.g. GPUS="0 1 2 3"
THREADS=${THREADS:-$(nproc)}            # CPU threads of every build

BS=$BENCH/src/build/build.sh
PY=$ENV_BUILD/bin/python
LOGDIR=$OUT_DIR/construction/logs; mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d_%H%M%S)
# L lists of the sweep, per dataset
lslist(){ case $1 in sift1M|deep1M) echo 10,16,24,32,48,64,96,128,192,256,384,512;;
          nytimes) echo 10,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048;;
          *) echo 10,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096;; esac; }
tagdir(){ "$PY" "$BENCH/src/build/tags.py" "$1" "$2"; }

# run_queue <name> <gpu> <job>...   jobs are "dataset|system|build_args", run one after another
run_queue(){
  local name=$1 gpu=$2 job ds sys args; shift 2
  for job in "$@"; do
    IFS='|' read -r ds sys args <<< "$job"
    if [ -f "$INDEX_DIR/$ds/$sys/$(tagdir "$sys" "$args")_sweep/search.tsv" ]; then echo "skip $ds $sys $args"; continue; fi
    echo "[$(date '+%F %T')] $ds $sys $args gpu=$gpu"
    env GPU="$gpu" THREADS="$THREADS" TAGSUFFIX=_sweep SEARCH_REPS=1 LS="$(lslist "$ds")" bash "$BS" "$sys" "$ds" $args || echo "FAILED $ds $sys $args"
  done > "$LOGDIR/sweep_${STAMP}_$name.log" 2>&1
}

gpu_jobs=(); hnsw_jobs=(); vamana_jobs=(); song_jobs=()
while IFS=$'\t' read -r ds sys args; do
  case $sys in
    cagra|ganns|ggnn|pilotann) gpu_jobs+=("$ds|$sys|$args") ;;
    hnsw) hnsw_jobs+=("$ds|$sys|$args") ;;
    vamana) vamana_jobs+=("$ds|$sys|$args") ;;
    song) song_jobs+=("$ds|$sys|$args") ;;
    *) echo "unknown system $sys in $GRID"; exit 1 ;;
  esac
done < <(tail -n +2 "$GRID")
echo "sweep: ${#gpu_jobs[@]} GPU, ${#hnsw_jobs[@]} HNSW, ${#vamana_jobs[@]} Vamana, ${#song_jobs[@]} SONG configurations; logs $LOGDIR/sweep_${STAMP}_*.log"

# SONG reads LIBSVM text; write it once per dataset before the parallel SONG builds start
for job in "${song_jobs[@]}"; do
  ds=${job%%|*}
  [ -f "$DATA_DIR/$ds/base.libsvm" ] || "$PY" "$BENCH/datasets/to_libsvm.py" "$DATA_DIR/$ds/base.fbin" "$DATA_DIR/$ds/base.libsvm"
done

pids=()
for g in "${!GPUS[@]}"; do
  q=()
  for ((i = g; i < ${#gpu_jobs[@]}; i += ${#GPUS[@]})); do q+=("${gpu_jobs[$i]}"); done
  [ ${#q[@]} -eq 0 ] || { run_queue "gpu${GPUS[$g]}" "${GPUS[$g]}" "${q[@]}" & pids+=($!); }
done
[ ${#hnsw_jobs[@]} -eq 0 ] || { run_queue hnsw "${GPUS[0]}" "${hnsw_jobs[@]}" & pids+=($!); }
[ ${#vamana_jobs[@]} -eq 0 ] || { run_queue vamana "${GPUS[0]}" "${vamana_jobs[@]}" & pids+=($!); }
for i in "${!song_jobs[@]}"; do
  run_queue "song$i" "${GPUS[0]}" "${song_jobs[$i]}" & pids+=($!)
done
[ ${#pids[@]} -eq 0 ] || wait "${pids[@]}"
echo "sweep done"
