#!/bin/bash
# Nsight Compute on the main search kernel at the operating point (Table 9 and the hardware columns of Table 10).
# usage: [PREC=fp16] profile_ncu.sh <gist|deep1M> <system> [kernel regex on the demangled name]
#   The kernel regex defaults to the kernel_regex column of points.tsv (found by find_kernel.sh).
# output: $OUT_DIR/profiling/<dataset>_<system>[_fp16]/ncu/<tag>.ncu-rep and raw.csv (all raw metrics including the
#   stall reasons, in base units), ncu_version.txt, gpu.txt, and the run directory with sweep.tsv
#
# Settings that matter:
#   NCU_CLOCK=none        GPU clocks are not locked (--clock-control none)
#   NCU_REPLAY=application  every metric pass runs the whole program again; with the data loading and PRE_SLEEP_S
#                         each pass starts from an idle GPU instead of a sustained load near the power limit
#   PRE_SLEEP_S=3         sleep before each pass
#   --cache-control all   caches are flushed before each replay pass
#   NCU_LAUNCH_SKIP=1, NCU_LAUNCH_COUNT=1  skip the warm-up launch and measure one launch, the first timed search.
#                         BANG launches its main kernel once per iteration and PilotANN once per chunk of queries;
#                         profile_all.sh and profile_pilotann.sh set these to all launches of the first timed search.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
DS=${1:?dataset}
SYS=${2:?system}
KREGEX=${3:-$(awk -F'\t' -v d="$DS" -v s="$SYS" '$1==d && $2==s {print $6}' "$HERE/points.tsv")}
NCU=${NCU:-ncu}   # Nsight Compute 2024.1.1 for the paper
PREC=${PREC:-fp32}
CLOCK=${NCU_CLOCK:-none}
REPLAY=${NCU_REPLAY:-application}
SKIP=${NCU_LAUNCH_SKIP:-1}
COUNT=${NCU_LAUNCH_COUNT:-1}
export PRE_SLEEP_S=${PRE_SLEEP_S:-3}
TAG=${DS}_${SYS}; [[ "$PREC" == fp16 ]] && TAG=${TAG}_fp16
OUT=$OUT_DIR/profiling/$TAG/ncu
mkdir -p "$OUT"
# Table 9 columns; dram__bytes_read.sum gives the GPU DRAM bytes read per query (Table 10)
METRICS=sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,lts__t_sector_hit_rate.pct,dram__bytes_read.sum
# Weights for combining several launches into one row (summarize.py): duration, active SM cycles, instructions,
# L2 sectors, issue-active cycles
METRICS=$METRICS,gpu__time_duration.sum,sm__cycles_active.sum,smsp__inst_executed.sum,lts__t_sectors.sum,smsp__issue_active.sum
# Actual SM clock during the measurement (the clocks are not locked)
METRICS=$METRICS,sm__cycles_elapsed.avg.per_second
"$NCU" --version | tail -1 | tee "$OUT/ncu_version.txt"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | tee "$OUT/gpu.txt"
"$NCU" --section WarpStateStats --metrics "$METRICS" \
  --clock-control "$CLOCK" --replay-mode "$REPLAY" --cache-control all \
  --kernel-name-base demangled --kernel-name "regex:$KREGEX" --launch-skip "$SKIP" --launch-count "$COUNT" \
  --force-overwrite -o "$OUT/$TAG" \
  bash "$HERE/run_point.sh" "$DS" "$SYS" "$OUT/run" "$PREC" > "$OUT/ncu.log" 2>&1 || true
if grep -q ERR_NVGPUCTRPERM "$OUT/ncu.log"; then echo "FAIL: ERR_NVGPUCTRPERM, run platform/run_smoke.sh first"; exit 2; fi
[[ -f "$OUT/$TAG.ncu-rep" ]] || { echo "FAIL: no report, see $OUT/ncu.log (the kernel regex may not match)"; exit 3; }
"$NCU" --import "$OUT/$TAG.ncu-rep" --csv --page raw --print-units base > "$OUT/raw.csv"
echo "[profile_ncu] $TAG -> $OUT/raw.csv"
