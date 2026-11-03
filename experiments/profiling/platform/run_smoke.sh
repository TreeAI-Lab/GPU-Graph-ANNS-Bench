#!/bin/bash
# Check that Nsight Compute can read the GPU performance counters on this machine.
# Pass: the last line is "SMOKE PASS", after the values of the five Table 9 throughput and occupancy metrics.
# A small gather kernel is launched twice and profiled with the WarpStateStats section, skipping the first launch.
# usage: bash run_smoke.sh
# output: $OUT_DIR/profiling/smoke/ (ncu.log, smoke.csv)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../../env.sh"
NCU=${NCU:-ncu}
NVCC=${NVCC:-$ENV_BUILD/bin/nvcc}
OUT=$OUT_DIR/profiling/smoke
mkdir -p "$OUT"

echo "== machine"
id
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm --format=csv
grep -i profil /proc/driver/nvidia/params || true
grep CapEff /proc/self/status   # root can read the counters when CAP_SYS_ADMIN (bit 21) is set
"$NCU" --version | tail -1
"$NVCC" --version | tail -1

echo "== build"
"$NVCC" -O2 -arch=sm_80 "$HERE/ncu_smoke.cu" -o "$OUT/ncu_smoke" || { echo "SMOKE FAIL: nvcc"; exit 1; }
"$OUT/ncu_smoke" || { echo "SMOKE FAIL: kernel"; exit 1; }

echo "== ncu"
METRICS=sm__throughput.avg.pct_of_peak_sustained_elapsed,dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__thread_inst_executed_per_inst_executed.ratio,lts__t_sector_hit_rate.pct
"$NCU" --section WarpStateStats --metrics "$METRICS" \
  --clock-control base --cache-control all \
  --kernel-name gather --launch-skip 1 --launch-count 1 \
  -f -o "$OUT/smoke" "$OUT/ncu_smoke" > "$OUT/ncu.log" 2>&1 || true
cat "$OUT/ncu.log"
if grep -q ERR_NVGPUCTRPERM "$OUT/ncu.log"; then
  echo "SMOKE FAIL: ERR_NVGPUCTRPERM"
  echo "The driver restricts the counters to administrators. As root on a full VM or bare-metal host, write"
  echo "  options nvidia NVreg_RestrictProfilingToAdminUsers=0"
  echo "to /etc/modprobe.d/nvidia-profiling.conf, run update-initramfs -u (Ubuntu) or dracut -f, reboot, and run this script again."
  exit 2
fi
"$NCU" --import "$OUT/smoke.ncu-rep" --csv --page raw --metrics "$METRICS" > "$OUT/smoke.csv" 2>&1
echo "== metrics"
cat "$OUT/smoke.csv"
if grep -q 'sm__throughput' "$OUT/smoke.csv"; then echo "SMOKE PASS"; else echo "SMOKE FAIL: no metrics"; exit 3; fi
