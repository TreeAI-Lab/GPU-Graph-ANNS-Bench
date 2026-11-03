#!/bin/bash
# Record the configuration of the machine the measurements ran on, as plain text files.
# usage: bash collect_platform.sh [out dir, default $OUT_DIR/profiling/platform]
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../../env.sh"
OUT=${1:-$OUT_DIR/profiling/platform}
NCU=${NCU:-ncu}
NVCC=${NVCC:-$ENV_BUILD/bin/nvcc}
mkdir -p "$OUT"

date -Is > "$OUT/collected_at.txt"
nvidia-smi -q > "$OUT/nvidia-smi-q.txt" 2>&1 || true
nvidia-smi --query-gpu=name,pci.bus_id,driver_version,vbios_version,memory.total,power.limit,clocks.max.sm,clocks.max.memory \
  --format=csv > "$OUT/gpu.csv" 2>&1 || true
{ cat /proc/driver/nvidia/version; grep -i profil /proc/driver/nvidia/params; } > "$OUT/driver.txt" 2>&1 || true
{ "$NCU" --version; "$NVCC" --version; } > "$OUT/toolkit.txt" 2>&1 || true
{ cat /etc/os-release; uname -a; systemd-detect-virt; cat /sys/class/dmi/id/product_name
  [ -f /.dockerenv ] && echo "container: /.dockerenv present"; } > "$OUT/os.txt" 2>&1 || true
{ lscpu; free -g; df -h -x tmpfs -x devtmpfs -x overlay; } > "$OUT/host.txt" 2>&1 || true
# In a container lscpu and free show the host; the actual quota is in the cgroup (v1 or v2)
{ echo "nproc $(nproc)"
  for f in /sys/fs/cgroup/cpu.max /sys/fs/cgroup/cpu/cpu.cfs_quota_us /sys/fs/cgroup/cpu/cpu.cfs_period_us \
           /sys/fs/cgroup/memory.max /sys/fs/cgroup/memory/memory.limit_in_bytes; do
    if [ -r "$f" ]; then echo "$f $(cat "$f")"; fi
  done; } > "$OUT/cgroup_limits.txt" 2>&1
{ id; grep CapEff /proc/self/status; } > "$OUT/permission.txt" 2>&1
# Monitoring agents (for example DCGM) can hold the performance counters
ps -eo user,comm,args | grep -i -E 'dcgm|amperf|monitor|exporter' | grep -v grep > "$OUT/monitor_procs.txt" || true
echo "saved to $OUT"
