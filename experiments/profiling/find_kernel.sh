#!/bin/bash
# Find the main search kernel at the operating point with Nsight Systems (CUDA trace only, no performance
# counters needed). The kernel_regex column of points.tsv holds the result.
# usage: [PREC=fp16] find_kernel.sh <gist|deep1M> <system>
# output: $OUT_DIR/profiling/<dataset>_<system>[_fp16]/nsys/kern_sum.csv (kernels sorted by total time);
#   prints the top five kernels and the recall of the run
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
DS=${1:?dataset}
SYS=${2:?system}
NSYS=${NSYS:-nsys}
PYTHON=${PYTHON:-python3}
PREC=${PREC:-fp32}
TAG=${DS}_${SYS}; [[ "$PREC" == fp16 ]] && TAG=${TAG}_fp16
OUT=$OUT_DIR/profiling/$TAG/nsys
mkdir -p "$OUT"
"$NSYS" --version | tee "$OUT/nsys_version.txt"
"$NSYS" profile --trace=cuda --sample=none --cpuctxsw=none --force-overwrite=true \
  -o "$OUT/trace" bash "$HERE/run_point.sh" "$DS" "$SYS" "$OUT/run" "$PREC" > "$OUT/profile.log" 2>&1
"$NSYS" stats --report cuda_gpu_kern_sum --format csv --force-export=true \
  -o "$OUT/kern" "$OUT/trace.nsys-rep" > "$OUT/stats.log" 2>&1
mv "$OUT"/kern_cuda_gpu_kern_sum.csv "$OUT/kern_sum.csv"
echo "== $DS $SYS top kernels (Time %, total ns, instances, name)"
"$PYTHON" "$HERE/show_kernels.py" "$OUT/kern_sum.csv" 5
grep -h "recall" "$OUT/run/sweep.tsv" >/dev/null 2>&1 && tail -1 "$OUT/run/sweep.tsv"
