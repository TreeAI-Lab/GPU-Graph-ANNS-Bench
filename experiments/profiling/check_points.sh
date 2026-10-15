#!/bin/bash
# Check the programs on a new machine: one FP32 and one FP16 run of SONG, GGNN, GANNS and CAGRA at the operating
# point on GIST and DEEP1M, printed next to the reference recall of points.tsv. The FP32 recall should equal
# recall_ref (up to the rounding in points.tsv); the expected FP16 recall is listed in the README.
# usage: bash check_points.sh
# output: $OUT_DIR/profiling/check/<dataset>_<system>_<precision>/ and a table on stdout
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
OUT=$OUT_DIR/profiling/check
printf 'dataset\tsystem\tprecision\tit\trecall@10\trecall_ref\tkernel_ms\n'
for ds in gist deep1M; do
  for sys in song ggnn ganns cagra; do
    for prec in fp32 fp16; do
      d=$OUT/${ds}_${sys}_${prec}
      mkdir -p "$d"
      if ! bash "$HERE/run_point.sh" "$ds" "$sys" "$d" "$prec" > "$d/run.log" 2>&1; then
        printf '%s\t%s\t%s\tFAILED, see %s\n' "$ds" "$sys" "$prec" "$d/run.log"; continue
      fi
      ref=$(awk -F'\t' -v d="$ds" -v s="$sys" '$1==d && $2==s {print $4}' "$HERE/points.tsv")
      awk -F'\t' -v t="$(printf '%s\t%s\t%s' "$ds" "$sys" "$prec")" -v ref="$ref" 'NR==1{for(i=1;i<=NF;i++)c[$i]=i} NR==2{printf "%s\t%s\t%s\t%s\t%s\n", t, $c["it"], $c["recall@10"], ref, $c["kernel_ms"]}' "$d/sweep.tsv"
    done
  done
done
