#!/bin/bash
# Recall@10 and QPS of Table 10: 16 groups (GIST and DEEP1M; SONG, GGNN, GANNS, CAGRA; FP32 and FP16) at the
# operating point without a profiler. Each program run searches the operating point once as an untimed warm-up and
# then three times timed. QPS = nq / kernel_ms of a timed search, the kernel-only measure of Figure 7; Table 10
# reports the mean QPS of the three timed searches.
# REPS (default 1) program runs per group. With more than one, the runs form the outer loop, so the runs of one
# group are spread over the session, and qps.tsv pools the timed searches of all runs of a group.
# Runs whose sweep.tsv exists are skipped, so an interrupted session can be resumed.
# PRE_SLEEP_S (default 2) sleeps before each run so that every run starts from an idle GPU.
# usage: bash qps_repeat.sh [REPS, default 1]
# output: $OUT/<dataset>_<system>_<precision>/rep<i>/ (sweep.tsv, it_batches.tsv, logs) and $OUT/qps.tsv
#   (OUT default $OUT_DIR/profiling/table10_qps)
#   qps.tsv: dataset system precision reps recall@10 qps_mean qps_std qps_median qps_min qps_max kernel_ms_mean,
#   over the timed searches of a group (reps = number of timed searches)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
REPS=${1:-1}
OUT=${OUT:-$OUT_DIR/profiling/table10_qps}
export PRE_SLEEP_S=${PRE_SLEEP_S:-2}
mkdir -p "$OUT"
for i in $(seq 1 "$REPS"); do
  for ds in gist deep1M; do for sys in song ggnn ganns cagra; do for prec in fp32 fp16; do
    d=$OUT/${ds}_${sys}_${prec}/rep$i
    [[ -f $d/sweep.tsv ]] && continue
    mkdir -p "$d"
    bash "$HERE/run_point.sh" "$ds" "$sys" "$d" "$prec" > "$d/run.log" 2>&1 || echo "FAIL $ds $sys $prec rep$i"
  done; done; done
  echo "rep $i done $(date -Is)"
done

# Summary from the timed searches in it_batches.tsv (phase timed1, timed2, ...; the warm-up rows are left out).
# Columns are found by header name (the columns differ between systems). Per group: number of timed searches,
# recall (the same in every search), QPS mean (Table 10), sample standard deviation, median, minimum, maximum,
# mean kernel time.
{
  printf 'dataset\tsystem\tprecision\treps\trecall@10\tqps_mean\tqps_std\tqps_median\tqps_min\tqps_max\tkernel_ms_mean\n'
  for g in "$OUT"/*_*_fp*/; do
    name=$(basename "$g"); IFS=_ read -r ds sys prec <<< "$name"
    awk -F'\t' -v ds="$ds" -v sys="$sys" -v prec="$prec" '
      FNR == 1 { for (c = 1; c <= NF; c++) col[$c] = c; next }
      $(col["phase"]) !~ /^timed/ { next }
      { k[++n] = $(col["kernel_ms"]); q[n] = $(col["nq"]) / $(col["kernel_ms"]) * 1000; sq += q[n]; sk += k[n]; r[$(col["recall@10"])] = 1 }
      END {
        if (n == 0) exit
        mean = sq / n; for (i = 1; i <= n; i++) v += (q[i] - mean) ^ 2; sd = (n > 1) ? sqrt(v / (n - 1)) : 0
        for (i = 2; i <= n; i++) { x = q[i]; for (j = i - 1; j >= 1 && q[j] > x; j--) q[j + 1] = q[j]; q[j + 1] = x }
        m = int((n + 1) / 2)
        rec = ""; for (x in r) rec = rec (rec == "" ? "" : ",") x
        printf "%s\t%s\t%s\t%d\t%s\t%.0f\t%.0f\t%.0f\t%.0f\t%.0f\t%.4f\n", ds, sys, prec, n, rec, mean, sd, q[m], q[1], q[n], sk / n
      }' "$g"rep*/it_batches.tsv
  done
} > "$OUT/qps.tsv"
cat "$OUT/qps.tsv"
