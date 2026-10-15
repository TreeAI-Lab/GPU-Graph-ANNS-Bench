#!/bin/bash
# Nsight Compute for all 18 groups: SONG, GGNN, GANNS and CAGRA in FP32 and FP16, and BANG in FP32, on GIST and
# DEEP1M. Settings as in profile_ncu.sh (clocks not locked, application replay, sleep before each pass).
# usage: bash profile_all.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
# BANG launches its main kernel once per iteration: launches per search at the operating point of points.tsv
# (from find_kernel.sh). run_point.sh runs four searches (one warm-up, three timed), so the first timed search
# starts after 1 x launches; it is the search that is profiled.
BANG_LAUNCHES_DEEP1M=${BANG_LAUNCHES_DEEP1M:-87}
BANG_LAUNCHES_GIST=${BANG_LAUNCHES_GIST:-78}
for ds in gist deep1M; do
  for sys in song ggnn ganns cagra; do
    for prec in fp32 fp16; do
      echo "-- $ds $sys $prec $(date -Is)"
      PREC=$prec bash "$HERE/profile_ncu.sh" "$ds" "$sys" || echo "FAILED $ds $sys $prec"
    done
  done
done
echo "-- deep1M bang $(date -Is)"
NCU_LAUNCH_SKIP=$((1 * BANG_LAUNCHES_DEEP1M)) NCU_LAUNCH_COUNT=$BANG_LAUNCHES_DEEP1M bash "$HERE/profile_ncu.sh" deep1M bang || echo "FAILED deep1M bang"
echo "-- gist bang $(date -Is)"
NCU_LAUNCH_SKIP=$((1 * BANG_LAUNCHES_GIST)) NCU_LAUNCH_COUNT=$BANG_LAUNCHES_GIST bash "$HERE/profile_ncu.sh" gist bang || echo "FAILED gist bang"
