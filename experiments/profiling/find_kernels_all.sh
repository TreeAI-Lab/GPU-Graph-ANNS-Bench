#!/bin/bash
# Run find_kernel.sh for every group of Tables 9 and 10: five systems in FP32 and four in FP16, on GIST and DEEP1M.
# usage: bash find_kernels_all.sh
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
TIMEOUT_S=${TIMEOUT_S:-1800}
for prec in fp32 fp16; do
  for ds in deep1M gist; do
    for sys in song ggnn ganns cagra bang; do
      [[ $prec == fp16 && $sys == bang ]] && continue
      echo "######## $ds $sys $prec $(date +%T)"
      PREC=$prec timeout "$TIMEOUT_S" bash "$HERE/find_kernel.sh" "$ds" "$sys" || echo "FAILED $ds $sys $prec"
    done
  done
done
echo "ALL DONE $(date +%T)"
