#!/bin/bash
# Draw every figure from the result CSVs. Usage: bash plots/plot_all.sh [--results DIR] [--out DIR]
set -euo pipefail
cd "$(dirname "$0")"
for script in fig*.py; do
  python3 "$script" "$@"
done
