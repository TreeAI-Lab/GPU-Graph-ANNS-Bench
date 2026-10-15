#!/bin/bash
# Build the DiskANN search program used for the Vamana curves (search_disk_index of the original
# runs, with per-stage timers and counters) and its build_disk_index, from diskann-orig with
# diskann_orig.patch (third_party/build/diskann_orig.sh).
#
# Usage:  bash build_search_disk_index.sh
# Output: $W/build/diskann-orig/apps/search_disk_index, build_disk_index
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

bash "$BENCH/third_party/build/diskann_orig.sh"
