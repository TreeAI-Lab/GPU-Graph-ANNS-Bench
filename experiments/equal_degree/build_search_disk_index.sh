#!/bin/bash
# Build the DiskANN search program used for the Vamana curves (search_disk_index with per-stage
# timers and counters) and its build_disk_index, from diskann-instrumented with
# diskann_instrumented.patch (third_party/build/diskann_instrumented.sh).
#
# Usage:  bash build_search_disk_index.sh
# Output: $W/build/diskann-instrumented/apps/search_disk_index, build_disk_index
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

bash "$BENCH/third_party/build/diskann_instrumented.sh"
