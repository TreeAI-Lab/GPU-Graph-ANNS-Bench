#!/bin/bash
# Indexes of the batch-size runs (Figure 8), all on SIFT1M, built with experiments/qps_recall/build_graphs.sh:
#   CAGRA, GGNN, GANNS, SONG and PilotANN: the SIFT1M indexes of the QPS-recall runs
#   BANG: the SIFT1M DiskANN disk index of the QPS-recall runs, R = 32, L 60, M 80, B 80, QD 32
# Usage: bash experiments/batch_size/build_indexes.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
QR=$BENCH/experiments/qps_recall

bash "$QR/build_graphs.sh" sift1M
