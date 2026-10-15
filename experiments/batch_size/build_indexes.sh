#!/bin/bash
# Indexes of the batch-size runs (Figure 8), all on SIFT1M, built with experiments/qps_recall/build_graphs.sh:
#   CAGRA, GGNN, GANNS, SONG and PilotANN: the SIFT1M indexes of the QPS-recall runs
#   BANG: a DiskANN disk index with R = 64 (the QPS-recall runs use R = 32 on SIFT1M), L 100, M 16, B 4, QD 32
# Usage: bash experiments/batch_size/build_indexes.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
QR=$BENCH/experiments/qps_recall

SYSTEMS="cagra ggnn ganns song pilotann" bash "$QR/build_graphs.sh" sift1M
SYSTEMS=bang BANG_CFG="64 100 16 4 32" bash "$QR/build_graphs.sh" sift1M
