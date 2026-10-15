#!/bin/bash
# Exact top-10 of the 5,000,000 serving queries on one dataset (cuVS brute force, Figure 6).
# Usage: bash experiments/loading/gt_5m.sh <deep1M|deep10M-annb|deep100M>
# Output: $DATA_DIR/<dataset>/gt.5M.k10.ibin (and its .log). Skipped if the log says it is complete.
#
# The first 10,000 rows of the query file are the official Deep1B queries. For deep1M and deep100M,
# whose gt.ibin is computed for those queries, the first 10,000 rows are checked against gt.ibin.
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

HERE=$(cd "$(dirname "$0")" && pwd)

DS=${1:?usage: gt_5m.sh <dataset>}
QUERY=${QUERY:-$DATA_DIR/deep100M/query.5M.fbin}
BASE=${BASE:-$DATA_DIR/$DS/base.fbin}
GT=${GT:-$DATA_DIR/$DS/gt.5M.k10.ibin}
BF_INDEX=${BF_INDEX:-$INDEX_DIR/$DS/cuvs_bf/gt_5m_index.cuvsindex}
K=10
BATCH=10000
PY=$ENV_CAGRA/bin/python
LOG=${GT%.ibin}.log

[ -f "$QUERY" ] || { echo "missing $QUERY (datasets/sample_deep1b_queries.py)"; exit 2; }
[ -f "$BASE" ] || { echo "missing $BASE"; exit 2; }

if ! grep -q "DONE wrote" "$LOG" 2>/dev/null; then
  echo "===== GT $DS start $(date '+%F %T') ====="
  "$PY" "$HERE/gt_5m.py" --base "$BASE" --query "$QUERY" --out "$GT" \
    --k "$K" --batch "$BATCH" --index "$BF_INDEX"
  echo "===== GT $DS done $(date '+%F %T') ====="
fi

case "$DS" in
  deep1M|deep100M)
    echo "===== check the first 10,000 rows against $DATA_DIR/$DS/gt.ibin ====="
    "$PY" - "$GT" "$DATA_DIR/$DS/gt.ibin" "$K" <<'PY'
import sys

import numpy as np


def load(p: str) -> np.ndarray:
    n, k = np.fromfile(p, dtype=np.int32, count=2)
    return np.fromfile(p, dtype=np.int32, offset=8).reshape(int(n), int(k))


gt, ref, k = load(sys.argv[1]), load(sys.argv[2]), int(sys.argv[3])
a, b = gt[:10000, :k], ref[:10000, :k]
hits = sum(len(set(a[i].tolist()) & set(b[i].tolist())) for i in range(10000))
r = hits / (10000 * k)
print(f"official_overlap_recall@{k} {r:.6f}")
if r < 0.99:
    raise SystemExit("ground truth does not match gt.ibin on the first 10,000 queries")
PY
    ;;
esac
