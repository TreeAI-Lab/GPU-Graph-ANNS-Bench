#!/bin/bash
# Empty the host page cache before a search process starts, so that no earlier run leaves the base vectors or the
# index in memory. With write access to /proc/sys/vm/drop_caches the whole page cache is dropped; otherwise the
# cached pages of every file under the given paths (default: $DATA_DIR and $INDEX_DIR) are evicted with
# dd iflag=nocache. Device memory and the GPU caches start empty because every search runs in a new process.
#
# Usage:  bash src/common/clear_caches.sh [path ...]
# Called through clear_caches in env.sh; CLEAR_CACHES=0 skips it.
set -euo pipefail
sync
if [[ -w /proc/sys/vm/drop_caches ]]; then
  echo 3 > /proc/sys/vm/drop_caches
  exit 0
fi
PATHS=("$@")
[ ${#PATHS[@]} -gt 0 ] || PATHS=("${DATA_DIR:?}" "${INDEX_DIR:?}")
for p in "${PATHS[@]}"; do
  [[ -e $p ]] || continue
  find -L "$p" -type f -exec dd if={} iflag=nocache count=0 status=none \;
done
