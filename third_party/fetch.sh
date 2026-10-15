#!/bin/bash
# Fetch the third-party sources listed in versions.tsv into $W/src/<name>: fetch the pinned
# commit, check it out detached, and apply patches/<patch> when the patch column is set.
# Running it again is safe: a checkout at the pinned commit is kept, and a patch that is
# already applied is skipped.
#
# Usage: bash third_party/fetch.sh [name ...]      (default: every row of versions.tsv)
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../env.sh"

VERSIONS=${VERSIONS:-$HERE/versions.tsv}
PATCH_DIR=${PATCH_DIR:-$HERE/patches}
mkdir -p "$W/src"

fetch_one() {
  local name=$1 url=$2 commit=$3 patch=$4
  local dir=$W/src/$name
  if [ ! -d "$dir/.git" ]; then
    git init -q "$dir"
    git -C "$dir" remote add origin "$url"
  fi
  local head
  head=$(git -C "$dir" rev-parse -q --verify HEAD 2>/dev/null || true)
  if [ "$head" != "$commit" ]; then
    # Fetch only the pinned commit; fall back to a full fetch if the server refuses.
    git -C "$dir" fetch -q --depth 1 origin "$commit" || git -C "$dir" fetch -q origin
    git -C "$dir" -c advice.detachedHead=false checkout -q --detach "$commit"
  fi
  if [ -n "$patch" ]; then
    local p=$PATCH_DIR/$patch
    if git -C "$dir" apply --reverse --check "$p" 2>/dev/null; then
      echo "$name: $patch already applied"
    else
      git -C "$dir" apply "$p"
      echo "$name: applied $patch"
    fi
  fi
  echo "$name $(git -C "$dir" rev-parse HEAD)"
}

# versions.tsv columns: name, repository, commit, patch (may be empty), used_for.
# Tabs are converted to '|' so that an empty patch column stays an empty field.
ROWS=$(tail -n +2 "$VERSIONS" | tr '\t' '|' | grep -v '^$')
row_of() { printf '%s\n' "$ROWS" | awk -F'|' -v n="$1" '$1 == n'; }

[ $# -gt 0 ] || set -- $(printf '%s\n' "$ROWS" | cut -d'|' -f1)
for name in "$@"; do
  [ -n "$(row_of "$name")" ] || { echo "unknown name: $name (see $VERSIONS)" >&2; exit 1; }
done
for name in "$@"; do
  IFS='|' read -r _ url commit patch _ <<< "$(row_of "$name")"
  fetch_one "$name" "$url" "$commit" "$patch"
done
