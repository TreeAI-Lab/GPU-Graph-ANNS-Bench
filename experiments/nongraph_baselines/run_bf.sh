#!/bin/bash
# cuVS brute force (exact k-NN on the GPU): one point per dataset in Figures 1 and 7.
# glove200 and nytimes-full use cosine (IVF_METRIC=cosine: normalize and use inner product); the rest L2.
#
# Layout:
#   $OUT_DIR/nongraph_baselines/bf/<dataset>/cuvs_bf/{build,search,logs}/
#   $INDEX_DIR/<dataset>/cuvs_bf/index.cuvsindex   (holds the full base set)
#   $OUT_DIR/nongraph_baselines/bf/{failures.log,run_summary.txt,run_master.log}
#
# k: 100 for every dataset except mnist8M (10), as in the paper's runs. Exact search returns the
# same top 10 for both values.
#
# Usage:
#   bash experiments/nongraph_baselines/run_bf.sh                 # all datasets
#   bash experiments/nongraph_baselines/run_bf.sh sift1M gist
#   SKIP_BUILD=1 ...   reuse an existing index
#   CLEAR_RESULT=0 ... keep earlier metrics of the listed datasets
#   DRY_RUN=1 ...
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

RESULT_ROOT=${RESULT_ROOT:-$OUT_DIR/nongraph_baselines/bf}
BF_BIN=${BF_BIN:-$BIN_DIR/cuvs_brute_force}
CUVS_LIB_PATH=${CUVS_LIB_PATH:-$W/build/cuvs/install/lib:$ENV_CUVS/lib:$ENV_CUVS/targets/x86_64-linux/lib}
ALGO="cuvs_bf"
SKIP_BUILD=${SKIP_BUILD:-0}
DRY_RUN=${DRY_RUN:-0}
CLEAR_RESULT=${CLEAR_RESULT:-1}

DATASETS_DEFAULT=(sift1M deep1M-annb deep10M-annb deep100M gist glove200 nytimes-full mnist8M)

FAIL_COUNT=0
OK_BUILD=0
OK_SEARCH=0
FAIL_LOG=""
MASTER_LOG=""

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${MASTER_LOG:-/dev/null}"; }

bf_k() {
  case "$1" in
    mnist8M) echo 10 ;;
    *) echo 100 ;;
  esac
}

clear_dataset_result() {
  local ds="$1"
  local dir="$RESULT_ROOT/$ds"
  if [[ -e "$dir" || -L "$dir" ]]; then
    log "CLEAR result: $dir"
    rm -rf "$dir"
  else
    log "CLEAR result: $dir (absent, skip)"
  fi
}

record_fail() {
  local stage="$1" dataset="$2" algo="$3" config="$4"
  shift 4
  local detail="$*"
  FAIL_COUNT=$((FAIL_COUNT + 1))
  local line
  line="$(printf '%s\tstage=%s\tdataset=%s\talgo=%s\tconfig=%s\t%s' \
    "$(date '+%F %T')" "$stage" "$dataset" "$algo" "$config" "$detail")"
  printf '%s\n' "$line" | tee -a "$FAIL_LOG"
  log "FAIL#$FAIL_COUNT [$stage] $dataset/$algo/$config: $detail"
}

dataset_metric() {
  case "$1" in
    glove200|nytimes-full) echo cosine ;;
    *) echo l2 ;;
  esac
}

run_one_dataset() {
  local dataset="$1"
  local data_dir="$DATA_DIR/$dataset"
  local metric
  metric="$(dataset_metric "$dataset")"
  local k
  k="$(bf_k "$dataset")"

  local out_dir="$RESULT_ROOT/$dataset/$ALGO"
  local build_dir="$out_dir/build"
  local search_dir="$out_dir/search"
  local log_dir="$out_dir/logs"
  local index_dir="$INDEX_DIR/$dataset/$ALGO"
  local index_path="$index_dir/index.cuvsindex"
  local build_json="$build_dir/survey_metrics_build.json"
  local search_json="$search_dir/survey_metrics_search.json"
  local build_log="$log_dir/build.log"
  local search_log="$log_dir/search.log"

  if ! mkdir -p "$build_dir" "$search_dir" "$log_dir" "$index_dir"; then
    record_fail mkdir "$dataset" "$ALGO" "-" "cannot mkdir output or index dirs"
    return 0
  fi

  log "== $dataset / $ALGO  k=$k metric=$metric"

  cat >"$out_dir/config.txt" <<EOF || true
dataset=$dataset
algo=$ALGO
data_dir=$data_dir
k=$k
metric=$metric
bin=$BF_BIN
index_path=$index_path
EOF

  local have_index=0
  if [[ -f "$index_path" ]]; then
    have_index=1
  fi

  if [[ "$SKIP_BUILD" == "1" && "$have_index" == "1" ]]; then
    log "SKIP build: $index_path"
  elif [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN build: IVF_METRIC=$metric $BF_BIN build $data_dir $index_path $build_json"
    have_index=1
  else
    log "BUILD $ALGO $dataset (IVF_METRIC=$metric) ..."
    local rc=0
    LD_LIBRARY_PATH="$CUVS_LIB_PATH:${LD_LIBRARY_PATH:-}" IVF_METRIC="$metric" \
      "$BF_BIN" build "$data_dir" "$index_path" "$build_json" \
      >"$build_log" 2>&1 || rc=$?
    if [[ $rc -ne 0 ]]; then
      record_fail build "$dataset" "$ALGO" "-" "rc=$rc log=$build_log"
      if [[ ! -f "$index_path" ]]; then
        log "SKIP search (no index after failed build): $dataset"
        return 0
      fi
      log "WARN: build failed but the index file exists; still trying search"
      have_index=1
    else
      OK_BUILD=$((OK_BUILD + 1))
      have_index=1
      log "BUILD ok: $index_path"
    fi
  fi

  if [[ "$have_index" != "1" && "$DRY_RUN" != "1" ]]; then
    record_fail build "$dataset" "$ALGO" "-" "missing index $index_path"
    return 0
  fi

  if [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN search: IVF_METRIC=$metric $BF_BIN search $data_dir $index_path $k $search_json"
    return 0
  fi

  log "SEARCH $ALGO $dataset k=$k (IVF_METRIC=$metric) ..."
  local rc=0
  LD_LIBRARY_PATH="$CUVS_LIB_PATH:${LD_LIBRARY_PATH:-}" IVF_METRIC="$metric" \
    "$BF_BIN" search "$data_dir" "$index_path" "$k" "$search_json" \
    >"$search_log" 2>&1 || rc=$?
  if [[ $rc -ne 0 ]]; then
    record_fail search "$dataset" "$ALGO" "-" "k=$k rc=$rc log=$search_log"
    return 0
  fi
  OK_SEARCH=$((OK_SEARCH + 1))
  return 0
}

write_summary() {
  local summary="$RESULT_ROOT/run_summary.txt"
  {
    echo "finished_at=$(date '+%F %T')"
    echo "ok_build=$OK_BUILD"
    echo "ok_search=$OK_SEARCH"
    echo "fail_count=$FAIL_COUNT"
    echo "failures_log=$FAIL_LOG"
    echo "master_log=$MASTER_LOG"
    if [[ -f "$FAIL_LOG" ]]; then
      echo "---- failures ----"
      cat "$FAIL_LOG"
    fi
  } >"$summary"
  log "SUMMARY: ok_build=$OK_BUILD ok_search=$OK_SEARCH fail_count=$FAIL_COUNT"
  log "SUMMARY file: $summary"
}

main() {
  mkdir -p "$RESULT_ROOT"
  FAIL_LOG="$RESULT_ROOT/failures.log"
  MASTER_LOG="$RESULT_ROOT/run_master.log"
  : >"$FAIL_LOG"
  {
    echo "===== run start $(date '+%F %T') ====="
    echo "datasets_args=$*"
  } >>"$MASTER_LOG"

  if [[ ! -x "$BF_BIN" ]]; then
    record_fail setup "-" "$ALGO" "-" "missing executable $BF_BIN"
    write_summary
    return 1
  fi

  local datasets=("$@")
  if [[ ${#datasets[@]} -eq 0 ]]; then
    datasets=("${DATASETS_DEFAULT[@]}")
  fi

  log "RESULT_ROOT=$RESULT_ROOT"
  log "INDEX_DIR=$INDEX_DIR"
  log "DATA_DIR=$DATA_DIR"
  log "datasets=${datasets[*]}"

  local ds
  for ds in "${datasets[@]}"; do
    log "======== dataset=$ds ========"
    if [[ "$CLEAR_RESULT" == "1" && "$DRY_RUN" != "1" ]]; then
      clear_dataset_result "$ds"
    fi
    if [[ ! -f "$DATA_DIR/$ds/base.fbin" ]]; then
      record_fail dataset "$ds" "$ALGO" "-" "missing $DATA_DIR/$ds/base.fbin"
      continue
    fi
    run_one_dataset "$ds" || record_fail dataset "$ds" "$ALGO" "-" "run_one_dataset status=$?"
  done

  log "ALL DONE. Results under $RESULT_ROOT"
  write_summary
  if [[ "$FAIL_COUNT" -gt 0 ]]; then
    return 2
  fi
  return 0
}

main "$@"
