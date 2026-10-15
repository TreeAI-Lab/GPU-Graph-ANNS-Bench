#!/bin/bash
# Final IVF-PQ runs of Figures 1 and 7: search only, k = 10, on the one configuration per dataset and
# library listed in SELECTED_CSV. The indexes are those built by run_ivf_sweep.sh and are never rebuilt.
#
# Layout:
#   $OUT_DIR/nongraph_baselines/ivf_k10/<dataset>/<algo>/nlist<N>_m<M>/
#     config.txt
#     search/nprobe_*/survey_metrics_search.json
#     curve/qps_recall.csv
#     logs/search_nprobe_*.log
#
# Usage:
#   bash experiments/nongraph_baselines/run_ivf_k10.sh                  # every row of SELECTED_CSV
#   bash experiments/nongraph_baselines/run_ivf_k10.sh sift1M mnist8M
#   DRY_RUN=1 ...
#   CLEAR_RESULT=0 ...   keep earlier results and skip nprobe points that are already done
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$ENV_BUILD/bin/python}
RESULT_ROOT=${RESULT_ROOT:-$OUT_DIR/nongraph_baselines/ivf_k10}
# The configurations of the paper; select_configs.py regenerates this file from the k = 100 sweeps.
SELECTED_CSV=${SELECTED_CSV:-$HERE/selected_configs.csv}
FAISS_BIN=${FAISS_BIN:-$BIN_DIR/faiss_ivf_pq}
CUVS_BIN=${CUVS_BIN:-$BIN_DIR/cuvs_ivf_pq}
FAISS_LIB_PATH=${FAISS_LIB_PATH:-$W/build/faiss/install/lib:$ENV_FAISS/lib}
CUVS_LIB_PATH=${CUVS_LIB_PATH:-$W/build/cuvs/install/lib:$ENV_CUVS/lib:$ENV_CUVS/targets/x86_64-linux/lib}
K=${K:-10}
NPROBES=${NPROBES:-1 2 3 4 5 6 8 10 12 14 16 20 24 28 32 40 48 56 64 80 96 112 128 160 192 224 256 320 384 448 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 3584 4096 5120 6144 8192 10240 12288 16384 20480 24576 32768}
FAISS_MAX_NPROBE=${FAISS_MAX_NPROBE:-2048}
CUVS_MAX_NPROBE=${CUVS_MAX_NPROBE:-32768}
NPROBE_NO_ANCHORS=${NPROBE_NO_ANCHORS:-0}
DRY_RUN=${DRY_RUN:-0}
CLEAR_RESULT=${CLEAR_RESULT:-1}

FAIL_COUNT=0
OK_SEARCH=0
FAIL_LOG=""
MASTER_LOG=""

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${MASTER_LOG:-/dev/null}"; }

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

make_nprobes() {
  local nlist="$1"
  local max_np="$2"
  local cap=$nlist
  if (( max_np < cap )); then
    cap=$max_np
  fi
  local np prev=-1
  local out=()
  for np in $NPROBES; do
    if (( np <= cap && np != prev )); then
      out+=("$np")
      prev=$np
    fi
  done
  if [[ "$NPROBE_NO_ANCHORS" != "1" ]]; then
    local a
    for a in $((nlist / 4)) $((nlist / 2)) $nlist; do
      if (( a >= 1 && a <= cap )); then
        local seen=0 x
        for x in "${out[@]}"; do
          if (( x == a )); then seen=1; break; fi
        done
        if (( seen == 0 )); then
          out+=("$a")
        fi
      fi
    done
  fi
  if [[ ${#out[@]} -eq 0 ]]; then
    out=(1)
    if (( cap >= 8 )); then out+=(8); fi
    if (( cap >= 64 )); then out+=(64); fi
  fi
  printf '%s\n' "${out[@]}" | sort -n | uniq
}

ensure_bins() {
  local ok=1
  if [[ ! -x "$FAISS_BIN" ]]; then
    log "ERROR: missing Faiss binary: $FAISS_BIN"
    ok=0
  fi
  if [[ ! -x "$CUVS_BIN" ]]; then
    log "ERROR: missing cuVS binary: $CUVS_BIN"
    ok=0
  fi
  [[ "$ok" == "1" ]]
}

# Rows of SELECTED_CSV, optionally only the given datasets. Prints: dataset algo nlist m config
load_selected() {
  "$PY" - "$SELECTED_CSV" "$@" <<'PY'
import csv
import sys

want = set(sys.argv[2:])
with open(sys.argv[1]) as f:
    for r in csv.DictReader(f):
        if want and r["dataset"] not in want:
            continue
        print(r["dataset"], r["algo"], r["nlist"], r["m"], r["config"])
PY
}

run_one_search() {
  local dataset="$1" algo="$2" nlist="$3" m="$4" cfg_name="$5"
  local bin index_ext max_np lib_path
  if [[ "$algo" == *cuvs* ]]; then
    bin="$CUVS_BIN"
    index_ext="cuvsindex"
    max_np="$CUVS_MAX_NPROBE"
    lib_path="$CUVS_LIB_PATH"
  else
    bin="$FAISS_BIN"
    index_ext="faissindex"
    max_np="$FAISS_MAX_NPROBE"
    lib_path="$FAISS_LIB_PATH"
  fi

  local data_dir="$DATA_DIR/$dataset"
  local metric
  metric="$(dataset_metric "$dataset")"
  local out_dir="$RESULT_ROOT/$dataset/$algo/$cfg_name"
  local search_root="$out_dir/search"
  local log_dir="$out_dir/logs"
  local curve_dir="$out_dir/curve"
  local index_path="$INDEX_DIR/$dataset/$algo/$cfg_name/index.$index_ext"

  mkdir -p "$search_root" "$log_dir" "$curve_dir" || {
    record_fail mkdir "$dataset" "$algo" "$cfg_name" "cannot mkdir output dirs"
    return 0
  }

  if [[ ! -f "$index_path" ]]; then
    record_fail index "$dataset" "$algo" "$cfg_name" "missing $index_path"
    return 0
  fi
  if [[ ! -d "$data_dir" ]]; then
    record_fail dataset "$dataset" "$algo" "$cfg_name" "missing $data_dir"
    return 0
  fi

  local nprobes_eff=()
  if ! mapfile -t nprobes_eff < <(make_nprobes "$nlist" "$max_np"); then
    record_fail nprobe_list "$dataset" "$algo" "$cfg_name" "make_nprobes failed"
    return 0
  fi
  if [[ ${#nprobes_eff[@]} -eq 0 ]]; then
    record_fail nprobe_list "$dataset" "$algo" "$cfg_name" "empty nprobe list"
    return 0
  fi

  log "== $dataset / $algo / $cfg_name  k=$K metric=$metric max_nprobe=$max_np (#nprobe=${#nprobes_eff[@]}) index=$index_path"

  cat >"$out_dir/config.txt" <<EOF
dataset=$dataset
algo=$algo
config=$cfg_name
data_dir=$data_dir
nlist=$nlist
m=$m
k=$K
metric=$metric
nprobes=${nprobes_eff[*]}
bin=$bin
index_path=$index_path
selected_csv=$SELECTED_CSV
skip_build=1
EOF

  local np
  for np in "${nprobes_eff[@]}"; do
    local sdir="$search_root/nprobe_${np}"
    local sjson="$sdir/survey_metrics_search.json"
    local slog="$log_dir/search_nprobe_${np}.log"
    mkdir -p "$sdir" || {
      record_fail search "$dataset" "$algo" "$cfg_name" "mkdir $sdir failed nprobe=$np"
      continue
    }
    if [[ "$DRY_RUN" == "1" ]]; then
      log "DRY_RUN search: $cfg_name nprobe=$np k=$K"
      continue
    fi
    # Skip nprobe points that are already complete.
    if [[ -f "$sjson" && -s "$slog" ]]; then
      if grep -q 'SURVEY METRICS\|e2e_QPS\|search_qps' "$slog" 2>/dev/null || \
         "$PY" -c "import json, sys; json.load(open(sys.argv[1]))" "$sjson" 2>/dev/null; then
        log "SKIP search (exists): $cfg_name nprobe=$np"
        OK_SEARCH=$((OK_SEARCH + 1))
        continue
      fi
    fi
    log "SEARCH $algo $dataset $cfg_name nprobe=$np k=$K (IVF_METRIC=$metric) ..."
    local rc=0
    LD_LIBRARY_PATH="$lib_path:${LD_LIBRARY_PATH:-}" IVF_METRIC="$metric" \
      "$bin" search "$data_dir" "$index_path" "$np" "$K" "$sjson" \
      >"$slog" 2>&1 || rc=$?
    if [[ $rc -ne 0 ]]; then
      record_fail search "$dataset" "$algo" "$cfg_name" "nprobe=$np rc=$rc log=$slog"
      continue
    fi
    OK_SEARCH=$((OK_SEARCH + 1))
  done

  if [[ "$DRY_RUN" != "1" ]]; then
    local rc=0
    "$PY" "$HERE/curve_csv.py" "$search_root" "$curve_dir/qps_recall.csv" \
      "$nlist" "$m" "$K" "$algo" "$cfg_name" || rc=$?
    if [[ $rc -ne 0 ]]; then
      record_fail aggregate "$dataset" "$algo" "$cfg_name" "curve_csv.py rc=$rc"
    fi
  fi
  return 0
}

write_summary() {
  local summary="$RESULT_ROOT/run_summary.txt"
  {
    echo "finished_at=$(date '+%F %T')"
    echo "k=$K"
    echo "ok_search=$OK_SEARCH"
    echo "fail_count=$FAIL_COUNT"
    echo "selected_csv=$SELECTED_CSV"
    echo "failures_log=$FAIL_LOG"
    echo "master_log=$MASTER_LOG"
    if [[ -f "$FAIL_LOG" ]]; then
      echo "---- failures ----"
      cat "$FAIL_LOG"
    fi
  } >"$summary"
  log "SUMMARY: ok_search=$OK_SEARCH fail_count=$FAIL_COUNT  file=$summary"
}

main() {
  mkdir -p "$RESULT_ROOT"
  FAIL_LOG="$RESULT_ROOT/failures.log"
  MASTER_LOG="$RESULT_ROOT/run_master.log"
  : >"$FAIL_LOG"
  {
    echo "===== ivf_k10 start $(date '+%F %T') ====="
    echo "k=$K selected_csv=$SELECTED_CSV args=$*"
  } >>"$MASTER_LOG"

  if [[ ! -f "$SELECTED_CSV" ]]; then
    record_fail setup "-" "-" "-" "missing $SELECTED_CSV"
    write_summary
    return 1
  fi
  if ! ensure_bins; then
    record_fail setup "-" "-" "-" "missing binaries"
    write_summary
    return 1
  fi

  log "RESULT_ROOT=$RESULT_ROOT"
  log "INDEX_DIR=$INDEX_DIR"
  log "DATA_DIR=$DATA_DIR"
  log "K=$K (search only; indexes from run_ivf_sweep.sh)"
  log "CLEAR_RESULT=$CLEAR_RESULT"

  local rows=()
  mapfile -t rows < <(load_selected "$@")
  if [[ ${#rows[@]} -eq 0 ]]; then
    record_fail setup "-" "-" "-" "no selected configs (filter=${*:-none})"
    write_summary
    return 1
  fi

  # Clear the result directory of every listed dataset once.
  if [[ "$CLEAR_RESULT" == "1" && "$DRY_RUN" != "1" ]]; then
    local seen_ds=""
    local line ds
    for line in "${rows[@]}"; do
      ds="${line%% *}"
      if [[ " $seen_ds " != *" $ds "* ]]; then
        log "CLEAR result: $RESULT_ROOT/$ds"
        rm -rf "${RESULT_ROOT:?}/$ds"
        seen_ds+=" $ds"
      fi
    done
  fi

  local line dataset algo nlist m cfg_name
  for line in "${rows[@]}"; do
    read -r dataset algo nlist m cfg_name <<<"$line"
    log "======== $dataset / $algo / $cfg_name ========"
    run_one_search "$dataset" "$algo" "$nlist" "$m" "$cfg_name" \
      || record_fail config "$dataset" "$algo" "$cfg_name" "run_one_search status=$?"
  done

  log "ALL DONE. Results under $RESULT_ROOT"
  write_summary
  if [[ "$FAIL_COUNT" -gt 0 ]]; then
    return 2
  fi
  return 0
}

main "$@"
