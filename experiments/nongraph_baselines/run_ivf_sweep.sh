#!/bin/bash
# IVF-PQ in Faiss and cuVS: build several (nlist, m) indexes per dataset and sweep nprobe with k = 100.
# These sweeps are the candidates from which select_configs.py picks one configuration per dataset
# and library; the final k = 10 runs of Figures 1 and 7 are done by run_ivf_k10.sh.
#
# A failed build or search is recorded in failures.log and skipped; the script continues.
#
# Layout:
#   $OUT_DIR/nongraph_baselines/ivf_sweep/<dataset>/<algo>/nlist<N>_m<M>/{build,search,curve,logs}
#   $INDEX_DIR/<dataset>/<algo>/nlist<N>_m<M>/index.{faissindex,cuvsindex}
#   $OUT_DIR/nongraph_baselines/ivf_sweep/{failures.log,run_summary.txt,run_master.log}
#
# Usage:
#   bash experiments/nongraph_baselines/run_ivf_sweep.sh                 # all datasets
#   bash experiments/nongraph_baselines/run_ivf_sweep.sh sift1M gist
#   SKIP_BUILD=1 bash experiments/nongraph_baselines/run_ivf_sweep.sh sift1M
#   BUILD_CONFIGS="4096:16 8192:16" bash experiments/nongraph_baselines/run_ivf_sweep.sh sift1M
#   CLEAR_RESULT=0 ...   keep earlier metrics of the listed datasets (indexes are never deleted)
#   DRY_RUN=1 ...        print what would run
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

HERE=$(cd "$(dirname "$0")" && pwd)
PY=${PY:-$ENV_BUILD/bin/python}
RESULT_ROOT=${RESULT_ROOT:-$OUT_DIR/nongraph_baselines/ivf_sweep}
FAISS_BIN=${FAISS_BIN:-$BIN_DIR/faiss_ivf_pq}
CUVS_BIN=${CUVS_BIN:-$BIN_DIR/cuvs_ivf_pq}
FAISS_LIB_PATH=${FAISS_LIB_PATH:-$W/build/faiss/install/lib:$ENV_FAISS/lib}
CUVS_LIB_PATH=${CUVS_LIB_PATH:-$W/build/cuvs/install/lib:$ENV_CUVS/lib:$ENV_CUVS/targets/x86_64-linux/lib}
K=${K:-100}
BUILD_CONFIGS=${BUILD_CONFIGS:-}
NPROBES=${NPROBES:-1 2 3 4 5 6 8 10 12 14 16 20 24 28 32 40 48 56 64 80 96 112 128 160 192 224 256 320 384 448 512 640 768 896 1024 1280 1536 1792 2048 2560 3072 3584 4096 5120 6144 8192 10240 12288 16384 20480 24576 32768}
# Faiss GPU IVF rejects nprobe > 2048; cuVS accepts larger values.
FAISS_MAX_NPROBE=${FAISS_MAX_NPROBE:-2048}
CUVS_MAX_NPROBE=${CUVS_MAX_NPROBE:-32768}
# 1: use only the NPROBES list, without the nlist/4, nlist/2 and nlist anchors.
NPROBE_NO_ANCHORS=${NPROBE_NO_ANCHORS:-0}
SKIP_BUILD=${SKIP_BUILD:-0}
DRY_RUN=${DRY_RUN:-0}
# 1: delete RESULT_ROOT/<dataset> before running that dataset (indexes under INDEX_DIR are kept).
CLEAR_RESULT=${CLEAR_RESULT:-1}

DATASETS_DEFAULT=(sift1M deep1M-annb deep10M-annb deep100M gist glove200 nytimes-full mnist8M)

FAIL_COUNT=0
OK_BUILD=0
OK_SEARCH=0
FAIL_LOG=""
MASTER_LOG=""

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${MASTER_LOG:-/dev/null}"; }

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

# Record a failure and continue. Args: stage dataset algo config detail...
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

# Prints "dim n" from the fbin header.
read_dim_nb() {
  local hdr
  hdr=$(od -An -t d4 -N 8 "$1")
  # shellcheck disable=SC2086
  set -- $hdr
  echo "$2 $1"
}

# Prints "nlist m" per line; warnings go to stderr so that callers can read stdout.
list_build_configs() {
  local dataset="$1"
  local dim="$2"

  if [[ -n "$BUILD_CONFIGS" ]]; then
    local pair nlist m
    for pair in $BUILD_CONFIGS; do
      nlist="${pair%%:*}"
      m="${pair##*:}"
      if (( dim % m != 0 )); then
        log "WARN: skip $pair (m=$m does not divide dim=$dim)" >&2
        continue
      fi
      echo "$nlist $m"
    done
    return 0
  fi

  local configs=()
  case "$dataset" in
    sift1M)
      configs=(
        "2048 8" "2048 16"
        "4096 8" "4096 16" "4096 32"
        "8192 16" "16384 16"
      )
      ;;
    deep1M-annb)
      configs=(
        "2048 8" "2048 16"
        "4096 8" "4096 12" "4096 16"
        "8192 16" "16384 16"
      )
      ;;
    deep10M-annb)
      configs=(
        "8192 16"
        "16384 8" "16384 16"
        "32768 16"
      )
      ;;
    mnist8M)
      # n = 8.09M, d = 784: nlist a few times sqrt(n); m divides 784 and is supported by Faiss GPU PQ
      configs=(
        "8192 16" "8192 28"
        "16384 8" "16384 16" "16384 28"
        "32768 16" "32768 28"
      )
      ;;
    deep100M)
      # d = 96: nlist from about sqrt(n) up; m in {8, 12, 16} (divides 96, supported by Faiss GPU PQ)
      configs=(
        "32768 8" "32768 16"
        "65536 8" "65536 12" "65536 16"
        "131072 16"
      )
      ;;
    gist)
      configs=(
        "2048 16" "2048 32"
        "4096 16" "4096 32" "4096 48"
        "8192 32"
      )
      ;;
    glove200)
      # d = 200: m in {8, 20, 40}, the divisors of 200 that Faiss GPU PQ supports
      configs=(
        "2048 8" "2048 20"
        "4096 8" "4096 20" "4096 40"
        "8192 8" "8192 20"
      )
      ;;
    nytimes-full)
      configs=(
        "1024 8" "1024 16"
        "2048 8" "2048 16" "2048 32"
        "4096 16" "4096 32"
      )
      ;;
    *)
      configs=("4096 16")
      ;;
  esac

  local c nlist m
  for c in "${configs[@]}"; do
    nlist="${c%% *}"
    m="${c##* }"
    if (( dim % m != 0 )); then
      log "WARN: skip nlist=$nlist m=$m (does not divide dim=$dim)" >&2
      continue
    fi
    echo "$nlist $m"
  done
}

# Angular datasets: normalize and use inner product inside the programs.
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

# One (dataset, algo, nlist, m): build and nprobe sweep. Never aborts the batch.
run_one_config() {
  local dataset="$1"
  local algo="$2"
  local bin="$3"
  local index_ext="$4"
  local nlist="$5"
  local m="$6"
  local dim="$7"
  local nb="$8"
  local lib_path="$9"

  local data_dir="$DATA_DIR/$dataset"
  local metric
  metric="$(dataset_metric "$dataset")"
  local cfg_name="nlist${nlist}_m${m}"
  local out_dir="$RESULT_ROOT/$dataset/$algo/$cfg_name"
  local build_dir="$out_dir/build"
  local search_root="$out_dir/search"
  local log_dir="$out_dir/logs"
  local curve_dir="$out_dir/curve"
  local index_dir="$INDEX_DIR/$dataset/$algo/$cfg_name"
  local index_path="$index_dir/index.$index_ext"
  local build_json="$build_dir/survey_metrics_build.json"
  local build_log="$log_dir/build.log"
  local cfg_fail_log="$log_dir/failures.txt"

  if ! mkdir -p "$build_dir" "$search_root" "$log_dir" "$curve_dir" "$index_dir"; then
    record_fail mkdir "$dataset" "$algo" "$cfg_name" "cannot mkdir output or index dirs"
    return 0
  fi
  : >"$cfg_fail_log"

  local nprobes_eff=()
  local max_np="$FAISS_MAX_NPROBE"
  if [[ "$algo" == *cuvs* ]]; then
    max_np="$CUVS_MAX_NPROBE"
  fi
  if ! mapfile -t nprobes_eff < <(make_nprobes "$nlist" "$max_np"); then
    record_fail nprobe_list "$dataset" "$algo" "$cfg_name" "make_nprobes failed"
    return 0
  fi
  if [[ ${#nprobes_eff[@]} -eq 0 ]]; then
    record_fail nprobe_list "$dataset" "$algo" "$cfg_name" "empty nprobe list"
    return 0
  fi

  log "== $dataset / $algo / $cfg_name  nb=$nb d=$dim k=$K metric=$metric max_nprobe=$max_np (#nprobe=${#nprobes_eff[@]})"

  cat >"$out_dir/config.txt" <<EOF || true
dataset=$dataset
algo=$algo
config=$cfg_name
data_dir=$data_dir
nb=$nb
dim=$dim
nlist=$nlist
m=$m
k=$K
metric=$metric
nprobes=${nprobes_eff[*]}
bin=$bin
index_path=$index_path
EOF

  local have_index=0
  if [[ -f "$index_path" ]]; then
    have_index=1
  fi

  # ---- build ----
  if [[ "$SKIP_BUILD" == "1" && "$have_index" == "1" ]]; then
    log "SKIP build: $index_path"
  elif [[ "$DRY_RUN" == "1" ]]; then
    log "DRY_RUN build: IVF_METRIC=$metric $bin build $data_dir $nlist $m $index_path"
    have_index=1
  else
    log "BUILD $algo $dataset $cfg_name (IVF_METRIC=$metric) ..."
    local rc=0
    LD_LIBRARY_PATH="$lib_path:${LD_LIBRARY_PATH:-}" IVF_METRIC="$metric" \
      "$bin" build "$data_dir" "$nlist" "$m" "$index_path" "$build_json" \
      >"$build_log" 2>&1 || rc=$?
    if [[ $rc -ne 0 ]]; then
      record_fail build "$dataset" "$algo" "$cfg_name" "rc=$rc log=$build_log"
      echo "build rc=$rc" >>"$cfg_fail_log"
      if [[ ! -f "$index_path" ]]; then
        log "SKIP search (no index after failed build): $cfg_name"
        return 0
      fi
      log "WARN: build failed but the index file exists; still trying search"
      have_index=1
    else
      OK_BUILD=$((OK_BUILD + 1))
      have_index=1
    fi
  fi

  if [[ "$have_index" != "1" && "$DRY_RUN" != "1" ]]; then
    record_fail build "$dataset" "$algo" "$cfg_name" "missing index $index_path"
    return 0
  fi

  # ---- nprobe sweep ----
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
      log "DRY_RUN search: $cfg_name nprobe=$np"
      continue
    fi
    log "SEARCH $algo $dataset $cfg_name nprobe=$np (IVF_METRIC=$metric) ..."
    local rc=0
    LD_LIBRARY_PATH="$lib_path:${LD_LIBRARY_PATH:-}" IVF_METRIC="$metric" \
      "$bin" search "$data_dir" "$index_path" "$np" "$K" "$sjson" \
      >"$slog" 2>&1 || rc=$?
    if [[ $rc -ne 0 ]]; then
      record_fail search "$dataset" "$algo" "$cfg_name" "nprobe=$np rc=$rc log=$slog"
      echo "search nprobe=$np rc=$rc" >>"$cfg_fail_log"
      continue
    fi
    OK_SEARCH=$((OK_SEARCH + 1))
  done

  # ---- curve CSV ----
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

run_algo_all_configs() {
  local dataset="$1"
  local algo="$2"
  local bin="$3"
  local index_ext="$4"
  local dim="$5"
  local nb="$6"
  local lib_path="$7"

  local nlist m
  local any=0
  while read -r nlist m; do
    [[ -z "${nlist:-}" ]] && continue
    any=1
    run_one_config "$dataset" "$algo" "$bin" "$index_ext" "$nlist" "$m" "$dim" "$nb" "$lib_path" \
      || record_fail config "$dataset" "$algo" "nlist${nlist}_m${m}" "run_one_config status=$?"
  done < <(list_build_configs "$dataset" "$dim")

  if [[ "$any" == "0" ]]; then
    record_fail configs "$dataset" "$algo" "-" "no valid build configs for dim=$dim"
  fi
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
  if [[ "$FAIL_COUNT" -gt 0 ]]; then
    log "See failures: $FAIL_LOG"
  fi
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

  if ! ensure_bins; then
    record_fail setup "-" "-" "-" "missing binaries; abort"
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
  log "CLEAR_RESULT=$CLEAR_RESULT"

  local ds
  for ds in "${datasets[@]}"; do
    log "======== dataset=$ds ========"

    if [[ "$CLEAR_RESULT" == "1" && "$DRY_RUN" != "1" ]]; then
      clear_dataset_result "$ds"
    elif [[ "$CLEAR_RESULT" == "1" && "$DRY_RUN" == "1" ]]; then
      log "DRY_RUN would clear: $RESULT_ROOT/$ds"
    fi

    local base_file="$DATA_DIR/$ds/base.fbin"
    if [[ ! -f "$base_file" ]]; then
      record_fail dataset "$ds" "-" "-" "missing $base_file"
      continue
    fi
    local dim nb
    read -r dim nb < <(read_dim_nb "$base_file")
    if [[ -z "${dim:-}" || -z "${nb:-}" ]]; then
      record_fail dataset "$ds" "-" "-" "cannot read the header of $base_file"
      continue
    fi

    log "---- dataset=$ds nb=$nb dim=$dim base=$base_file ----"
    log "build configs:"
    list_build_configs "$ds" "$dim" | while read -r a b; do log "  nlist=$a m=$b"; done || true

    mkdir -p "$RESULT_ROOT/$ds" || {
      record_fail mkdir "$ds" "-" "-" "cannot mkdir $RESULT_ROOT/$ds"
      continue
    }

    run_algo_all_configs "$ds" "faiss_ivf_pq" "$FAISS_BIN" "faissindex" "$dim" "$nb" "$FAISS_LIB_PATH"
    run_algo_all_configs "$ds" "cuvs_ivf_pq" "$CUVS_BIN" "cuvsindex" "$dim" "$nb" "$CUVS_LIB_PATH"
  done

  log "ALL DONE. Results under $RESULT_ROOT"
  write_summary
  # 2: finished, but some steps failed
  if [[ "$FAIL_COUNT" -gt 0 ]]; then
    return 2
  fi
  return 0
}

main "$@"
