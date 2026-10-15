#!/bin/bash
# Dataset parameters shared by the build and run scripts of this folder. Sourced after env.sh.
#
# The original runs used the dataset names on the left; the release datasets on the right hold the same
# vectors (datasets/README.md). NYTimes was read raw (290,000 vectors, cosine), which is nytimes-full.
#   sift -> sift1M   gist -> gist   glove -> glove200   nytimes -> nytimes-full   deep1m -> deep1M
#   deep, deep10m -> deep10M   deep100m -> deep100M   mnist -> mnist8M

ORIG_HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ORIG_TOOLS=$ORIG_HERE/tools
ORIG_BIN=${ORIG_BIN:-$BIN_DIR/original}
ORIG_OUT=${ORIG_OUT:-$OUT_DIR/original}

# Runs per search setting: untimed warm-up runs, then timed runs, whose mean tools/parse_logs.py reports
WARMUP_RUNS=${WARMUP_RUNS:-1}
TIMED_RUNS=${TIMED_RUNS:-3}

# Runs one search setting WARMUP_RUNS + TIMED_RUNS times and writes the output of all runs to one log, each run
# after a marker line "### run warmup <i>/<n>" or "### run timed <i>/<n>" (parse_logs.py ignores the warm-up runs).
# <hook> is called after every timed run with the file that holds the output of that run and the run number, for
# the recall computed outside the program; ":" for none.
# Usage: repeat_runs <log> <hook> <command ...>
repeat_runs() {
  local log=$1 hook=$2
  shift 2
  local part=$log.part i rc
  : > "$log"
  for ((i = 1; i <= WARMUP_RUNS + TIMED_RUNS; i++)); do
    rc=0
    "$@" > "$part" 2>&1 || rc=$?
    if ((i <= WARMUP_RUNS)); then
      echo "### run warmup $i/$WARMUP_RUNS" >> "$log"
    else
      echo "### run timed $((i - WARMUP_RUNS))/$TIMED_RUNS" >> "$log"
    fi
    cat "$part" >> "$log"
    if ((rc != 0)); then
      rm -f "$part"
      return "$rc"
    fi
    ((i <= WARMUP_RUNS)) || "$hook" "$part" "$((i - WARMUP_RUNS))"
    rm -f "$part"
  done
}

# Vector dimension
dim_of() {
  case $1 in
    sift1M) echo 128 ;; gist) echo 960 ;; glove200) echo 200 ;; nytimes-full|nytimes) echo 256 ;;
    deep1M|deep10M|deep100M) echo 96 ;; mnist8M) echo 784 ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

# Graph degree used for every system on a dataset: KBuild of run_ggnn.sh of the original release, and the
# maximum degree of the CAGRA and Vamana graphs in Tables 5 and 6
degree_of() {
  case $1 in
    sift1M|deep1M|deep100M) echo 32 ;; nytimes-full|nytimes|deep10M) echo 64 ;; glove200) echo 96 ;;
    gist|mnist8M) echo 128 ;;
    *) echo "unknown dataset: $1" >&2; return 1 ;;
  esac
}

# Number of rows of an fbin file (first int32 of the header)
rows_of() {
  od -An -t d4 -N 4 "$1" | tr -d ' '
}

nq_of() {
  rows_of "$DATA_DIR/$1/query.fbin"
}

nb_of() {
  rows_of "$DATA_DIR/$1/base.fbin"
}

# Writes the run description read by tools/parse_logs.py: write_meta <dir> <system> <dataset> <k> [nq]
write_meta() {
  local dir=$1 system=$2 dataset=$3 k=$4 nq=${5:-}
  [ -n "$nq" ] || nq=$(nq_of "$dataset")
  mkdir -p "$dir"
  printf 'system\t%s\ndataset\t%s\nk\t%s\nnq\t%s\n' "$system" "$dataset" "$k" "$nq" > "$dir/meta.tsv"
}

# Ground truth as ivecs (GANNS and PilotANN read only this format), written once per dataset
gt_ivecs() {
  local ds=$1 out=$ORIG_OUT/gt/$1/groundtruth.ivecs
  if [ ! -s "$out" ]; then
    mkdir -p "$(dirname "$out")"
    python3 "$ORIG_TOOLS/ibin_to_ivecs.py" "$DATA_DIR/$ds/gt.ibin" "$out" >&2
  fi
  echo "$out"
}

# GNU time -v reports the peak resident memory of a build (Tables 5 and 6, host memory)
time_cmd() {
  if [ -x "$ENV_BUILD/bin/time" ]; then
    echo "$ENV_BUILD/bin/time -v"
  elif [ -x /usr/bin/time ]; then
    echo "/usr/bin/time -v"
  fi
}
