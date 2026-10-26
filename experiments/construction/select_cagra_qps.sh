#!/bin/bash
# Table 7, CAGRA rows: select the CAGRA setting at a recall-QPS point comparable to the other builders. The bound B
# alone gives CAGRA sparse graphs that search at 0.39 to 0.92 of the GPU QPS of GANNS, so CAGRA must in addition
# reach FRAC (0.95) of the QPS of GANNS at Recall@10 = 0.9, read from $OUT_DIR/construction/gpu_qps.tsv (run step 3
# of the README first).
# Candidates per dataset: the degrees below in increasing order, at each degree NITER 5, 10, 15 and 20, IGD = 2 GD.
# The first candidate within B that reaches the target is selected and the dataset stops. Each candidate is built
# once (untimed) and swept over L with the GPU version of the common searcher, one warm-up and REPS timed passes per
# L; its graph files are deleted afterwards.
# Output: $OUT_DIR/construction/cagra_qps/<dataset>_<tag>.gpu.tsv and $OUT_DIR/construction/cagra_qps_selection.tsv,
#   one row per candidate (cagra_qps_selection.tsv in this directory is the table measured for the paper). The
#   selected settings are the CAGRA rows of selected_configs.tsv, which run_timing.sh and run_gpu_qps.sh time and
#   measure.
# Usage: bash select_cagra_qps.sh        DATASETS="nytimes glove200" bash select_cagra_qps.sh
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"

REPS=${REPS:-3}              # timed passes per L, after one warm-up pass
THREADS=${THREADS:-14}       # CPU threads of the build
REF=${REF:-ganns}            # builder whose QPS sets the target
FRAC=${FRAC:-0.95}           # target = FRAC x QPS of REF
DATASETS=${DATASETS:-"sift1M deep1M nytimes glove200"}

BS=$BENCH/src/build/build.sh
PY=$ENV_BUILD/bin/python
OUT=$OUT_DIR/construction
RAW=$OUT/cagra_qps
SEL=$OUT/cagra_qps_selection.tsv
mkdir -p "$RAW" "$OUT/logs"; LOG=$OUT/logs/cagra_qps_$(date +%Y%m%d_%H%M%S).log
say(){ echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }
# L lists of the sweep, per dataset (those of run_timing.sh)
lslist(){ case $1 in sift1M|deep1M) echo 10,16,24,32,48,64,96,128,192,256,384,512;;
          nytimes) echo 10,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048;;
          *) echo 10,16,24,32,48,64,96,128,192,256,384,512,768,1024,1536,2048,3072,4096;; esac; }
declare -A B=([sift1M]=1000 [deep1M]=1000 [nytimes]=6000 [glove200]=14500)
declare -A DEGS=([sift1M]="8 9 10 11 12 16 24 32" [deep1M]="9 10 11 12 16 24 32"
                 [nytimes]="9 10 11 12 13 14 15 16 24 32 48 64 96" [glove200]="16 24 32 48 64 96")
target(){ awk -F'\t' -v ds="$1" -v r="$REF" -v f="$FRAC" 'NR>1 && $1==ds && $2==r {printf "%.0f\n", $7*f}' "$OUT/gpu_qps.tsv"; }
at90(){ "$PY" -c "import csv,sys; sys.path.insert(0,'$BENCH/src/unified_search'); from pathlib import Path; from operating_point import at_recall; f=Path('$1'); nq=int(next(csv.DictReader(open(f),delimiter='\t'))['nq']); s=at_recall(f,'search_s'); nd=at_recall(f,'mean_ndist'); print(f'{nd:.0f}' if nd else 'NA', f'{nq/s:.0f}' if s else 'NA')"; }

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$busy" -gt 0 ] && [ "${FORCE:-0}" != 1 ]; then say "$busy other processes on the GPU, timings would be off. Stop them first, or set FORCE=1."; exit 1; fi
[ -f "$SEL" ] || printf 'dataset\tB\tqps_target\tbuild_args\tndist_at_r90\tqps_gpu_at_r90\twithin_B\tmeets_target\tselected\n' > "$SEL"
say "start: ref=$REF frac=$FRAC reps=$REPS gpu=$GPU"
for ds in $DATASETS; do
  t=$(target "$ds"); b=${B[$ds]}
  [ -n "$t" ] || { say "$ds: no QPS of $REF in $OUT/gpu_qps.tsv, run run_gpu_qps.sh first"; continue; }
  say "$ds: B=$b target=$t"
  found=0
  for gd in ${DEGS[$ds]}; do
    for it in 5 10 15 20; do
      args="GD=$gd IGD=$((2*gd)) NITER=$it"; tag=gd${gd}_igd$((2*gd))_it$it; o=$RAW/${ds}_$tag.gpu.tsv
      dir=$INDEX_DIR/$ds/cagra/${tag}_qsel
      if [ ! -s "$o" ]; then
        say "build $ds $args"
        env GPU="$GPU" THREADS="$THREADS" NOSWEEP=1 TAGSUFFIX=_qsel bash "$BS" cagra "$ds" $args > "$OUT/logs/qsel_build_${ds}_$tag.out" 2>&1 || true
        if [ ! -f "$dir/graph.csr" ]; then say "FAILED build $ds $args, see $OUT/logs/qsel_build_${ds}_$tag.out"; continue; fi
        say "gpu   $ds $args"
        CUDA_VISIBLE_DEVICES=$GPU "$BIN_DIR/unified_search_gpu" --base "$DATA_DIR/$ds/base.fbin" --query "$DATA_DIR/$ds/query.fbin" \
          --gt "$DATA_DIR/$ds/gt.ibin" --graph "$dir/graph.csr" --entry "$DATA_DIR/$ds/entry_m8.ibin" --L "$(lslist "$ds")" \
          --reps "$REPS" > "$o.part" 2>> "$LOG" && mv "$o.part" "$o" || { say "FAILED gpu $ds $args"; continue; }
        rm -f "$dir/index.cagra" "$dir/graph.csr"
      fi
      read -r nd q < <(at90 "$o")
      inb=no; [ "$nd" != NA ] && [ "$nd" -le "$b" ] && inb=yes
      met=no; [ "$q" != NA ] && awk -v q="$q" -v t="$t" 'BEGIN{exit !(q>=t)}' && met=yes
      sel=no; [ $inb = yes ] && [ $met = yes ] && sel=yes
      printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "$ds" "$b" "$t" "$args" "$nd" "$q" $inb $met $sel >> "$SEL"
      say "$ds $args: ndist@0.9 $nd (B $b) QPS@0.9 $q (target $t) selected $sel"
      if [ $sel = yes ]; then found=1; break; fi
    done
    [ $found = 1 ] && break
  done
  [ $found = 1 ] || say "$ds: no candidate is within B and reaches the target"
done
say "done: $SEL"
