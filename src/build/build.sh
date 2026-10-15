#!/bin/bash
# Build one graph, convert it to CSR, record its statistics and, unless NOSWEEP=1, sweep the common searcher
# over L.
# Usage: bash src/build/build.sh <system> <dataset> [KEY=VALUE ...]
#
# Output directory $GRAPHS/<dataset>/<system>/<tag><TAGSUFFIX>/ (GRAPHS defaults to $INDEX_DIR):
#   params.txt    system, parameters, start and end time, wall time and peak host memory of the build
#   build.log     builder output followed by the GNU time -v report
#   gpu_mem.txt   GPU memory used, sampled every 100 ms during the build (GPU builders and PilotANN);
#                 gpu_mem_baseline.txt holds the value before the build
#   graph.csr     the graph in the format of the common searcher
#   stats.tsv     unified_search stats
#   search.tsv    unified_search search over the L list
# The tag is made of the parameters, so a run with the same parameters and suffix replaces the directory.
#
# Systems and parameters, with defaults:
#   song     SD=15 SHIFT=5 BUDGET=150          degree, log2 of the adjacency stride, insertion search budget
#   ganns    K=16 M=64                         neighbors per point (degree up to 2K), candidate list size
#   ggnn     K=24 TAU=0.5 REFINE=2             KBuild, tau_build, refinement_iterations
#   cagra    GD=64 IGD=128 NITER=20            graph_degree, intermediate_graph_degree, NN-descent iterations
#   pilotann TYPE=nsg NN=32 EFC=80 DP=192 RATIO=0.25   graph type, n_neighbors, ef_construct, d_principle, sample_ratio
#   hnsw     M=32 EFC=40                       M, efConstruction
#   vamana   R=64 L=100 ALPHA=1.2              max degree, build list size, alpha
# Other settings, as KEY=VALUE or in the environment:
#   GPU        CUDA device (env.sh)
#   THREADS    CPU threads (default nproc; in a container nproc can exceed the CPU quota, so set it explicitly)
#   LS         L list of the sweep;  NQ  number of queries searched (default 10000, at most all)
#   NOSWEEP=1  skip the sweep;  CLEAN=1  keep only logs and tables, delete the graph files
#   GRAPHS     output root;  TAGSUFFIX  suffix of the output directory, so that repeated builds do not replace
#              each other
set -euo pipefail
source "$(cd "$(dirname "$0")" && pwd)/../../env.sh"
SRC=$BENCH/src/build
SYS=${1:?usage: build.sh <system> <dataset> [KEY=VALUE ...]}
DS=${2:?usage: build.sh <system> <dataset> [KEY=VALUE ...]}
shift 2
for kv in "$@"; do export "$kv"; done
D=$DATA_DIR/$DS
THREADS=${THREADS:-$(nproc)}; export OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS
LS=${LS:-10,16,24,32,48,64,96,128,192,256,384,512}; NQ=${NQ:-10000}
PY_BUILD=$ENV_BUILD/bin/python
PY_PILOT=$ENV_PILOT/bin/python

# The Python of the build environment reads the header (a system python3 may be missing)
read -r N DIM < <("$PY_BUILD" -c "import struct,sys;print(*struct.unpack('<ii',open(sys.argv[1],'rb').read(8)))" "$D/base.fbin") || true
[ -n "${N:-}" ] && [ -n "${DIM:-}" ] || { echo "FAILED: cannot read header of $D/base.fbin"; exit 2; }
case $SYS in
  song)     : "${SD:=15}" "${SHIFT:=5}" "${BUDGET:=150}"; TAG=sd${SD}_shift${SHIFT}_b${BUDGET} ;;
  ganns)    : "${K:=16}" "${M:=64}"; TAG=k${K}_m${M} ;;
  ggnn)     : "${K:=24}" "${TAU:=0.5}" "${REFINE:=2}"; TAG=k${K}_tau${TAU}_r${REFINE} ;;
  cagra)    : "${GD:=64}" "${IGD:=128}" "${NITER:=20}"; TAG=gd${GD}_igd${IGD}_it${NITER} ;;
  pilotann) : "${TYPE:=nsg}" "${NN:=32}" "${EFC:=80}" "${DP:=192}" "${RATIO:=0.25}"; TAG=${TYPE}${NN}_efc${EFC}_dp${DP}_r${RATIO} ;;
  hnsw)     : "${M:=32}" "${EFC:=40}"; TAG=m${M}_efc${EFC} ;;
  vamana)   : "${R:=64}" "${L:=100}" "${ALPHA:=1.2}"; TAG=r${R}_l${L}_a${ALPHA} ;;
  *) echo "unknown system $SYS"; exit 1 ;;
esac
O=${GRAPHS:-$INDEX_DIR}/$DS/$SYS/$TAG${TAGSUFFIX:-}
rm -rf "$O"; mkdir -p "$O"; cd "$O"
{ echo "system=$SYS dataset=$DS tag=$TAG n=$N dim=$DIM gpu=$GPU threads=$THREADS host=$(hostname) start=$(date '+%F %T')"
  env | grep -E '^(SD|SHIFT|BUDGET|K|M|TAU|REFINE|GD|IGD|NITER|TYPE|NN|EFC|DP|RATIO|R|L|ALPHA)=' | sort || true; } > params.txt

# GNU time reports the wall time and the peak resident memory: /usr/bin/time if present, else the build environment's
TIME_BIN=/usr/bin/time; [ -x "$TIME_BIN" ] || TIME_BIN=$ENV_BUILD/bin/time
# Peak GPU memory of a build: the maximum of samples taken every 100 ms on its device minus the value before it
GPUMON=
case $SYS in ganns|ggnn|cagra|pilotann)
  nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits > gpu_mem_baseline.txt 2>/dev/null || true
  nvidia-smi --id="$GPU" --query-gpu=memory.used --format=csv,noheader,nounits -lms 100 > gpu_mem.txt 2>/dev/null &
  GPUMON=$! ;;
esac

# A failed step shows up below as a missing graph.csr
set +e
case $SYS in
  song)
    # SONG parameters are compile-time constants: one build of SONG per parameter set and dimension
    BIN=$W/build/song/d${SD}_s${SHIFT}_b${BUDGET}_dim${DIM}/build_template/song
    [ -x "$BIN" ] || bash "$BENCH/third_party/build/song.sh" "$SD" "$SHIFT" "$BUDGET" "$DIM" > song_compile.log 2>&1
    [ -f "$D/base.libsvm" ] || "$PY_BUILD" "$BENCH/datasets/to_libsvm.py" "$D/base.fbin" "$D/base.libsvm"
    "$TIME_BIN" -v "$BIN" build "$D/base.libsvm" 0 0 "$N" "$DIM" 0 l2 > build.log 2>&1
    "$PY_PILOT" "$SRC/to_csr.py" song bfsg.graph graph.csr --n "$N" --shift "$SHIFT" >> build.log 2>&1
    rm -f bfsg.data ;;
  ganns)
    # The dimension is a compile-time constant of GANNS: one build per dimension
    BIN=$W/build/ganns/dim${DIM}/GANNS
    [ -x "$BIN" ] || bash "$BENCH/third_party/build/ganns.sh" "$DIM" > ganns_compile.log 2>&1
    CUDA_VISIBLE_DEVICES=$GPU "$TIME_BIN" -v "$BIN" -b "$D/base.fvecs" nsw "$K" "$M" graph.nsw > build.log 2>&1
    "$PY_PILOT" "$SRC/to_csr.py" ganns graph.nsw graph.csr >> build.log 2>&1 ;;
  ggnn)
    CUDA_VISIBLE_DEVICES=$GPU "$TIME_BIN" -v "$BIN_DIR/ggnn_build" "$D/base.fvecs" "$O" "$K" "$TAU" "$REFINE" 0 > build.log 2>&1
    "$PY_PILOT" "$SRC/to_csr.py" ggnn part_0.ggnn graph.csr --n "$N" --kbuild "$K" >> build.log 2>&1 ;;
  cagra)
    CUDA_VISIBLE_DEVICES=$GPU "$TIME_BIN" -v "$ENV_CAGRA/bin/python" "$SRC/cagra_build.py" "$D/base.fbin" index.cagra "$GD" "$IGD" "$NITER" > build.log 2>&1
    "$PY_PILOT" "$SRC/to_csr.py" cagra index.cagra graph.csr >> build.log 2>&1 ;;
  pilotann)
    CUDA_VISIBLE_DEVICES=$GPU "$TIME_BIN" -v "$PY_PILOT" "$SRC/pilot_build.py" "$D/base.fbin" graph.npz "$TYPE" "$NN" "$EFC" "$DP" "$RATIO" > build.log 2>&1
    "$PY_PILOT" "$SRC/to_csr.py" npz graph.npz graph.csr >> build.log 2>&1 ;;
  hnsw)
    "$TIME_BIN" -v "$PY_PILOT" "$SRC/hnsw_build.py" "$D/base.fbin" index.faiss "$M" "$EFC" > build.log 2>&1
    "$PY_PILOT" "$SRC/to_csr.py" faiss_hnsw index.faiss graph.csr >> build.log 2>&1 ;;
  vamana)
    "$TIME_BIN" -v "$W/build/diskann/apps/build_memory_index" --data_type float --dist_fn l2 --data_path "$D/base.fbin" \
      --index_path_prefix "$O/index" -R "$R" -L "$L" --alpha "$ALPHA" -T "$THREADS" > build.log 2>&1
    "$PY_PILOT" "$SRC/to_csr.py" diskann index graph.csr >> build.log 2>&1 ;;
esac
set -e
[ -z "$GPUMON" ] || kill "$GPUMON" 2>/dev/null || true
if [ ! -f graph.csr ]; then echo "FAILED: no graph.csr, see $O/build.log"; tail -5 build.log || true; exit 2; fi
echo "wall_s=$(grep -m1 'Elapsed (wall clock)' build.log | awk '{print $NF}') peak_rss_kb=$(grep -m1 'Maximum resident set size' build.log | awk '{print $NF}') end=$(date '+%F %T')" >> params.txt
"$BIN_DIR/unified_search" stats --graph graph.csr > stats.tsv
if [ "${NOSWEEP:-0}" != 1 ]; then
  "$BIN_DIR/unified_search" search --base "$D/base.fbin" --query "$D/query.fbin" --gt "$D/gt.ibin" \
    --graph graph.csr --entry "$D/entry_m8.ibin" --L "$LS" --nq "$NQ" --threads "$THREADS" \
    --reps "${SEARCH_REPS:-3}" > search.tsv
fi
if [ "${CLEAN:-0}" = 1 ]; then
  find "$O" -maxdepth 1 -type f ! -name '*.txt' ! -name '*.tsv' ! -name '*.log' -delete
  rm -rf "$O"/*/
fi
echo "OK $O"; tail -1 stats.tsv
