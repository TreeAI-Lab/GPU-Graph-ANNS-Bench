# Paths shared by every script. Source it from the repository root: `source env.sh`.
# Override any variable by exporting it before sourcing.

export BENCH=${BENCH:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}

# Work directory: third-party sources ($W/src), builds ($W/build), conda environments ($W/envs)
export W=${W:-$BENCH/work}
# Datasets prepared by datasets/prepare.py, one directory per dataset
export DATA_DIR=${DATA_DIR:-$W/data}
# Built graphs and indexes
export INDEX_DIR=${INDEX_DIR:-$W/indexes}
# Raw output of the experiments (logs, sweep tables)
export OUT_DIR=${OUT_DIR:-$W/out}
# Programs compiled from src/
export BIN_DIR=${BIN_DIR:-$W/bin}

# Conda environments, see environment/README.md
export ENV_BUILD=${ENV_BUILD:-$W/envs/build}        # CUDA 12.4 toolchain: SONG, GGNN, GANNS, DiskANN, BANG, our drivers
export ENV_CAGRA=${ENV_CAGRA:-$W/envs/cagra}        # libcuvs 25.02.01: CAGRA
export ENV_PILOT=${ENV_PILOT:-$W/envs/pilot}        # PilotANN, Faiss CPU (HNSW)
export ENV_FAISS=${ENV_FAISS:-$W/envs/faiss-build}  # CUDA 12.4: Faiss GPU built from source (IVF-PQ)
export ENV_CUVS=${ENV_CUVS:-$W/envs/cuvs-build}     # CUDA 12.4: cuVS built from source (IVF-PQ, brute force)

export GPU=${GPU:-0}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-$GPU}
mkdir -p "$DATA_DIR" "$INDEX_DIR" "$OUT_DIR" "$BIN_DIR"
