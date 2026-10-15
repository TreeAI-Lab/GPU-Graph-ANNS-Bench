# Per-dataset settings of the QPS-recall runs (Figures 1, 7 and 10). Sourced by the scripts of this folder.
#
# Dataset ids are the names used in the figures. DEEP10M is the 9,990,000-vector deep-image-96 set
# (data directory deep10M-annb); the other ids are also the data directory names under $DATA_DIR.
#
# Every system gets the same maximum degree D per dataset:
#   CAGRA  graph_degree D, intermediate_graph_degree 2D, 20 NN-descent iterations
#   GGNN   KBuild D, tau_build 0.5, 2 refinement iterations
#   GANNS  k = D/2 (stored degree up to 2k = D), candidate list M = 2D
#   SONG   FIXED_DEGREE D (forward degree D/2), 2^shift >= D + 1 adjacency slots per vertex

# shellcheck disable=SC2034
GRAPH_DATASETS="sift1M deep1M deep10M glove200 nytimes gist deep100M"
BANG_DATASETS="sift1M deep1M deep10M gist glove200 nytimes"
PILOTANN_DATASETS="sift1M deep1M deep10M gist"

# data n dim D nq cagra_tag ggnn_tag ganns_tag song_tag song_shift ("-": not run on this dataset)
cfg_graph() {
  case "$1" in
    sift1M)   echo "sift1M 1000000 128 32 10000 gd32_igd64_it20 k32_tau0.5_r2 k16_m64 fd32_shift6_deg16 6" ;;
    deep1M)   echo "deep1M 1000000 96 32 10000 gd32_igd64_it20 k32_tau0.5_r2 k16_m64 fd32_shift6_deg16 6" ;;
    deep10M)  echo "deep10M-annb 9990000 96 32 10000 gd32_igd64_it20 k32_tau0.5_r2 k16_m64 fd32_shift6_deg16 6" ;;
    glove200) echo "glove200 1183514 200 64 10000 gd64_igd128_it20 k64_tau0.5_r2 k32_m128 fd64_shift7_deg32 7" ;;
    nytimes)  echo "nytimes 289761 256 32 9991 gd32_igd64_it20 k32_tau0.5_r2 k16_m64 fd32_shift6_deg16 6" ;;
    gist)     echo "gist 1000000 960 128 1000 gd128_igd256_it20 k128_tau0.5_r2 k64_m256 fd128_shift8_deg64 8" ;;
    # DEEP100M: CAGRA built with the IVF-PQ graph builder (NN-descent does not fit in GPU memory), GGNN built in
    # 5 shards of 20,000,000 vectors; GANNS and SONG were not run.
    deep100M) echo "deep100M 100000000 96 32 10000 gd32_igd64_ivfpq k32_tau0.5_r2 - - -" ;;
    *) echo "unknown dataset $1" >&2; return 1 ;;
  esac
}

# BANG on a DiskANN disk index: R L M B QD (build_disk_index -R -L -M -B --QD, PQ bytes 0)
cfg_bang() {
  case "$1" in
    sift1M)   echo "32 100 16 4 32" ;;
    deep1M)   echo "64 100 16 4 32" ;;
    deep10M)  echo "64 100 32 16 32" ;;
    gist)     echo "64 100 32 8 64" ;;
    glove200) echo "96 100 40 4 40" ;;
    nytimes)  echo "64 100 32 4 32" ;;
    *) echo "no BANG settings for $1" >&2; return 1 ;;
  esac
}

# PilotANN IndexStaged: graph_method n_neighbors ef_construct d_principle sample_ratio
cfg_pilotann() {
  case "$1" in
    sift1M)  echo "nsg 32 80 192 0.25" ;;
    deep1M)  echo "nsg 32 80 192 0.25" ;;
    deep10M) echo "nsg 64 80 192 0.25" ;;
    gist)    echo "nsg 128 80 192 0.25" ;;
    *) echo "no PilotANN settings for $1" >&2; return 1 ;;
  esac
}

bang_tag() { echo "r${1}_l${2}"; }
pilotann_tag() { echo "${1}${2}_efc${3}_dp${4}_r${5}"; }
