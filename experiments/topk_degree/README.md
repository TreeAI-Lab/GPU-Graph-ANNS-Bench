# Top-K and graph degree (Figures 11 and 12)

Figure 11 varies the number of returned neighbors K on SIFT1M. This folder produces its curves of CAGRA, GGNN,
GANNS, SONG, BANG and PilotANN for K = 40, 70, 100, and its HNSW curves: Faiss HNSW on the CPU with 1, 10, 20 and 40
threads for K = 10, 40, 70, 100. The K = 10 curves of the other systems are those of Figure 7
(`experiments/qps_recall`), and the Vamana curves for K = 40, 70, 100 are data of the original manuscript
(`experiments/original`). Figure 12 varies the graph degree on SIFT1M; this folder produces its PilotANN curves, with
NSG graphs of degree 64 and 128. Its other curves are data of the original manuscript (`experiments/original`).

## Prerequisites

- Dataset `sift1M` in `$DATA_DIR` (`datasets/README.md`).
- GPU and hybrid systems: the SIFT1M indexes and search programs of Figure 7 (`experiments/qps_recall/README.md`,
  `build_graphs.sh sift1M`), with BANG in `$W/build/bang/r128` and PilotANN in `$ENV_PILOT`.
- HNSW: Faiss built by `third_party/build/faiss.sh`, then `bash src/baselines/build.sh faiss` (`$BIN_DIR/faiss_hnsw`).
- PilotANN: `bash third_party/build/pilotann.sh` (installs into `$ENV_PILOT`). Degree 128 needs the larger traversal
  buffer of `third_party/patches/pilotann.patch` (BUFFER_SIZE 512, at least beam width 4 x degree). The index builder
  and the sweep are `src/search/pilotann/pilotann_build.py` and `pilotann_sweep.py`, shared with Figure 7.

## Commands

```bash
source env.sh
bash experiments/topk_degree/run_gpu_topk.sh           # CAGRA, GGNN, GANNS, SONG, BANG, PilotANN; ONLY=<system>
$ENV_BUILD/bin/python experiments/topk_degree/summarize_gpu_topk.py --root $OUT_DIR/topk_degree/gpu
bash src/baselines/build.sh faiss
bash experiments/topk_degree/run_hnsw_topk.sh          # builds the HNSW index if missing, then the sweep
bash experiments/topk_degree/run_pilotann_degree.sh    # builds the degree-64 and degree-128 indexes if missing
```

## Outputs

| File | Content |
|---|---|
| `$OUT_DIR/topk_degree/gpu/gt.k<K>.ibin` | the first K ids of every query of `gt.ibin` |
| `$OUT_DIR/topk_degree/gpu/k<K>/<system>/it.txt` | the effort values of the sweep |
| `$OUT_DIR/topk_degree/gpu/k<K>/<system>/sweep.tsv` | the columns of the Figure 7 sweep of the system (`experiments/qps_recall/README.md`); the column recall@10 holds Recall@K |
| `$OUT_DIR/topk_degree/gpu/topk.txt` | dataset, algo, degree, k, it, itopk, M, nq, qps, recall, wall_ms, kernel_ms (all K and systems) |
| `$INDEX_DIR/sift1M/hnsw/M32_efc200/hnsw_M32.faissindex` | Faiss IndexHNSWFlat |
| `$OUT_DIR/topk_degree/hnsw/k<K>/t<T>/sweep.tsv` | threads, batch, efSearch, k, nq, n_batches, wall_ms, qps, recall@K, wall_ms_std, n_timed |
| `$OUT_DIR/topk_degree/hnsw/summary.tsv` | dataset, algo, degree, efSearch, k, threads, nq, wall_ms, qps, recall, wall_ms_std, n_timed (all K and thread counts) |
| `$INDEX_DIR/sift1M/pilotann/nsg<D>_efc80_dp192_r0.25/index.pt` | PilotANN checkpoint, D = 64, 128 (the layout of `experiments/qps_recall`) |
| `$OUT_DIR/topk_degree/pilotann/degree<D>/sweep.tsv` | it (= ef), itopk, nq, warmup_pipeline_ms, h2d_query_ms, pipeline_ms, d2h_topk_ms, wall_ms, qps, recall@10, gpu_used_mib, host_rss_mib, svd_ms, entry_ms, parts_sum_ms, pipeline_ms_std, wall_ms_std, n_timed; times are means over the three timed runs |
| `$OUT_DIR/topk_degree/pilotann/degree<D>/run.log` | one line per run and one `[pilotann] sift1M ef=<ef> mean wall=... qps=... recall=...` line per ef |

## Settings

GPU and hybrid systems (Figure 11): the indexes, batch (all 10,000 queries), warm-up and timed runs of Figure 7,
with K neighbors returned. The sweeps are those of Figure 7 without the points that cannot return K neighbors.

| System | Sweep for K = 40, 70, 100 | QPS |
|---|---|---|
| CAGRA | max_iterations from `it_schedule.txt`; internal top-k = max(32, K, max_iterations) rounded up to a multiple of 32 | nq / mean kernel time |
| GGNN | max_iterations from `it_schedule.txt`; KQuery = K | nq / mean kernel time |
| GANNS | n_e from `it_schedule.txt` whose candidate list holds K: all for K = 40, from 68 for K = 70 and 100 | nq / mean kernel time |
| SONG | max_iterations from `it_schedule_song.txt` above K | nq / mean kernel time |
| BANG | L = K to 512, step 8 | nq / mean wall time |
| PilotANN | ef of Figure 7 that are >= K | nq / mean wall time |

- The GANNS driver sizes its candidate list from n_e: 64, doubled until it holds n_e. SONG returns max_iterations - 1
  neighbors per query.

HNSW (Figure 11) and PilotANN (Figure 12):

| | HNSW (Figure 11) | PilotANN (Figure 12) |
|---|---|---|
| index | IndexHNSWFlat, M = 32, efConstruction = 200, L2, built with 10 threads | NSG, degree 64 or 128, ef_construct 80, d_principle 192, sample ratio 0.25 |
| sweep | efSearch of `ef_schedule.txt` (15 to 500) that are >= K, plus K | ef = 16 to 64 step 4, 68 to 128 step 8, 144 to 256 step 16, 288 to 512 step 32 |
| queries | all 10,000 in one `search()` call, threads 1, 10, 20, 40 | the 10,000 queries as one batch |
| repetitions | 1 untimed, then 3 timed (`SEARCH_REPEATS`), QPS = 10,000 / mean wall time | 1 untimed, then 3 timed, QPS = 10,000 / mean wall time |
| recall | Recall@K (K = 10, 40, 70, 100) | Recall@10 |

- PilotANN wall time covers the CPU projection of the queries, their H2D copy, the GPU entry search and the CPU-GPU
  pipeline, with a CUDA synchronization at every stage boundary.
- For HNSW, Faiss searches with max(efSearch, K), so only efSearch >= K is swept.
