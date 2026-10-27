# Batch size (Figure 8)

Figure 8 shows QPS and the time per batch on SIFT1M at 95% Recall@10 (k = 10) for batch sizes 1, 10, 100, 1,000
and 10,000, for SONG, GGNN, GANNS, CAGRA, BANG and PilotANN, and for HNSW on the CPU with 20 threads (HNSW_20). For the
GPU systems a batch of size b is the first b queries of the SIFT1M query set; HNSW_20 searches all 10,000 queries in
search calls of b queries. Every system sweeps its search effort at every batch size; the plotted setting is the smallest
iteration count (L for BANG, ef for PilotANN) that reaches Recall@10 >= 0.95 at batch 10,000, and the same setting
is used for all batch sizes, because recall over the few queries of a small batch varies a lot.

## Prerequisites

- Dataset `$DATA_DIR/sift1M` (`base.fbin`, `base.fvecs`, `query.fbin`, `gt.ibin`), see `datasets/README.md`.
- Everything listed under Prerequisites in `experiments/qps_recall/README.md` for SIFT1M: environments, third-party
  builds (cuVS, GGNN, GANNS for dimension 128, the SONG builder, DiskANN, BANG in `$W/build/bang/r128`, PilotANN),
  and the search drivers in `$BIN_DIR` (`bash src/search/build.sh`). The PilotANN sweep is
  `src/search/pilotann/pilotann_batch.py`.

## Commands

```bash
source env.sh
bash experiments/batch_size/build_indexes.sh      # SIFT1M indexes, BANG with R = 64
bash experiments/batch_size/run_gpu_systems.sh    # CAGRA, GGNN, GANNS, SONG
bash experiments/batch_size/run_bang.sh
bash experiments/batch_size/run_pilotann.sh
$ENV_BUILD/bin/python experiments/batch_size/summarize.py --root $OUT_DIR/batch_size

# HNSW_20 on the CPU (src/baselines/build.sh faiss): IndexHNSWFlat with M 32 and efConstruction 200
"$BIN_DIR/faiss_hnsw" build "$DATA_DIR/sift1M" 32 200 20 "$INDEX_DIR/sift1M/hnsw_M32.faissindex"
mkdir -p "$OUT_DIR/batch_size/hnsw"
for b in 1 10 100 1000 10000; do
  "$BIN_DIR/faiss_hnsw" search "$DATA_DIR/sift1M" "$INDEX_DIR/sift1M/hnsw_M32.faissindex" 64 10 --threads 20 \
    --batch "$b" --ef-file experiments/qps_recall/it_schedule.txt > "$OUT_DIR/batch_size/hnsw/$b.log"
done
```

The run scripts take batch sizes as arguments (default `1 10 100 1000 10000`).

## Outputs

Under `$OUT_DIR/batch_size/` (`gt.k10.ibin` is the top-10 slice of `gt.ibin`):

| File | Content |
|---|---|
| `<batch>/<system>/sweep.tsv` (CAGRA, GGNN, GANNS, SONG) | per iteration count the means over the three timed runs, columns as in `experiments/qps_recall` |
| `bang/<batch>/raw.log`, `bang/<batch>/sweep.tsv` | per L the means over the three timed runs: `L nq recall wall_ms qps qps_no_transfer iterations calc_count path_length pq_table_ms gpu_distance_ms sort_merge_ms neighbor_filter_ms prefetch_ms l2_rerank_gpu_ms transfer_ms neighbor_seek_cpu_ms timer_sum_ms result_d2h_us wall_ms_std n_timed` |
| `pilotann/<batch>/sweep.tsv` | per ef the means over the three timed runs: `ef nq chunk_size svd_ms query_h2d_ms entry_ms pipeline_ms wall_ms qps recall@10 pipeline_ms_std wall_ms_std n_timed` |
| `summary.tsv` | CAGRA, GGNN, GANNS, SONG: `batch_size algo it itopk nq warmup_kernel_ms h2d_query_ms kernel_ms d2h_topk_ms wall_ms qps recall@10 gpu_used_mib kernel_ms_std wall_ms_std n_timed` |
| `figure08.tsv` | `system batch_size param qps time_ms`, the bars of Figure 8 (`time_ms` = batch / QPS) |
| `hnsw/<batch>.log` | one `row` line per efSearch, columns in `src/baselines/README.md` |
| `run.log`, `bang/run.log`, `pilotann_run.log` | logs |

## Settings

| System | Index | Effort sweep | Timed runs per point | QPS |
|---|---|---|---|---|
| CAGRA | `gd32_igd64_it20` of the QPS-recall runs | max_iterations, `it_schedule.txt` | 1 warm-up, 3 timed | batch / mean kernel time |
| GGNN | `k32_tau0.5_r2` | max_iterations, `it_schedule.txt` | 1 warm-up, 3 timed | batch / mean kernel time |
| GANNS | `k16_m64` | n_e, `it_schedule.txt` | 1 warm-up, 3 timed | batch / mean kernel time |
| SONG | `fd32_shift6_deg16` | max_iterations, `it_schedule_song.txt` | 1 warm-up, 3 timed | batch / mean kernel time |
| BANG | DiskANN disk index R 64, L 100, M 16, B 4, QD 32 (`r64_l100`) | L = 10 to 502, step 12 | 1 warm-up, 3 timed | batch / mean wall time of the search call |
| PilotANN | `nsg32_efc80_dp192_r0.25` of the QPS-recall runs | ef, `it_schedule.txt` | 1 warm-up, 3 timed | batch / mean wall time of the search |
| HNSW_20 | Faiss IndexHNSWFlat, M 32, efConstruction 200 | efSearch, `it_schedule.txt` | 1 warm-up, 3 timed | 10,000 / mean wall time of all search calls |

The schedules are those of `experiments/qps_recall`. Settings selected at batch 10,000 in the paper: SONG it 112,
GGNN it 64, GANNS it 104, CAGRA it 46, BANG L 34, PilotANN ef 18, HNSW_20 efSearch 36.

## Notes

- The drivers of CAGRA, GGNN, GANNS and SONG search the first `--batch-size` queries of the query file, as one
  search call (one kernel launch for SONG and GANNS).
- BANG reports its wall time in whole milliseconds, so its values for small batches only give the order of
  magnitude.
- BANG: `bang_search` (the same build as in `experiments/qps_recall`) runs every L four times in a row, one
  untimed warm-up run and three timed runs, and `parse_bang_batch.py` averages the three timed runs
  (`qps` = batch / mean wall time, `qps_no_transfer` = batch / (mean wall time - mean transfer time)).
- PilotANN splits a batch into chunks of the largest of 256, 250, 200, 125, 100, 50, 25, 20, 10, 8, 5, 4, 2, 1 that
  divides it (250 for 1,000 and 10,000).
- PilotANN reuses the SIFT1M index of the QPS-recall runs. The PilotANN values of Figure 8 in the paper come from a
  separately trained index with the same parameters.
- HNSW_20: Faiss parallelizes over the queries of one search call, so a call with a single query runs on one
  thread. The HNSW_20 values in the paper were measured on the A800 server, whose container allows 14 CPU cores.
