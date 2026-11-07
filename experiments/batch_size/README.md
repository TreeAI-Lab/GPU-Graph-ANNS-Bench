# Batch Size and Latency

Paper: Figure 10.

## Settings

Figure 10 evaluates batch sizes of {1, 10, 100, 1000, 10000} on SIFT1M at 95% Recall@10, comparing the six GPU-based graph methods with HNSW using 20 threads. Every method completes its initial loading before timing begins. GPU-resident methods reuse the loaded vectors and index across batches. Their batch time includes query uploads, search, and result downloads. For CPU-GPU methods, it additionally includes recurring host-side work and communication; HNSW uses the complete CPU search-call time. We report throughput and average latency per batch, excluding the waiting time needed to collect queries.

## Run

Run from the repository root after `source env.sh`.

```bash
E=experiments/batch_size
bash $E/build_indexes.sh
bash $E/run_gpu_systems.sh; bash $E/run_bang.sh; bash $E/run_pilotann.sh; bash $E/run_hnsw.sh
$ENV_BUILD/bin/python $E/summarize.py --root $OUT_DIR/batch_size
```
