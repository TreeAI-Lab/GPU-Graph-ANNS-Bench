# QPS and Recall

Paper: The GPU graph curves of Figures 1, 7, and 11.

## Settings

We measure QPS and Recall@10 by varying the number of search iterations in the range [10, 500]. Each batch contains 10,000 queries, except on GIST, where it contains 1,000. Initial loading is excluded. For the pure GPU indexes, QPS uses the search kernel time on the resident index, measured with CUDA events. BANG and PilotANN also work on the host, so their QPS uses the wall-clock time of the whole search call. GGNN processes DEEP100M partition by partition, so its curve on that dataset does not represent a fully GPU-resident index.

We evaluate the scalability of GPU-accelerated algorithms using the DEEP dataset series, ranging from 1M to 100M vectors (Figure 11).

See `parameters/subsec_settings.md` for the search settings at target recall.

## Run

Run from the repository root after `source env.sh`.

```bash
E=experiments/qps_recall
bash $E/build_graphs.sh
bash $E/run_graph_systems.sh                 # CAGRA, GGNN, GANNS, SONG
bash $E/run_bang.sh
bash $E/run_pilotann.sh
$ENV_BUILD/bin/python $E/summarize.py --root $OUT_DIR/qps_recall
```
