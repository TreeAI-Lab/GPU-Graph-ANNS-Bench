# Hardware Profiling and FP16

Paper: Tables 7 and 8, and Figure 15.

## Settings

Hardware counters require driver administrator rights that are unavailable on the A800 server, so all experiments in this subsection run on a virtual machine with an NVIDIA A10 and 24 GB of GDDR6.

We profile the main search kernel of each algorithm on GIST and DEEP1M at 95% Recall@10, with batches of 10,000 queries, 1,000 on GIST.

We repeat this on SONG, GGNN, GANNS, and CAGRA with the base and query vectors stored in FP16, while still computing in FP32. BANG and PilotANN are left out, since they already read compressed or reduced vectors on the GPU.

## Run

Run from the repository root after `source env.sh`.

```bash
E=experiments/profiling
bash $E/build_bang_graphs.sh
bash $E/platform/run_smoke.sh; bash $E/platform/collect_platform.sh
bash $E/check_points.sh
bash $E/bang_sweep.sh deep1M; bash $E/bang_sweep.sh gist
bash $E/find_kernels_all.sh
bash $E/run_all.sh                           # Nsight Compute, Table 8 QPS, Figure 15 curves
bash $E/profile_pilotann.sh
bash $E/make_tables.sh > "$OUT_DIR/profiling/tables.tex"
```
