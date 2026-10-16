# Search kernel profiling and FP16 vectors (Tables 9 and 10, Figure 14)

Table 9 reports Nsight Compute metrics of the main search kernel of SONG, GGNN, GANNS, CAGRA, BANG and PilotANN on
DEEP1M and GIST at 95% Recall@10: SM and GPU DRAM throughput, occupancy, active threads per warp, L2 hit rate, the share of
the long-scoreboard stall, and the GPU DRAM bytes read per query. Table 10 and Figure 14 compare FP32 and FP16
vectors for SONG, GGNN, GANNS and CAGRA: Recall@10, QPS and the same hardware metrics at the operating point
(Table 10), and full QPS-recall curves (Figure 14). The FP16 programs store base and query vectors as half and
compute in FP32 (`src/fp16/README.md`).

## Platform

All measurements of this folder ran on one NVIDIA A10 24GB in a full virtual machine with the GPU passed through;
the other experiments of the paper ran on an A800 80GB PCIe. The NVIDIA driver restricts the performance counters
to administrators (`RmProfilingAdminOnly=1`), so Nsight Compute must run as root with `CAP_SYS_ADMIN`, which a
container usually lacks. `platform/run_smoke.sh` checks this. L2 hit rate, occupancy and GPU DRAM throughput depend
directly on the L2 size, the threads per SM and the memory bandwidth, so the numbers describe the kernels on the A10.

| | A10 (this folder) | A800 80GB PCIe |
|---|---|---|
| Compute capability | 8.6 | 8.0 |
| SMs | 72 | 108 |
| L2 | 6 MB | 40 MB |
| Shared memory per SM | 100 KB | 164 KB |
| Threads per SM | 1,536 | 2,048 |
| Peak memory bandwidth | 600 GB/s | 1,935 GB/s |
| Power limit, maximum SM clock | 150 W, 1,695 MHz | 300 W, 1,410 MHz |

Software on the A10: driver 550.144.03, CUDA 12.4 (nvcc 12.4.131), Nsight Compute 2024.1.1, Nsight Systems 2023.4.4.

## Prerequisites

- Datasets `$DATA_DIR/deep1M` and `$DATA_DIR/gist` (`datasets/README.md`). The batch is 10,000 queries on DEEP1M and
  1,000 on GIST (its whole query set).
- The Figure 7 graphs (`experiments/qps_recall/build_graphs.sh`), under `$INDEX_DIR/<dataset>/<system>/<tag>/`:

| System | DEEP1M | GIST | File |
|---|---|---|---|
| SONG | `fd32_shift6_deg16` | `fd128_shift8_deg64` | `bfsg.graph` |
| GGNN | `k32_tau0.5_r2` | `k128_tau0.5_r2` | the GGNN graph directory |
| GANNS | `k16_m64` | `k64_m256` | `graph.nsw` |
| CAGRA | `gd32_igd64_it20` | `gd128_igd256_it20` | `_search_cache/index_with_dataset.cagra`, the float index with its dataset that the Figure 7 CAGRA sweep (`experiments/qps_recall/run_graph_systems.sh`) writes and searches |
| PilotANN | `nsg32_efc80_dp192_r0.25` | `nsg128_efc80_dp192_r0.25` | `index.pt` under `$INDEX_DIR/<dataset>/pilotann/<tag>/`, the index of the Figure 7 PilotANN sweep; `PILOT_CKPT_DEEP1M` and `PILOT_CKPT_GIST` point to another copy (read by `run_point.sh`) |

- Programs: `bash src/fp16/build.sh` (FP32 and FP16 search programs in `$BIN_DIR`), `bash third_party/build/diskann.sh`
  and `build_bang_graphs.sh` below (BANG graphs and `$W/build/bang/r<R>/bang_search`), and PilotANN in `$ENV_PILOT`
  compiled for the A10 without the search statistics that the patch adds to its GPU kernel
  (`PILOT_NO_KERNEL_STATS=1 TORCH_CUDA_ARCH_LIST=8.6 bash third_party/build/pilotann.sh`). The statistics (cycle
  timers, counters and the block-wide barriers they need) record path lengths and stage times; without them the
  kernel is the upstream one, as the programs of the other systems are. `profile_pilotann.sh` checks the installed extension with
  `cuobjdump` (CUDA 11.1 or newer, `CUOBJDUMP` to choose one) and stops if its SASS still reads the clock.
- Nsight Compute and Nsight Systems on `PATH`, or set `NCU` and `NSYS`. Python 3 with numpy (`$ENV_BUILD/bin/python`).

## Commands

```bash
source env.sh
bash src/fp16/build.sh
bash third_party/build/diskann.sh
bash experiments/profiling/build_bang_graphs.sh             # DEEP1M R=32, GIST R=128
bash experiments/profiling/platform/run_smoke.sh            # must end with SMOKE PASS
bash experiments/profiling/platform/collect_platform.sh
bash experiments/profiling/check_points.sh                  # FP32 and FP16 recall at the operating points
bash experiments/profiling/bang_sweep.sh deep1M             # BANG operating point, also for gist
bash experiments/profiling/find_kernels_all.sh              # main kernels and BANG launches per search
bash experiments/profiling/run_all.sh                       # Nsight Compute, Table 10 QPS, Figure 14 curves
bash experiments/profiling/profile_pilotann.sh              # PilotANN in Table 9: point, main kernel, Nsight Compute
bash experiments/profiling/make_tables.sh > "$OUT_DIR/profiling/tables.tex"
```

`run_all.sh` runs three steps, each under `log_clocks.sh`, which records SM clock, memory clock, power and
throttle reasons every 200 ms:

1. `profile_all.sh`: `profile_ncu.sh` for 18 groups (four systems in FP32 and FP16, BANG in FP32, two datasets),
   then `summarize.py` writes `summary.tsv`.
2. `qps_repeat.sh 1`: Recall@10 and QPS of Table 10, one run per group with a 2 s sleep before it. The run
   searches the operating point once as an untimed warm-up and then three times timed.
3. `qps_curve.sh`: the 16 curves of Figure 14, one sweep per curve with a 0.5 s sleep before each iteration count.
   Each iteration count is searched once as an untimed warm-up and then three times timed.

`make_tables.sh` writes the LaTeX rows of Table 9 from `summary.tsv` and of Table 10 from `summary.tsv` and
`table10_qps/qps.tsv`. `plots/fig14_fp16.py` draws Figure 14 from the `mean.tsv` files of step 3, leaving out the
points whose QPS standard deviation over the three timed searches exceeds 10% of the mean.

## Operating points

`points.tsv` holds, per dataset and system, the iteration count `it` (the L of BANG), the reference Recall@10 and
QPS, and the regular expression of the main kernel. For SONG, GGNN, GANNS and CAGRA the operating point is the
setting with the highest QPS among those with Recall@10 >= 0.95 in the Figure 7 sweep; CAGRA uses itopk = it rounded
up to a multiple of 32. For BANG it is the first L of the `bang_sweep.sh` sweep (step 8) with Recall@10 >= 0.95,
measured on the A10: L = 50 on DEEP1M (0.9601; L = 42 gives 0.9497) and L = 66 on GIST (0.9519). PilotANN searches
the index of Figure 7, so its operating point is the ef with the highest QPS among those with Recall@10 >= 0.95 in
the Figure 7 sweep: ef = 20 on DEEP1M (0.9502) and ef = 100 on GIST (0.9521). `profile_pilotann.sh` stops if the
A10 run of that ef falls below 0.95; the next candidates of Figure 7 are ef = 24 on DEEP1M (0.9532) and ef = 108 on
GIST (0.9521).

| Dataset | SONG | GGNN | GANNS | CAGRA | BANG | PilotANN |
|---|---|---|---|---|---|---|
| DEEP1M | 112 | 76 | 112 | 48 | 50 | 20 |
| GIST | 84 | 152 | 80 | 80 | 66 | 100 |

`run_point.sh` runs the operating point with the Figure 7 search parameters. SONG, GGNN, GANNS and CAGRA search the
first batch once as an untimed warm-up and then three times timed, so their main kernel is launched four times;
`sweep.tsv` holds the means over the timed searches and `it_batches.tsv` every search.
BANG runs four searches at one L (one untimed warm-up, then three timed) with `OMP_NUM_THREADS=14`; its `sweep.tsv`
has the mean wall time of the timed searches. PilotANN does the same at one ef (`pilotann_sweep.py --efs`), also with
14 CPU threads. Recall@10 is computed against the top-10
ground truth that `make_gt_k10.py` derives from `gt.ibin`.

Expected recall (`check_points.sh`); the FP32 values are those of the Figure 7 programs:

| System | DEEP1M FP32 | DEEP1M FP16 | GIST FP32 | GIST FP16 |
|---|---|---|---|---|
| SONG | 0.953700 | 0.953310 | 0.950800 | 0.950000 |
| GGNN | 0.950410 | 0.950070 | 0.951300 | 0.950400 |
| GANNS | 0.952620 | 0.952230 | 0.951800 | 0.951400 |
| CAGRA | 0.951330 | 0.951030 | 0.953100 | 0.952900 |

## Main kernels

`find_kernel.sh` traces one run at the operating point with Nsight Systems (CUDA trace only, no counter access
needed) and lists the kernels by total time. SONG, GGNN, GANNS and CAGRA have one search kernel that takes 99.9% to
100% of the kernel time, in FP32 and FP16; in the FP16 CAGRA run the one-time `float_to_half` conversion is the only
other kernel of note. BANG launches `compute_neighborDist_par` once per search iteration; it takes 72.3% of the
kernel time on DEEP1M and 90.5% on GIST, with 87 and 78 launches per search (`Instances` in `kern_sum.csv` divided
by the number of searches in the run). PilotANN searches on the GPU with `traverse_cuda_kernel`, launched once per
chunk of 250 queries, so 40 times per search on DEEP1M and 4 times on GIST, after one `routing_cuda_kernel` launch
that finds the entry points; the rest of its search runs on the CPU. `profile_all.sh` (BANG) and
`profile_pilotann.sh` therefore profile all launches of the first timed search (`NCU_LAUNCH_SKIP` = 1 x launches,
which skips the warm-up search, `NCU_LAUNCH_COUNT` = launches; set `BANG_LAUNCHES_DEEP1M` and `BANG_LAUNCHES_GIST` if
a sweep gives another L; `profile_pilotann.sh` takes the PilotANN launches from `kern_sum.csv`), and `summarize.py`
combines them into one row.

| System | Main kernel (`kernel_regex`) |
|---|---|
| SONG | `warp_independent_search_kernel` |
| GGNN | `ggnn::query<QueryKernel<...>>` |
| GANNS | `SearchK` |
| CAGRA | `cuvs::neighbors::cagra::detail::single_cta_search::search_kernel<...>` |
| BANG | `compute_neighborDist_par` |
| PilotANN | `traverse_cuda_kernel` |

## Nsight Compute settings

| Setting | Value |
|---|---|
| Clocks | not locked (`--clock-control none`); `run_all.sh` resets any locked clock first |
| Replay | `--replay-mode application`: each metric pass runs the whole program again, and `PRE_SLEEP_S=3` sleeps before each pass, so every pass starts from an idle GPU |
| Caches | flushed before each pass (`--cache-control all`) |
| Launches | the second launch of the main kernel, the first timed search (`--launch-skip 1 --launch-count 1`); BANG as above |
| Sections | `WarpStateStats` and the metrics below |

| Column | Metric |
|---|---|
| SM | `sm__throughput.avg.pct_of_peak_sustained_elapsed` |
| GPU DRAM | `dram__throughput.avg.pct_of_peak_sustained_elapsed` |
| Occ. | `sm__warps_active.avg.pct_of_peak_sustained_active` |
| Thr. | `smsp__thread_inst_executed_per_inst_executed.ratio` |
| L2 | `lts__t_sector_hit_rate.pct` |
| Long scoreboard | share of `smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio` among all stall reasons |
| Bytes per query | `dram__bytes_read.sum` divided by the number of queries |

The stall shares leave out `selected` (issuing), `not_selected` (ready but not picked) and `not_issued`, which are
not stalls. `make_tables.sh` stops if long scoreboard is not the main stall of a row. For BANG the ratios of the
launches are weighted by their own denominators (duration for SM and GPU DRAM, active SM cycles for Occ.,
instructions for Thr., issue-active cycles for the stalls, L2 sectors for the L2 hit rate) and the bytes are summed.
PilotANN is combined in the same way.

## Outputs

All under `$OUT_DIR/profiling/`.

| Path | Content |
|---|---|
| `gt/<dataset>_k10.ibin` | top-10 ground truth |
| `<dataset>_<system>[_fp16]/nsys/kern_sum.csv` | Nsight Systems kernel summary (`Time (%)`, `Total Time (ns)`, `Instances`, `Name`, ...) |
| `<dataset>_<system>[_fp16]/ncu/` | `<tag>.ncu-rep`, `raw.csv` (all metrics, base units), `ncu.log`, `ncu_version.txt`, `gpu.txt`, `run/sweep.tsv` |
| `summary.tsv` | one row per group: `dataset system precision kernel launches it recall@10 SM GPU DRAM Occ. Thr. L2 stall stall_share_pct math_pipe_share_pct dram_read_bytes_per_query ncu gpu sm_clock_mhz` (Table 9, hardware columns of Table 10) |
| `table10_qps/<dataset>_<system>_<precision>/rep<i>/` | `sweep.tsv`, `it_batches.tsv` and the program logs of each run (`rep1` only with the default of one run) |
| `table10_qps/qps.tsv` | `dataset system precision reps recall@10 qps_mean qps_std qps_median qps_min qps_max kernel_ms_mean` over the timed searches of a group (`reps` = number of timed searches, 3; QPS of a search = nq / kernel_ms); Table 10 uses `qps_mean` |
| `fp16_curves/<dataset>_<system>_<precision>/mean.tsv` | `it nq recall@10 reps qps_mean qps_std kernel_ms_mean` per iteration count over the timed searches (`reps` = 3) (Figure 14) |
| `ncu_log/`, `table10_qps/`, `fp16_curves/` | `clocks.csv` and `clocks_summary.txt` of `log_clocks.sh` |
| `tables.tex` | LaTeX rows of Tables 9 and 10 |
| `bang_sweep/<dataset>/sweep.tsv` | BANG `L`, mean wall ms, QPS, Recall@10 in percent, wall ms standard deviation, number of timed runs (one warm-up and three timed runs per L) |
| `<dataset>_pilotann/point/`, `ncu_log_pilotann/<dataset>/` | PilotANN operating-point run and clock log of `profile_pilotann.sh` |
| `smoke/`, `platform/`, `check/` | smoke test, machine description, program check |

## Notes

- QPS is `nq / kernel_ms` in Table 10 and Figure 14, the kernel-only measure of Figure 7. The QPS of BANG in
  `points.tsv` is the wall-clock value that BANG prints, and that of PilotANN the wall-clock value of Figure 7.
- The clocks are not locked because the measurements should reflect normal use. Under a sustained load the 150 W
  limit of the A10 lowers the SM clock, so every measurement starts from an idle GPU: 3 s before each Nsight
  Compute pass, 2 s before each Table 10 run, 0.5 s before each iteration count of a curve (before its warm-up
  search). `summary.tsv` gives the SM clock during each measurement (`sm_clock_mhz`), and `clocks.csv` the clock
  over the whole session.
- SONG allocates memory inside its kernel, and its kernel time can differ by up to 40% between two runs of the same
  program; the other systems differ by less than 1%. Table 10 and Figure 14 therefore report the mean over three
  timed searches, and `qps_std` their sample standard deviation. `qps_repeat.sh <n>` and `REPS=<n> qps_curve.sh`
  repeat the program runs and pool the timed searches of all runs.
- CAGRA uses `cagra_fp16` for both precisions, with the same float index and graph (`src/fp16/README.md`).
- `math_pipe_share_pct` is the share of `math_pipe_throttle` (waiting for an arithmetic pipeline), with the same
  denominator as the other stall shares; it tells whether arithmetic limits a kernel.
