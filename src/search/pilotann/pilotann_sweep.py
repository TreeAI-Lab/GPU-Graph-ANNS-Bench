"""PilotANN QPS-recall sweep over ef_search (Figures 7 and 12).

The first --batch queries of the query file are searched as one batch. For each ef the search runs once as an
untimed warm-up and then TIMED_RUNS = 3 times timed, back to back. The wall time of a run covers the whole search:
query projection (SVD) on the CPU, H2D of the principal dimensions, entry search on the GPU, and the overlapped GPU
subgraph / CPU refinement pipeline (synchronized). sweep.tsv holds per ef the means over the timed runs,
QPS = nq / mean wall time, and Recall@10 of the timed runs, followed by the sample standard deviations
pipeline_ms_std and wall_ms_std and n_timed.
Outputs in --out: sweep.tsv, it_batches.tsv (every run), load_timing.tsv, params.txt.
The log prints one line per run and "[pilotann] <dataset> ef=<ef> mean wall=<ms> qps=<qps> recall=<r>" per ef.
Usage (environment $ENV_PILOT):
  python src/search/pilotann/pilotann_sweep.py --dataset sift1M --ckpt index.pt --query query.fbin --gt gt.ibin
      --out DIR [--batch 10000] [--k 10] [--efs 20,24]
--efs searches only the listed ef values (used for the operating points of experiments/profiling).
"""
from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from pilot_ann import kernels, layers

DEFAULT_EFS = (
    list(range(16, 65, 4))
    + list(range(68, 129, 8))
    + list(range(144, 257, 16))
    + list(range(288, 513, 32))
)

TIMED_RUNS = 3  # timed runs per ef, after one untimed warm-up run


def chunk_for(bs: int) -> int:
    for c in (256, 250, 200, 125, 100, 50, 25, 20, 10, 8, 5, 4, 2, 1):
        if c <= bs and bs % c == 0:
            return c
    return 1


def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    hits = 0.0
    for i in range(pred.shape[0]):
        truth = set(int(x) for x in gt[i, :k])
        hits += sum(int(x) in truth for x in pred[i, :k])
    return hits / (pred.shape[0] * k)


def gpu_mem_mib() -> float:
    try:
        free, total = torch.cuda.mem_get_info()
        return (total - free) / (1024**2)
    except Exception:
        return float("nan")


def host_rss_mib() -> float:
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return float("nan")


def to_numpy_ids(pred) -> np.ndarray:
    if torch.is_tensor(pred):
        return pred.detach().cpu().numpy()
    return np.asarray(pred)


def search(index, query, k, ef, chunk):
    """Time end-to-end search with CUDA sync at stage boundaries.

    Stages:
      svd_ms      : CPU projection query @ VT.T
      query_h2d_ms: principal dims H2D (+ sync)
      entry_ms    : entry routing on GPU (+ sync)
      pipeline_ms : overlapped subgraph/refine pipeline (+ sync)
      wall_ms     : t0..t4 full e2e
    """
    torch.cuda.synchronize()
    t0 = time.perf_counter()

    # Projection of the queries on the CPU (the queries stay on the host)
    q = torch.matmul(query, index.VT.T)
    t1 = time.perf_counter()

    q_cuda = q[:, : index.d_principle].to(device=index.cuda_device, non_blocking=False)
    torch.cuda.synchronize()
    t2 = time.perf_counter()

    ep_i, ep_d = index.entry.search(q_cuda)
    torch.cuda.synchronize()
    t3 = time.perf_counter()

    top_i, _top_d = kernels.pipeline(
        graph=[index.fullgraph.indptr, index.fullgraph.indices],
        storage=index.storage,
        subgraph=[index.subgraph.indptr, index.subgraph.indices],
        subgraph_cuda=[
            index.subgraph_cuda.indptr,
            index.subgraph_cuda.indices,
            index.mapping,
        ],
        storage_cuda=index.storage_cuda,
        query=q,
        query_cuda=q_cuda,
        initial_I=ep_i,
        initial_D=ep_d,
        k=k,
        n_neighbors=index.n_neighbors,
        ef_search=ef,
        d_principle=index.d_principle,
        chunk_size=chunk,
    )
    # Wait for the device before stopping the clock; results still on the GPU are copied to the host inside the
    # timed region, so that the wall time covers usable results.
    torch.cuda.synchronize()
    if torch.is_tensor(top_i) and top_i.is_cuda:
        top_i = top_i.cpu()
        torch.cuda.synchronize()
    t4 = time.perf_counter()

    ms = lambda a, b: (b - a) * 1e3
    return top_i, {
        "svd_ms": ms(t0, t1),
        "query_h2d_ms": ms(t1, t2),
        "entry_ms": ms(t2, t3),
        "pipeline_ms": ms(t3, t4),
        "wall_ms": ms(t0, t4),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dataset", required=True, help="dataset id, written to params.txt")
    ap.add_argument("--ckpt", required=True, type=Path, help="IndexStaged checkpoint (pilotann_build.py)")
    ap.add_argument("--query", required=True, help="queries, fbin")
    ap.add_argument("--gt", required=True, help="ground truth, ibin (the first k ids per query are used)")
    ap.add_argument("--out", required=True, type=Path, help="output directory")
    ap.add_argument("--batch", type=int, default=10000, help="number of queries searched")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--efs", default="", help="comma-separated ef values to search instead of the default sweep")
    args = ap.parse_args()

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    ckpt = args.ckpt
    assert ckpt.is_file(), f"missing {ckpt}"

    qn, qd = np.fromfile(args.query, dtype=np.int32, count=2)
    queries = np.fromfile(args.query, dtype=np.float32, offset=8).reshape(int(qn), int(qd))
    gn, gk = np.fromfile(args.gt, dtype=np.int32, count=2)
    gt = np.fromfile(args.gt, dtype=np.int32, offset=8).reshape(int(gn), int(gk))

    bs = min(int(args.batch), int(qn))
    q = torch.from_numpy(queries[:bs].copy())
    chunk = chunk_for(bs)
    efs = [int(x) for x in args.efs.split(",") if x.strip()] or DEFAULT_EFS

    print(f"load {ckpt}", flush=True)
    t_load0 = time.perf_counter()
    index = layers.IndexStaged.load(ckpt)
    index.to(device="cuda")
    load_ms = (time.perf_counter() - t_load0) * 1e3

    (out / "load_timing.tsv").write_text(f"stage\tms\nload_index\t{load_ms:.3f}\n")
    (out / "params.txt").write_text(
        f"algo\tpilotann\n"
        f"dataset\t{args.dataset}\n"
        f"ckpt\t{ckpt}\n"
        f"batch_size\t{bs}\n"
        f"chunk\t{chunk}\n"
        f"k\t{args.k}\n"
        f"n_ef\t{len(efs)}\n"
        f"runs_per_ef\t{1 + TIMED_RUNS}\n"
        f"warmup_per_ef\t1\n"
        f"timed_per_ef\t{TIMED_RUNS}\n"
        f"protocol\twarmup1_timed{TIMED_RUNS}_mean_wall_qps_svd_h2d_entry_pipeline\n"
        f"qps_definition\tnq/mean_wall_ms*1e3  wall=svd+h2d+entry+pipeline\n"
    )

    sweep_path = out / "sweep.tsv"
    detail_path = out / "it_batches.tsv"
    with sweep_path.open("w") as sf, detail_path.open("w") as df:
        sf.write(
            "it\titopk\tnq\twarmup_pipeline_ms\th2d_query_ms\tpipeline_ms\t"
            "d2h_topk_ms\twall_ms\tqps\trecall@10\tgpu_used_mib\thost_rss_mib\t"
            "svd_ms\tentry_ms\tparts_sum_ms\tpipeline_ms_std\twall_ms_std\tn_timed\n"
        )
        df.write(
            "it\trun\tphase\tnq\twall_ms\tqps\trecall@10\t"
            "svd_ms\th2d_ms\tentry_ms\tpipeline_ms\tparts_sum_ms\n"
        )

        for ef in efs:
            try:
                warmup_pipe = ""
                timed = []
                for run_i in range(1 + TIMED_RUNS):
                    phase = "warmup" if run_i == 0 else f"timed{run_i}"
                    pred, t = search(index, q, args.k, ef, chunk)
                    pred_np = to_numpy_ids(pred)
                    if pred_np.ndim == 2 and pred_np.shape[1] > args.k:
                        pred_np = pred_np[:, : args.k]
                    rec = recall_at_k(pred_np, gt[:bs], args.k)
                    qps = bs / (t["wall_ms"] / 1e3) if t["wall_ms"] > 0 else 0.0
                    parts = (
                        t["svd_ms"] + t["query_h2d_ms"] + t["entry_ms"] + t["pipeline_ms"]
                    )
                    df.write(
                        f"{ef}\t{run_i}\t{phase}\t{bs}\t{t['wall_ms']:.4f}\t{qps:.2f}\t"
                        f"{rec:.6f}\t{t['svd_ms']:.4f}\t{t['query_h2d_ms']:.4f}\t"
                        f"{t['entry_ms']:.4f}\t{t['pipeline_ms']:.4f}\t{parts:.4f}\n"
                    )
                    if run_i == 0:
                        warmup_pipe = f"{t['pipeline_ms']:.4f}"
                    else:
                        timed.append((t, rec, parts))
                    print(
                        f"[pilotann] {args.dataset} ef={ef} {phase} "
                        f"wall={t['wall_ms']:.2f} qps={qps:.0f} recall={rec:.4f} "
                        f"parts={parts:.2f}",
                        flush=True,
                    )

                t = {key: statistics.mean(tt[key] for tt, _, _ in timed) for key in timed[0][0]}
                rec = statistics.mean(r for _, r, _ in timed)
                parts = statistics.mean(pp for _, _, pp in timed)
                qps = bs / (t["wall_ms"] / 1e3) if t["wall_ms"] > 0 else 0.0
                pipes = [tt["pipeline_ms"] for tt, _, _ in timed]
                walls = [tt["wall_ms"] for tt, _, _ in timed]
                pipe_sd = statistics.stdev(pipes) if len(pipes) > 1 else 0.0
                wall_sd = statistics.stdev(walls) if len(walls) > 1 else 0.0
                used = gpu_mem_mib()
                rss = host_rss_mib()
                # The D2H of the results is part of pipeline_ms (results come back on the host)
                sf.write(
                    f"{ef}\t{ef}\t{bs}\t{warmup_pipe}\t{t['query_h2d_ms']:.4f}\t"
                    f"{t['pipeline_ms']:.4f}\t\t{t['wall_ms']:.4f}\t{qps:.2f}\t{rec:.6f}\t"
                    f"{used:.1f}\t{rss:.1f}\t{t['svd_ms']:.4f}\t{t['entry_ms']:.4f}\t"
                    f"{parts:.4f}\t{pipe_sd:.4f}\t{wall_sd:.4f}\t{TIMED_RUNS}\n"
                )
                print(
                    f"[pilotann] {args.dataset} ef={ef} mean wall={t['wall_ms']:.2f} qps={qps:.0f} "
                    f"recall={rec:.4f} (mean of {TIMED_RUNS})",
                    flush=True,
                )
                sf.flush()
                df.flush()
            except Exception as e:
                print(f"[pilotann] FAIL ef={ef}: {e}", flush=True)
                sf.write(f"{ef}\t{ef}\t{bs}\t\t\t\t\t\t\tFAIL:{e}\t\t\t\t\t\n")
                sf.flush()
                # clear sticky CUDA errors so later efs can proceed
                try:
                    torch.cuda.synchronize()
                except Exception:
                    pass
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    print(f"[pilotann] sweep done -> {out}", flush=True)


if __name__ == "__main__":
    main()
