#!/usr/bin/env python3
"""CAGRA iteration sweep (cuVS 25.02 Python API).

Loads the index once (rebuilding it with the dataset if it is graph-only, see cagra_py_common.py), takes the first
batch of queries, and for every max_iterations value in --it-file runs the search once as an untimed warm-up and
then TIMED_RUNS = 3 times timed. itopk_size = max(32, k, it) rounded up to a multiple of 32, or --fixed-itopk for
every it when it is given (Figure 13); --itopk-size is not used by the sweep. --team-size 0 lets cuVS choose the
team size.
Times: query H2D, cagra.search and top-k D2H with CUDA events; wall_ms is the host time around the three steps.
Outputs in --out: sweep.tsv (per it the means over the timed runs, qps = nq / mean wall_ms, then the sample standard
deviations kernel_ms_std and wall_ms_std and n_timed), it_batches.tsv (every run), gpu_mem_cuda.tsv,
load_timing.tsv, params.txt.
"""
from __future__ import annotations

import argparse
import os
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cagra_py_common import (  # noqa: E402
    _cuda_check,
    append_mem,
    ensure_index_with_dataset,
    gpu_mem_mib,
    pinned_array,
    read_fbin_header,
    read_ibin,
    recall_at_k,
)

import numpy as np
from cuda import cudart
from pylibraft.common import device_ndarray
from cuvs.neighbors import cagra

# Timed runs per max_iterations value, after one untimed warm-up run
TIMED_RUNS = 3


def itopk_for_it(it: int, k: int = 10) -> int:
    """itopk follows it, rounded up to a multiple of 32 and at least max(32, k)."""
    need = max(it, k, 32)
    return ((need + 31) // 32) * 32


def load_its(path: str) -> list[int]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(int(line))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--gt", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--batch-size", type=int, required=True)
    ap.add_argument("--it-file", required=True)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--itopk-size", type=int, default=64)
    ap.add_argument("--search-width", type=int, default=1)
    ap.add_argument("--fixed-itopk", type=int, default=0, help="itopk_size for every it; 0: itopk follows it")
    ap.add_argument("--team-size", type=int, default=0, help="threads per distance; 0: chosen by cuVS")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--cache-dir", default="")
    ap.add_argument("--graph-degree", type=int, default=32)
    ap.add_argument("--intermediate-graph-degree", type=int, default=64)
    ap.add_argument("--nn-descent-niter", type=int, default=20)
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    its = load_its(args.it_file)
    bs = args.batch_size

    cache_dir = Path(args.cache_dir) if args.cache_dir else out.parent / "_cagra_cache"
    t0 = time.perf_counter()
    _cuda_check(cudart.cudaSetDevice(0))
    _ = device_ndarray(np.zeros(1, dtype=np.float32))
    index = ensure_index_with_dataset(
        args.index,
        args.base,
        cache_dir / "index_with_dataset.cagra",
        args.graph_degree,
        args.intermediate_graph_degree,
        args.nn_descent_niter,
    )
    nq_total, dim = read_fbin_header(args.query)
    gt = read_ibin(args.gt) if args.gt and Path(args.gt).is_file() else None
    q_mm = np.memmap(args.query, dtype=np.float32, mode="r", offset=8, shape=(nq_total, dim))
    nq = min(bs, nq_total)

    d_q = device_ndarray(np.zeros((nq, dim), dtype=np.float32))
    d_n = device_ndarray(np.zeros((nq, args.k), dtype=np.uint32))
    d_d = device_ndarray(np.zeros((nq, args.k), dtype=np.float32))
    h_q, h_q_ptr = pinned_array((nq, dim), np.float32)
    h_n, h_n_ptr = pinned_array((nq, args.k), np.uint32)
    np.copyto(h_q, q_mm[:nq], casting="no")
    ptr_q = d_q.__cuda_array_interface__["data"][0]

    ev = {}
    for name in ("h2d0", "h2d1", "k0", "k1", "d2h0", "d2h1"):
        ev[name] = _cuda_check(cudart.cudaEventCreate())

    load_ms = (time.perf_counter() - t0) * 1000.0
    (out / "load_timing.tsv").write_text(f"stage\tms\nload_index_and_init\t{load_ms:.3f}\n")
    append_mem(out / "gpu_mem_cuda.tsv", "after_load")
    with open(out / "params.txt", "w") as f:
        f.write("algo\tCAGRA\n")
        f.write(f"batch_size\t{bs}\n")
        f.write(f"n_query_this_run\t{nq}\n")
        f.write(f"protocol\tfirst_batch_only_warmup1_timed{TIMED_RUNS}_mean\n")
        f.write(f"it_file\t{args.it_file}\n")
        f.write(f"n_it\t{len(its)}\n")
        if args.fixed_itopk:
            f.write(f"itopk_policy\tfixed: itopk={args.fixed_itopk}\n")
        else:
            f.write("itopk_policy\tlinear_round32: itopk=ceil32(max(32, k, it))\n")
        f.write(f"search_width\t{args.search_width}\n")
        f.write(f"team_size\t{args.team_size}\n")
        f.write("h2d_src\tcudaMallocHost pinned\n")
        f.write(f"graph_degree\t{args.graph_degree}\n")
        f.write(f"intermediate_graph_degree\t{args.intermediate_graph_degree}\n")
        f.write(f"nn_descent_niter\t{args.nn_descent_niter}\n")

    sweep = out / "sweep.tsv"
    detail = out / "it_batches.tsv"
    with open(sweep, "w") as sf, open(detail, "w") as df:
        sf.write(
            "it\titopk\tnq\twarmup_kernel_ms\th2d_query_ms\tkernel_ms\td2h_topk_ms\t"
            "wall_ms\tqps\trecall@10\tgpu_used_mib\tkernel_ms_std\twall_ms_std\tn_timed\n"
        )
        df.write(
            "it\titopk\tphase\tnq\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10\n"
        )

        def run_once(it: int, itopk: int, phase: str):
            sp = cagra.SearchParams(
                max_iterations=it,
                itopk_size=itopk,
                search_width=args.search_width,
                team_size=args.team_size,
            )
            wall0 = time.perf_counter()
            _cuda_check(cudart.cudaEventRecord(ev["h2d0"], None))
            _cuda_check(
                cudart.cudaMemcpy(
                    ptr_q,
                    h_q.ctypes.data,
                    nq * dim * 4,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )
            _cuda_check(cudart.cudaEventRecord(ev["h2d1"], None))
            _cuda_check(cudart.cudaEventSynchronize(ev["h2d1"]))
            err, h2d_ms = cudart.cudaEventElapsedTime(ev["h2d0"], ev["h2d1"])
            _cuda_check(err)

            _cuda_check(cudart.cudaEventRecord(ev["k0"], None))
            distances, neighbors = cagra.search(
                sp, index, d_q, args.k, neighbors=d_n, distances=d_d
            )
            _cuda_check(cudart.cudaEventRecord(ev["k1"], None))
            _cuda_check(cudart.cudaEventSynchronize(ev["k1"]))
            err, kernel_ms = cudart.cudaEventElapsedTime(ev["k0"], ev["k1"])
            _cuda_check(err)

            ptr_n = neighbors.__cuda_array_interface__["data"][0]
            _cuda_check(cudart.cudaEventRecord(ev["d2h0"], None))
            _cuda_check(
                cudart.cudaMemcpy(
                    h_n.ctypes.data,
                    ptr_n,
                    nq * args.k * 4,
                    cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost,
                )
            )
            _cuda_check(cudart.cudaEventRecord(ev["d2h1"], None))
            _cuda_check(cudart.cudaEventSynchronize(ev["d2h1"]))
            err, d2h_ms = cudart.cudaEventElapsedTime(ev["d2h0"], ev["d2h1"])
            _cuda_check(err)
            wall_ms = (time.perf_counter() - wall0) * 1000.0
            qps = nq / (wall_ms / 1000.0) if wall_ms > 0 else 0.0
            rec = ""
            if gt is not None:
                r = recall_at_k(h_n.astype(np.int64), gt[:nq, : args.k])
                rec = f"{r:.6f}"
            df.write(
                f"{it}\t{itopk}\t{phase}\t{nq}\t{h2d_ms:.4f}\t{kernel_ms:.4f}\t{d2h_ms:.4f}\t"
                f"{wall_ms:.4f}\t{qps:.2f}\t{rec}\n"
            )
            df.flush()
            return h2d_ms, kernel_ms, d2h_ms, wall_ms, qps, rec

        for i, it in enumerate(its):
            itopk = args.fixed_itopk or itopk_for_it(it, args.k)
            try:
                wh2d, wk, wd2h, ww, wq, wrec = run_once(it, itopk, "warmup")
                timed = [run_once(it, itopk, f"timed{r + 1}") for r in range(TIMED_RUNS)]
            except Exception as e:
                print(f"[cagra] FAIL it={it} itopk={itopk}: {e}", flush=True)
                sf.write(f"{it}\t{itopk}\t{nq}\t\t\t\t\t\t\t\tFAIL:{e}\n")
                sf.flush()
                continue
            h2d = statistics.mean(t[0] for t in timed)
            kerns = [t[1] for t in timed]
            walls = [t[3] for t in timed]
            kern = statistics.mean(kerns)
            d2h = statistics.mean(t[2] for t in timed)
            wall = statistics.mean(walls)
            qps = nq / (wall / 1000.0) if wall > 0 else 0.0
            recs = [float(t[5]) for t in timed if t[5]]
            rec = f"{statistics.mean(recs):.6f}" if recs else ""
            kern_sd = statistics.stdev(kerns) if len(kerns) > 1 else 0.0
            wall_sd = statistics.stdev(walls) if len(walls) > 1 else 0.0
            used, _, _ = gpu_mem_mib()
            if i == 0:
                append_mem(out / "gpu_mem_cuda.tsv", "after_first_it")
            sf.write(
                f"{it}\t{itopk}\t{nq}\t{wk:.4f}\t{h2d:.4f}\t{kern:.4f}\t{d2h:.4f}\t"
                f"{wall:.4f}\t{qps:.2f}\t{rec}\t{used:.1f}\t{kern_sd:.4f}\t{wall_sd:.4f}\t{TIMED_RUNS}\n"
            )
            sf.flush()
            print(
                f"[cagra] it={it}/{its[-1]} itopk={itopk} nq={nq} "
                f"h2d={h2d:.2f} kern={kern:.2f} d2h={d2h:.2f} qps={qps:.0f} recall={rec} "
                f"(mean of {TIMED_RUNS})",
                flush=True,
            )

    append_mem(out / "gpu_mem_cuda.tsv", "after_last_it")
    _cuda_check(cudart.cudaFreeHost(h_q_ptr))
    _cuda_check(cudart.cudaFreeHost(h_n_ptr))
    print(f"[cagra] sweep done -> {out}", flush=True)


if __name__ == "__main__":
    main()
