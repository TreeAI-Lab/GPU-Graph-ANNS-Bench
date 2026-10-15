"""GPU-resident CAGRA batch search with the cuVS 25.02 Python API (Figure 6, deep1M and deep10M-annb).

Loads the index and the base vectors once, then searches the query file in batches and records
H2D query copy, search kernel and D2H top-k copy of every batch with CUDA events.

Output files in --out:
  params.txt, load_timing.tsv, gpu_mem_cuda.tsv,
  batch_timing.tsv  batch_id nq h2d_query_ms kernel_ms d2h_topk_ms wall_ms qps recall@10
  recall.tsv        batch_id recall@10   (when --gt is given)

The cuVS 25.02 Python API cannot attach a dataset to a graph-only index (no update_dataset()).
So if --index has no dataset, the index is rebuilt once with the same IndexParams
(graph_degree, intermediate_graph_degree, nn_descent_niter) from --base and cached as
<cache-dir>/index_with_dataset.cagra, which later runs load directly.

Usage (environment $ENV_CAGRA):
  python cagra_resident_search.py --index index.cagra --base base.fbin --query query.5M.fbin \
      --gt gt.5M.k10.ibin --out DIR --batch-size 10000 --k 10 --max-iterations 48 --itopk-size 64
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
from cuda import cudart
from pylibraft.common import device_ndarray
from cuvs.neighbors import cagra

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "search" / "cagra"))
from cagra_py_common import (  # noqa: E402
    _cuda_check, append_mem, ensure_index_with_dataset, pinned_array, read_fbin_header, read_ibin, recall_at_k,
)






def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", required=True, help="index.cagra (graph-only or with dataset)")
    ap.add_argument("--base", required=True, help="base.fbin (needed if index has no dataset)")
    ap.add_argument("--query", required=True, help="query fbin, e.g. query.5M.fbin")
    ap.add_argument("--gt", default="", help="optional gt.ibin (nq,k) for recall@10")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--batch-size", type=int, default=10000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--max-iterations", type=int, default=55)
    ap.add_argument("--itopk-size", type=int, default=64)
    ap.add_argument("--search-width", type=int, default=1)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--graph-degree", type=int, default=32)
    ap.add_argument("--intermediate-graph-degree", type=int, default=64)
    ap.add_argument("--nn-descent-niter", type=int, default=20)
    ap.add_argument(
        "--cache-dir",
        default="",
        help="where to store index_with_dataset.cagra if rebuild needed",
    )
    ap.add_argument("--max-batches", type=int, default=0, help="0=all")
    args = ap.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else out.parent / "_cagra_cache"
    cache_path = cache_dir / "index_with_dataset.cagra"

    t_load0 = time.perf_counter()
    # CUDA init
    _cuda_check(cudart.cudaSetDevice(0))
    _ = device_ndarray(np.zeros(1, dtype=np.float32))

    index = ensure_index_with_dataset(
        args.index,
        args.base,
        cache_path,
        args.graph_degree,
        args.intermediate_graph_degree,
        args.nn_descent_niter,
    )
    nq_total, dim = read_fbin_header(args.query)
    assert dim == index.dim, f"query dim {dim} != index dim {index.dim}"

    gt = None
    if args.gt and Path(args.gt).is_file():
        gt = read_ibin(args.gt)
        assert gt.shape[0] == nq_total, f"gt nq {gt.shape[0]} != query nq {nq_total}"
        print(f"[cagra] GT loaded {gt.shape}", flush=True)
    else:
        print("[cagra] no GT; recall column will be empty", flush=True)

    # read the query file batch by batch through np.memmap
    q_mm = np.memmap(args.query, dtype=np.float32, mode="r", offset=8, shape=(nq_total, dim))

    sp = cagra.SearchParams(
        max_iterations=args.max_iterations,
        itopk_size=args.itopk_size,
        search_width=args.search_width,
    )

    # Device and pinned host buffers for the largest batch
    # (the memmap-to-pinned copy is outside the H2D event, as in the C++ drivers)
    bs = args.batch_size
    d_q = device_ndarray(np.zeros((bs, dim), dtype=np.float32))
    d_n = device_ndarray(np.zeros((bs, args.k), dtype=np.uint32))
    d_d = device_ndarray(np.zeros((bs, args.k), dtype=np.float32))
    h_q, h_q_ptr = pinned_array((bs, dim), np.float32)
    h_n, h_n_ptr = pinned_array((bs, args.k), np.uint32)
    ptr_q = d_q.__cuda_array_interface__["data"][0]
    ptr_n = d_n.__cuda_array_interface__["data"][0]

    # cudaEvents
    ev = {}
    for name in ("h2d0", "h2d1", "k0", "k1", "d2h0", "d2h1"):
        err, e = cudart.cudaEventCreate()
        _cuda_check(err)
        ev[name] = e

    load_ms = (time.perf_counter() - t_load0) * 1000.0
    with open(out / "load_timing.tsv", "w") as f:
        f.write("stage\tms\n")
        f.write(f"load_index_and_init\t{load_ms:.3f}\n")
    append_mem(out / "gpu_mem_cuda.tsv", "after_load")

    with open(out / "params.txt", "w") as f:
        f.write(f"algo\tCAGRA\n")
        f.write(f"index\t{args.index}\n")
        f.write(f"base\t{args.base}\n")
        f.write(f"query\t{args.query}\n")
        f.write(f"gt\t{args.gt}\n")
        f.write(f"batch_size\t{bs}\n")
        f.write(f"k\t{args.k}\n")
        f.write(f"max_iterations\t{args.max_iterations}\n")
        f.write(f"itopk_size\t{args.itopk_size}\n")
        f.write(f"search_width\t{args.search_width}\n")
        f.write(f"n_query_total\t{nq_total}\n")
        f.write(f"dim\t{dim}\n")
        f.write(f"gpu\t{args.gpu}\n")
        f.write(
            "note\tPython cuvs 25.02 has no update_dataset; "
            "graph-only index triggers one-time rebuild+cache with same IndexParams\n"
        )
        f.write("h2d_src\tcudaMallocHost pinned (memmap copy is outside H2D event)\n")

    batch_path = out / "batch_timing.tsv"
    recall_path = out / "recall.tsv"
    n_batches = (nq_total + bs - 1) // bs
    if args.max_batches > 0:
        n_batches = min(n_batches, args.max_batches)

    print(f"[cagra] {nq_total} queries, batch={bs}, batches={n_batches}", flush=True)

    with open(batch_path, "w") as bf, open(recall_path, "w") as rf:
        bf.write(
            "batch_id\tnq\th2d_query_ms\tkernel_ms\td2h_topk_ms\twall_ms\tqps\trecall@10\n"
        )
        rf.write("batch_id\trecall@10\n")

        for bi in range(n_batches):
            s = bi * bs
            e = min(s + bs, nq_total)
            nq = e - s
            # memmap to pinned host memory (not in the H2D timer, like the file read of the C++ drivers)
            np.copyto(h_q[:nq], q_mm[s:e], casting="no")

            if nq != bs:
                d_q_b = device_ndarray(np.zeros((nq, dim), dtype=np.float32))
                d_n_b = device_ndarray(np.zeros((nq, args.k), dtype=np.uint32))
                d_d_b = device_ndarray(np.zeros((nq, args.k), dtype=np.float32))
                ptr_q_b = d_q_b.__cuda_array_interface__["data"][0]
                ptr_n_b = d_n_b.__cuda_array_interface__["data"][0]
            else:
                d_q_b, d_n_b, d_d_b = d_q, d_n, d_d
                ptr_q_b, ptr_n_b = ptr_q, ptr_n

            wall0 = time.perf_counter()

            # H2D query copy (pinned to device)
            _cuda_check(cudart.cudaEventRecord(ev["h2d0"], None))
            _cuda_check(
                cudart.cudaMemcpy(
                    ptr_q_b,
                    h_q.ctypes.data,
                    nq * dim * 4,
                    cudart.cudaMemcpyKind.cudaMemcpyHostToDevice,
                )
            )
            _cuda_check(cudart.cudaEventRecord(ev["h2d1"], None))
            _cuda_check(cudart.cudaEventSynchronize(ev["h2d1"]))
            err, h2d_ms = cudart.cudaEventElapsedTime(ev["h2d0"], ev["h2d1"])
            _cuda_check(err)

            # kernel (cagra.search syncs internally via auto_sync_resources)
            _cuda_check(cudart.cudaEventRecord(ev["k0"], None))
            distances, neighbors = cagra.search(
                sp, index, d_q_b, args.k, neighbors=d_n_b, distances=d_d_b
            )
            _cuda_check(cudart.cudaEventRecord(ev["k1"], None))
            _cuda_check(cudart.cudaEventSynchronize(ev["k1"]))
            err, kernel_ms = cudart.cudaEventElapsedTime(ev["k0"], ev["k1"])
            _cuda_check(err)

            # D2H top-k ids to pinned memory
            ptr_n_b = neighbors.__cuda_array_interface__["data"][0]
            _cuda_check(cudart.cudaEventRecord(ev["d2h0"], None))
            _cuda_check(
                cudart.cudaMemcpy(
                    h_n.ctypes.data,
                    ptr_n_b,
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
                r = recall_at_k(h_n[:nq].astype(np.int64), gt[s:e, : args.k])
                rec = f"{r:.6f}"
                rf.write(f"{bi}\t{r:.6f}\n")
                rf.flush()

            bf.write(
                f"{bi}\t{nq}\t{h2d_ms:.4f}\t{kernel_ms:.4f}\t{d2h_ms:.4f}\t"
                f"{wall_ms:.4f}\t{qps:.2f}\t{rec}\n"
            )
            bf.flush()
            if bi == 0:
                append_mem(out / "gpu_mem_cuda.tsv", "after_first_batch")
            if bi == n_batches - 1:
                append_mem(out / "gpu_mem_cuda.tsv", "after_last_batch")
            if bi % 10 == 0 or bi == n_batches - 1:
                print(
                    f"[cagra] batch {bi}/{n_batches-1} nq={nq} "
                    f"h2d={h2d_ms:.2f} kern={kernel_ms:.2f} d2h={d2h_ms:.2f} "
                    f"wall={wall_ms:.2f} qps={qps:.0f} recall={rec}",
                    flush=True,
                )

    _cuda_check(cudart.cudaFreeHost(h_q_ptr))
    _cuda_check(cudart.cudaFreeHost(h_n_ptr))
    print(f"[cagra] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
