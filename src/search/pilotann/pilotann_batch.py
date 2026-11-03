"""PilotANN batch-size sweep on SIFT1M (Figure 8).

For every batch size the first <batch> queries are searched as one batch, for every ef of --it-file: one untimed
warm-up search, then TIMED_RUNS = 3 timed searches. The wall time of a search covers the whole search: query
projection (SVD) on the CPU, H2D of the principal dimensions, entry search on the GPU, and the overlapped GPU
subgraph / CPU refinement pipeline. It is not a GPU kernel time. The time columns are the means over the timed
searches, QPS = batch / mean wall time, and Recall@10 is that of the timed searches.
The pipeline splits a batch into chunks; chunk_size is the largest of 256, 250, 200, ... that divides the batch.
Output: <out>/<batch>/sweep.tsv with ef nq chunk_size svd_ms query_h2d_ms entry_ms pipeline_ms wall_ms qps recall@10
pipeline_ms_std wall_ms_std n_timed (sample standard deviations over the timed searches)
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import numpy as np
import torch
from pilot_ann import kernels, layers

K = 10
TIMED_RUNS = 3  # timed searches per ef and batch size, after one untimed warm-up search


def chunk_for(bs: int) -> int:
    for c in (256, 250, 200, 125, 100, 50, 25, 20, 10, 8, 5, 4, 2, 1):
        if c <= bs and bs % c == 0:
            return c
    return 1


def load_its(path: str) -> list[int]:
    out = []
    for line in Path(path).read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            out.append(int(line))
    return out


def recall_at_k(pred: np.ndarray, gt: np.ndarray, k: int) -> float:
    hits = 0.0
    for i in range(pred.shape[0]):
        truth = set(int(x) for x in gt[i, :k])
        hits += sum(int(x) in truth for x in pred[i, :k])
    return hits / (pred.shape[0] * k)


def official_search(index, query, k, ef, chunk):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    q = torch.matmul(query, index.VT.T)
    t1 = time.perf_counter()
    q_cuda = q[:, : index.d_principle].to(device=index.cuda_device)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    ep_i, ep_d = index.entry.search(q_cuda)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    top_i, top_d = kernels.pipeline(
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
    ap.add_argument("--ckpt", required=True, type=Path, help="IndexStaged checkpoint")
    ap.add_argument("--query", required=True, help="queries, fbin")
    ap.add_argument("--gt", required=True, help="top-10 ground truth, ibin")
    ap.add_argument("--it-file", required=True, help="ef values, one per line")
    ap.add_argument("--out", required=True, type=Path, help="output folder; one sub-folder per batch size")
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 10, 100, 1000, 10000])
    args = ap.parse_args()
    its = load_its(args.it_file)
    qn, qd = np.fromfile(args.query, dtype=np.int32, count=2)
    queries = np.fromfile(args.query, dtype=np.float32, offset=8).reshape(int(qn), int(qd))
    gn, gk = np.fromfile(args.gt, dtype=np.int32, count=2)
    gt = np.fromfile(args.gt, dtype=np.int32, offset=8).reshape(int(gn), int(gk))
    print(f"load index {args.ckpt}", flush=True)
    index = layers.IndexStaged.load(args.ckpt)
    index.to(device="cuda")
    header = (
        "ef\tnq\tchunk_size\tsvd_ms\tquery_h2d_ms\tentry_ms\tpipeline_ms\twall_ms\t"
        "qps\trecall@10\tpipeline_ms_std\twall_ms_std\tn_timed\n"
    )
    for bs in args.batches:
        chunk = chunk_for(bs)
        q = torch.from_numpy(queries[:bs].copy())
        g = gt[:bs]
        out = args.out / str(bs)
        out.mkdir(parents=True, exist_ok=True)
        print(f"batch {bs} chunk {chunk}", flush=True)
        with (out / "sweep.tsv").open("w") as f:
            f.write(header)
            for ef in its:
                official_search(index, q, K, ef, chunk)  # warm-up, not timed
                runs = []
                for _ in range(TIMED_RUNS):
                    pred, t = official_search(index, q, K, ef, chunk)
                    runs.append((t, recall_at_k(pred.numpy(), g, K)))
                tm = {key: statistics.mean(t[key] for t, _ in runs) for key in runs[0][0]}
                rec = statistics.mean(r for _, r in runs)
                qps = bs / (tm["wall_ms"] / 1e3)
                pipes = [t["pipeline_ms"] for t, _ in runs]
                walls = [t["wall_ms"] for t, _ in runs]
                pipe_sd = statistics.stdev(pipes) if len(pipes) > 1 else 0.0
                wall_sd = statistics.stdev(walls) if len(walls) > 1 else 0.0
                row = (
                    f"{ef}\t{bs}\t{chunk}\t{tm['svd_ms']:.4f}\t{tm['query_h2d_ms']:.4f}\t"
                    f"{tm['entry_ms']:.4f}\t{tm['pipeline_ms']:.4f}\t{tm['wall_ms']:.4f}\t"
                    f"{qps:.2f}\t{rec:.6f}\t{pipe_sd:.4f}\t{wall_sd:.4f}\t{TIMED_RUNS}\n"
                )
                f.write(row)
                f.flush()
                print(
                    f"[pilot] bs={bs} ef={ef} wall={tm['wall_ms']:.2f} "
                    f"pipe={tm['pipeline_ms']:.2f} qps={qps:.0f} recall={rec:.4f} (mean of {TIMED_RUNS})",
                    flush=True,
                )
    print("PILOT_DONE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
