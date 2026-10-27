"""Merge the batch-size sweeps on SIFT1M and pick the points of Figure 8.

Reads <root>/<batch>/<algo>/sweep.tsv for CAGRA, GGNN, GANNS and SONG, <root>/bang/<batch>/sweep.tsv and
<root>/pilotann/<batch>/sweep.tsv, and, if present, the `row` lines of <root>/hnsw/<batch>.log (faiss_hnsw). Writes
  <root>/summary.tsv    CAGRA, GGNN, GANNS, SONG: batch_size, algo, it, itopk, nq, warmup_kernel_ms, h2d_query_ms,
                        kernel_ms, d2h_topk_ms, wall_ms, qps, recall@10, gpu_used_mib, kernel_ms_std, wall_ms_std,
                        n_timed (times are means over the n_timed timed runs; empty: not measured)
  <root>/figure08.tsv   system, batch_size, param, qps, time_ms
For every system the setting is the smallest iteration count (L for BANG, ef for PilotANN) that reaches
Recall@10 >= 0.95 at batch 10,000; the same setting is used for every batch size, because recall over the few
queries of a small batch varies a lot. QPS: CAGRA, GGNN, GANNS, SONG: batch / kernel time; BANG, PilotANN: the qps
column, batch / wall time of the whole search; HNSW: the qps column of faiss_hnsw, all queries / wall time of all
search calls of the batch size. time_ms = batch / QPS, the time of one batch.
Usage: python summarize.py --root $OUT_DIR/batch_size
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List, Tuple

BATCHES = (1, 10, 100, 1000, 10000)
GPU_ALGOS = ("cagra", "ggnn", "ganns", "song")
FIELDS = ["batch_size", "algo", "it", "itopk", "nq", "warmup_kernel_ms", "h2d_query_ms", "kernel_ms", "d2h_topk_ms",
          "wall_ms", "qps", "recall@10", "gpu_used_mib", "kernel_ms_std", "wall_ms_std", "n_timed"]
SYSTEMS = ("SONG", "GGNN", "GANNS", "CAGRA", "BANG", "PilotANN")
GPU_KEY = {"SONG": "song", "GGNN": "ggnn", "GANNS": "ganns", "CAGRA": "cagra"}
HYBRID = {"BANG": ("bang", "L", "recall"), "PilotANN": ("pilotann", "ef", "recall@10")}  # folder, parameter, recall
HNSW_COLS = ["threads", "batch", "efSearch", "k", "nq", "n_batches", "wall_ms", "qps", "recall@10", "wall_ms_std",
             "n_timed"]  # fields after "row" in the output of faiss_hnsw search
TARGET = 0.95


def read_tsv(path: Path) -> List[Dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f, delimiter="\t"))


def smallest_reaching(rows: List[Dict[str, str]], key: str, rcol: str) -> int:
    ok = sorted(int(r[key]) for r in rows if r.get(rcol) and float(r[rcol]) >= TARGET)
    if not ok:
        raise SystemExit(f"no setting reaches Recall@10 {TARGET} at batch 10000")
    return ok[0]


def gpu_point(summary: List[Dict[str, str]], algo: str, bs: int) -> Tuple[float, str]:
    full = [r for r in summary if r["algo"] == algo and r["batch_size"] == "10000"]
    it = smallest_reaching(full, "it", "recall@10")
    r = next(r for r in summary if r["algo"] == algo and r["batch_size"] == str(bs) and int(r["it"]) == it)
    return bs / float(r["kernel_ms"]) * 1000.0, f"it={it}"


def hybrid_point(root: Path, system: str, bs: int) -> Tuple[float, str]:
    sub, key, rcol = HYBRID[system]
    param = smallest_reaching(read_tsv(root / sub / "10000" / "sweep.tsv"), key, rcol)
    rows = read_tsv(root / sub / str(bs) / "sweep.tsv")
    r = next(r for r in rows if int(r[key]) == param)
    return float(r["qps"]), f"{key}={param}"


def hnsw_rows(path: Path) -> List[Dict[str, str]]:
    return [dict(zip(HNSW_COLS, line.split("\t")[1:])) for line in path.read_text().splitlines()
            if line.startswith("row\t")]


def hnsw_point(root: Path, bs: int) -> Tuple[float, str]:
    param = smallest_reaching(hnsw_rows(root / "hnsw" / "10000.log"), "efSearch", "recall@10")
    r = next(r for r in hnsw_rows(root / "hnsw" / f"{bs}.log") if int(r["efSearch"]) == param)
    return float(r["qps"]), f"ef={param}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path, help="output folder of the runs ($OUT_DIR/batch_size)")
    args = ap.parse_args()
    root = args.root

    summary = []
    for bs in BATCHES:
        for algo in GPU_ALGOS:
            p = root / str(bs) / algo / "sweep.tsv"
            if not p.exists():
                continue
            for r in read_tsv(p):
                summary.append({**r, "batch_size": str(bs), "algo": algo})
    with (root / "summary.tsv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(summary)
    print(f"wrote {root / 'summary.tsv'} n={len(summary)}")

    rows = []
    for system in SYSTEMS:
        for bs in BATCHES:
            if system in HYBRID:
                q, param = hybrid_point(root, system, bs)
            else:
                q, param = gpu_point(summary, GPU_KEY[system], bs)
            q = round(q, 2)
            rows.append((system, bs, param, f"{q:.2f}", f"{bs / q * 1000.0:.4f}"))
    if all((root / "hnsw" / f"{bs}.log").exists() for bs in BATCHES):
        for bs in BATCHES:
            q, param = hnsw_point(root, bs)
            q = round(q, 2)
            rows.append(("HNSW", bs, param, f"{q:.2f}", f"{bs / q * 1000.0:.4f}"))
    else:
        print(f"no HNSW logs for every batch size in {root / 'hnsw'}, HNSW left out")
    with (root / "figure08.tsv").open("w", newline="") as f:
        w = csv.writer(f, delimiter="\t", lineterminator="\n")
        w.writerow(["system", "batch_size", "param", "qps", "time_ms"])
        w.writerows(rows)
    print(f"wrote {root / 'figure08.tsv'} n={len(rows)}")


if __name__ == "__main__":
    main()
