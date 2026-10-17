"""Merge the GPU top-K sweeps of Figure 11 into one table.

Reads <root>/k<K>/<system>/sweep.tsv, written by run_gpu_topk.sh, and writes <root>/topk.txt with one row per
point: dataset, algo, degree, k, it, itopk, M, nq, qps, recall, wall_ms, kernel_ms.
it is the effort value of the sweep (iterations, GANNS n_e, BANG L, PilotANN ef), itopk the CAGRA internal top-k
(BANG L, PilotANN ef) and M the GANNS candidate list. qps = nq / mean wall time, recall = Recall@K (the recall@10
column of sweep.tsv holds Recall@K in these runs). kernel_ms is the mean search kernel time, for PilotANN the
CPU-GPU pipeline time, and empty for BANG. Figure 11 plots nq / kernel_ms for CAGRA, GGNN, GANNS and SONG and qps
for BANG and PilotANN, as Figure 7 does.
Usage: python summarize_gpu_topk.py --root $OUT_DIR/topk_degree/gpu [--dataset sift1M] [--degree 32]
"""
import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SYSTEMS = ("cagra", "ggnn", "ganns", "song", "bang", "pilotann")
FIELDS = ["dataset", "algo", "degree", "k", "it", "itopk", "M", "nq", "qps", "recall", "wall_ms", "kernel_ms"]


def collect(root: Path, dataset: str, degree: int) -> List[Dict[str, str]]:
    """One row per point, by increasing K, then in the order of SYSTEMS and of the sweep."""
    rows = []
    kdirs = [p for p in root.glob("k*") if p.is_dir() and p.name[1:].isdigit()]
    for kdir in sorted(kdirs, key=lambda p: int(p.name[1:])):
        for algo in SYSTEMS:
            sweep = kdir / algo / "sweep.tsv"
            if not sweep.is_file():
                logger.warning("missing %s", sweep)
                continue
            with open(sweep) as f:
                for r in csv.DictReader(f, delimiter="\t"):
                    rows.append({"dataset": dataset, "algo": algo, "degree": degree, "k": kdir.name[1:],
                                 "it": r["it"], "itopk": r.get("itopk", ""), "M": r.get("M", ""), "nq": r["nq"],
                                 "qps": r["qps"], "recall": r["recall@10"], "wall_ms": r["wall_ms"],
                                 "kernel_ms": r.get("kernel_ms") or r.get("pipeline_ms") or ""})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="$OUT_DIR/topk_degree/gpu")
    ap.add_argument("--dataset", default="sift1M")
    ap.add_argument("--degree", type=int, default=32, help="maximum degree of the Figure 7 graphs")
    args = ap.parse_args()
    root = Path(args.root)
    rows = collect(root, args.dataset, args.degree)
    if not rows:
        raise SystemExit(f"no sweep.tsv under {root}/k<K>/")
    out = root / "topk.txt"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", out, len(rows))


if __name__ == "__main__":
    main()
