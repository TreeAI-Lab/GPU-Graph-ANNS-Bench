"""Pick one IVF-PQ configuration (nlist, m) per dataset and library from the k = 100 nprobe sweeps.

Rule, for each (dataset, library) with library in {cuvs_ivf_pq, faiss_ivf_pq}:
  operating point of a configuration = its sweep point with the highest recall_at_10,
    ties broken by recall_at_1, then by qps_kernel;
  selected configuration = the one whose operating point has the highest recall_at_10,
    ties broken by recall_at_1, then by qps_kernel, at that point.
This prefers configurations with a high recall ceiling over fast ones with a low ceiling.
max_recall_at_1 is the recall_at_1 of the operating point, max_recall_at_100 the highest
recall_at_100 of the sweep.

Input: qps_recall_curve.csv from collect_points.py (rows with family = ivf).
Output: selected_configs.csv (2 rows per dataset), all_candidates_ranked.csv.
Datasets listed in --fixed keep the configuration given there (default: mnist8M nlist8192_m28).

Usage: python select_configs.py --points qps_recall_curve.csv --out DIR
"""
import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)

RULE = "max recall_at_10 over nprobe; ties: recall_at_1, then qps_kernel"
SELECTED_FIELDS = ["dataset", "algo", "config", "nlist", "m", "max_recall_at_1", "max_recall_at_10",
                   "max_recall_at_100", "nprobe_at_max_r10", "qps_at_max_r10", "kernel_ms_at_max_r10",
                   "selection_rule"]
RANKED_FIELDS = ["dataset", "algo", "rank_in_algo", "config", "max_recall_at_10", "max_recall_at_1",
                 "max_recall_at_100", "qps_at_max_r10", "nprobe_at_max_r10", "n_points"]


def key_of(r: Dict) -> Tuple[float, float, float]:
    return float(r["recall_at_10"]), float(r["recall_at_1"]), float(r["qps_kernel"])


def summarize(points: List[Dict]) -> Dict:
    op = max(points, key=key_of)
    return {
        "op": op,
        "r10": float(op["recall_at_10"]),
        "r1": float(op["recall_at_1"]),
        "r100": max(float(p["recall_at_100"]) for p in points),
        "qps": float(op["qps_kernel"]),
        "n": len(points),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--points", type=Path, required=True, help="qps_recall_curve.csv from collect_points.py")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--fixed", nargs="*", default=["mnist8M:8192:28"],
                    help="dataset:nlist:m kept as given for both libraries")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    groups: Dict[Tuple[str, str], Dict[str, List[Dict]]] = defaultdict(lambda: defaultdict(list))
    with args.points.open() as f:
        for r in csv.DictReader(f):
            if r["family"] == "ivf":
                groups[(r["dataset"], r["algo"])][r["config"]].append(r)
    fixed = {}
    for item in args.fixed:
        ds, nlist, m = item.split(":")
        fixed[ds] = (int(nlist), int(m))

    selected, ranked = [], []
    for (ds, algo) in sorted(groups):
        if ds in fixed:
            continue
        cands = sorted(((cfg, summarize(pts)) for cfg, pts in groups[(ds, algo)].items()),
                       key=lambda c: (-c[1]["r10"], -c[1]["r1"], -c[1]["qps"]))
        for rank, (cfg, s) in enumerate(cands, 1):
            ranked.append({
                "dataset": ds, "algo": algo, "rank_in_algo": rank, "config": cfg,
                "max_recall_at_10": round(s["r10"], 4), "max_recall_at_1": round(s["r1"], 4),
                "max_recall_at_100": round(s["r100"], 4), "qps_at_max_r10": round(s["qps"], 3),
                "nprobe_at_max_r10": int(s["op"]["nprobe"]), "n_points": s["n"]})
        cfg, s = cands[0]
        op = s["op"]
        selected.append({
            "dataset": ds, "algo": algo, "config": cfg, "nlist": op["nlist"], "m": op["m"],
            "max_recall_at_1": f"{s['r1']:.6f}", "max_recall_at_10": f"{s['r10']:.6f}",
            "max_recall_at_100": f"{s['r100']:.6f}", "nprobe_at_max_r10": op["nprobe"],
            "qps_at_max_r10": op["qps_kernel"], "kernel_ms_at_max_r10": op["kernel_ms"],
            "selection_rule": RULE})
        logger.info("%s %s: %s maxR@10=%.4f R@1=%.4f nprobe=%s qps=%s", ds, algo, cfg, s["r10"], s["r1"],
                    op["nprobe"], op["qps_kernel"])
    for ds, (nlist, m) in sorted(fixed.items()):
        for algo in ("cuvs_ivf_pq", "faiss_ivf_pq"):
            row = {k: "" for k in SELECTED_FIELDS}
            row.update(dataset=ds, algo=algo, config=f"nlist{nlist}_m{m}", nlist=nlist, m=m,
                       selection_rule=f"fixed for {ds}")
            selected.append(row)

    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows, fields in (("selected_configs.csv", selected, SELECTED_FIELDS),
                               ("all_candidates_ranked.csv", ranked, RANKED_FIELDS)):
        with (args.out / name).open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        logger.info("wrote %s (%d rows)", args.out / name, len(rows))


if __name__ == "__main__":
    main()
