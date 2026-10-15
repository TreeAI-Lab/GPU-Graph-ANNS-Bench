"""Trial table of the parameter search, generated from the graph directories (never edited by hand).

One row per configuration of the grid that has a search.tsv: dataset, system, tag, average out-degree without
duplicates, weakly connected components, distance computations at Recall@10 = 0.90 and 0.95 (interpolated between
the two bracketing L; NA if the sweep does not reach the recall), the largest L of the sweep and its recall, and
the wall time of the build (reference only; the sweep machine need not be exclusive).
Usage: python collect_sweep.py <grid.tsv> <graphs_root> > sweep_trials.tsv
"""
import argparse
import csv
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "build"))
from tags import tag_of  # noqa: E402


def at(rows: list[dict], target: float) -> str:
    prev = None
    for r in rows:
        rec, nd = float(r["recall_at_k"]), float(r["mean_ndist"])
        if rec >= target:
            if prev is None:
                return f"{nd:.0f}"
            pr, pn = prev
            return f"{pn + (target - pr) / (rec - pr) * (nd - pn):.0f}"
        prev = (rec, nd)
    return "NA"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("grid", type=Path)
    ap.add_argument("graphs", type=Path)
    a = ap.parse_args()
    cols = ["dataset", "system", "tag", "uniq_out_degree", "weak_cc", "ndist_at_r90", "ndist_at_r95", "max_L", "recall_at_max_L", "wall_s_reference"]
    print("\t".join(cols))
    for g in csv.DictReader(open(a.grid), delimiter="\t"):
        ds, sysn = g["dataset"], g["system"]
        tag = tag_of(sysn, g["build_args"])
        d = a.graphs / ds / sysn / f"{tag}_sweep"
        if not (d / "search.tsv").exists():
            continue
        rows = list(csv.DictReader(open(d / "search.tsv"), delimiter="\t"))
        st = list(csv.DictReader(open(d / "stats.tsv"), delimiter="\t"))[0]
        wall = re.search(r"wall_s=(\S+)", (d / "params.txt").read_text())
        print("\t".join([ds, sysn, tag, f"{float(st['avg_unique_out_degree']):.2f}", st["weak_cc"],
                         at(rows, 0.90), at(rows, 0.95), rows[-1]["L"], f"{float(rows[-1]['recall_at_k']):.4f}",
                         wall.group(1) if wall else "NA"]))


if __name__ == "__main__":
    main()
