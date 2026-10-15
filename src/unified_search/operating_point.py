"""Values of a common-searcher sweep at a recall target.

A sweep is the output of `unified_search search` (or unified_search_gpu) over a list of L values, one row per L.
at_recall returns a column at the target Recall@10, interpolated linearly between the two L values whose recall
brackets the target; the first row if it already reaches the target; None if no row reaches it.

Usage: python src/unified_search/operating_point.py --check <search.tsv> <B>
prints "<distance computations at Recall@10 = 0.9> PASS" if they do not exceed B, else "... FAIL".
"""
import argparse
import csv
from pathlib import Path


def at_recall(search_tsv: Path, col: str, target: float = 0.90) -> float | None:
    """Column col of search_tsv where the recall reaches target, interpolated between the two bracketing L."""
    prev = None
    for r in csv.DictReader(open(search_tsv), delimiter="\t"):
        rec, v = float(r["recall_at_k"]), float(r[col])
        if rec >= target:
            if prev is None:
                return v
            pr, pv = prev
            return pv + (target - pr) / (rec - pr) * (v - pv)
        prev = (rec, v)
    return None


def ndist_at(search_tsv: Path, target: float = 0.90) -> float | None:
    return at_recall(search_tsv, "mean_ndist", target)


def qps_at(search_tsv: Path, target: float = 0.90) -> float | None:
    """Queries per second at target: number of queries over the interpolated search time."""
    nq = int(next(csv.DictReader(open(search_tsv), delimiter="\t"))["nq"])
    s = at_recall(search_tsv, "search_s", target)
    return nq / s if s else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", nargs=2, metavar=("SEARCH_TSV", "B"), required=True)
    a = ap.parse_args()
    nd, b = ndist_at(Path(a.check[0])), float(a.check[1])
    print(f"{nd:.0f} PASS" if nd is not None and nd <= b else f"{'NA' if nd is None else f'{nd:.0f}'} FAIL")


if __name__ == "__main__":
    main()
