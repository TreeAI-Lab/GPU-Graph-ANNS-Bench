"""Hops of the common searcher at Recall@10 = 0.9 (Figure 5(b)), generated from the graph directories (never edited
by hand).

Reads configs.tsv and <graphs_root>/<dataset>/<system>/<tag>_depth/{search.tsv,stats.tsv} and writes one row per
configuration: average out-degree without duplicates, weakly connected components, number of queries, the two L
around Recall 0.9, and the mean hops and distance computations at Recall@10 = 0.9 (interpolated between those two
L). A configuration whose sweep does not reach 0.9 gets NA, with the highest recall it reached.

Usage: python collect.py <graphs_root> <out.tsv> [--configs configs.tsv]
"""
import argparse
import csv
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "src" / "build"))
sys.path.insert(0, str(ROOT / "src" / "unified_search"))
from operating_point import at_recall  # noqa: E402
from tags import tag_of  # noqa: E402


def bracket(search_tsv: Path, target: float = 0.90) -> str:
    """The two L whose recall brackets target, as "64-96"."""
    prev = None
    for r in csv.DictReader(open(search_tsv), delimiter="\t"):
        if float(r["recall_at_k"]) >= target:
            return f"{prev['L'] if prev else r['L']}-{r['L']}"
        prev = r
    return "NA"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("graphs", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--configs", type=Path, default=HERE / "configs.tsv")
    a = ap.parse_args()
    out = []
    for r in csv.DictReader(open(a.configs), delimiter="\t"):
        ds, sysn, args = r["dataset"], r["system"], r["build_args"]
        d = a.graphs / ds / sysn / (tag_of(sysn, args) + "_depth")
        row = [ds, r["max_degree"], sysn, args]
        if not (d / "search.tsv").exists():
            out.append(row + ["NA"] * 7)
            continue
        st = list(csv.DictReader(open(d / "stats.tsv"), delimiter="\t"))[0]
        search = list(csv.DictReader(open(d / "search.tsv"), delimiter="\t"))
        hops, nd = at_recall(d / "search.tsv", "mean_hops"), at_recall(d / "search.tsv", "mean_ndist")
        fmt = lambda v: f"{v:.1f}" if v is not None else "NA"
        out.append(row + [f"{float(st['avg_unique_out_degree']):.2f}", st["weak_cc"], search[0]["nq"],
                          bracket(d / "search.tsv"), fmt(hops), f"{nd:.0f}" if nd is not None else "NA",
                          f"{max(float(x['recall_at_k']) for x in search):.4f}"])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as f:
        f.write("dataset\tmax_degree\tsystem\tbuild_args\tuniq_out_degree\tweak_cc\tnq\tL_at_r90\thops_at_r90\tndist_at_r90\tmax_recall\n")
        f.writelines("\t".join(x) + "\n" for x in out)
    done = sum(x[4] != "NA" for x in out)
    print(f"wrote {a.out}: {done} of {len(out)} configs have results")


if __name__ == "__main__":
    main()
