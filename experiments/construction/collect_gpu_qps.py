"""CPU and GPU QPS of the common searcher on the selected configurations, generated from the raw sweeps (never
edited by hand).

Reads selected_configs.tsv and <raw_dir>/<dataset>_<system>.{cpu,gpu}.tsv (the CPU and the GPU version swept over L
on the same graph) and writes one row per configuration with the distance computations and the QPS at
Recall@10 = 0.9, interpolated between the two bracketing L. QPS = nq / search_s, where search_s is the mean time of
the timed passes after one warm-up pass; n_timed_cpu and n_timed_gpu are the numbers of timed passes (NA if the
sweep does not record them). Both versions follow the same search rule, so their distance computations should
agree; both columns are kept as a check.
Usage: python collect_gpu_qps.py <selected_configs.tsv> <raw_dir> <out.tsv>
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src" / "unified_search"))
from operating_point import at_recall  # noqa: E402


def at90(f: Path) -> tuple[float | None, float | None, str]:
    """(distance computations, QPS, number of timed passes) at Recall@10 = 0.9."""
    if not f.exists():
        return None, None, "NA"
    first = next(csv.DictReader(open(f), delimiter="\t"))
    nq = int(first["nq"])
    s = at_recall(f, "search_s")
    return at_recall(f, "mean_ndist"), (nq / s if s else None), first.get("n_timed") or "NA"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", type=Path)
    ap.add_argument("raw", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    fmt = lambda v: f"{v:.0f}" if v is not None else "NA"
    out = []
    for r in csv.DictReader(open(a.configs), delimiter="\t"):
        ds, sysn = r["dataset"], r["system"]
        nd_c, q_c, t_c = at90(a.raw / f"{ds}_{sysn}.cpu.tsv")
        nd_g, q_g, t_g = at90(a.raw / f"{ds}_{sysn}.gpu.tsv")
        ratio = f"{q_g / q_c:.1f}" if q_c and q_g else "NA"
        out.append([ds, sysn, r["build_args"], fmt(nd_c), fmt(nd_g), fmt(q_c), fmt(q_g), ratio, t_c, t_g])
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w") as fo:
        fo.write("dataset\tsystem\tbuild_args\tndist_at_r90_cpu\tndist_at_r90_gpu\tqps_cpu_at_r90\tqps_gpu_at_r90\tgpu_over_cpu\t"
                 "n_timed_cpu\tn_timed_gpu\n")
        fo.writelines("\t".join(x) + "\n" for x in out)
    print(f"wrote {a.out}: {sum(x[6] != 'NA' for x in out)} of {len(out)} configs have GPU results")


if __name__ == "__main__":
    main()
