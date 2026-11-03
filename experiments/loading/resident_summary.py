"""Summary of the resident serving runs of one dataset: means over batches 1..end (batch 0 is the warm-up).

For every system, the chosen effort is read from <root>/<system>/<calib-dir>/calib_chosen.txt and the
full run from <root>/<system>/search_r1w1_b<batch>_it<it>_k<k>_r95/batch_timing.tsv.
Output: <root>/summary_resident_search_r95.tsv with columns
  algo chosen_it calib_recall@10 n_batches_summary mean_h2d_ms mean_kernel_ms mean_d2h_ms
  mean_wall_ms mean_qps mean_recall@10 tag out_dir status

Usage: python resident_summary.py --root DIR --systems cagra ggnn ganns song [--calib-suffix _itopklin]
"""
import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List

FIELDS = ["algo", "chosen_it", "calib_recall@10", "n_batches_summary", "mean_h2d_ms", "mean_kernel_ms",
          "mean_d2h_ms", "mean_wall_ms", "mean_qps", "mean_recall@10", "tag", "out_dir", "status"]


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else math.nan


def read_meta(path: Path) -> Dict[str, str]:
    meta = {}
    for line in path.read_text().splitlines():
        if "\t" in line:
            a, b = line.split("\t", 1)
            meta[a] = b
    return meta


def summarize(root: Path, algo: str, batch: int, k: int, calib_suffix: str) -> Dict:
    cal = root / algo / f"calib_r95_b{batch}_k{k}{calib_suffix}" / "calib_chosen.txt"
    meta = read_meta(cal)
    it = meta.get("chosen_it", "")
    tag = f"search_r1w1_b{batch}_it{it}_k{k}_r95"
    out = root / algo / tag
    bt = out / "batch_timing.tsv"
    if not bt.exists():
        return {"algo": algo, "chosen_it": it, "calib_recall@10": meta.get("achieved_recall@10", ""),
                "tag": tag, "status": "missing_batch_timing"}
    h2d, kern, d2h, wall, qps, rec = [], [], [], [], [], []
    with bt.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if int(r["batch_id"]) == 0:
                continue
            h2d.append(float(r["h2d_query_ms"]))
            kern.append(float(r["kernel_ms"]))
            d2h.append(float(r["d2h_topk_ms"]))
            wall.append(float(r["wall_ms"]))
            qps.append(float(r["qps"]))
            if r.get("recall@10"):
                rec.append(float(r["recall@10"]))
    return {
        "algo": algo,
        "chosen_it": it,
        "calib_recall@10": meta.get("achieved_recall@10", ""),
        "n_batches_summary": len(h2d),
        "mean_h2d_ms": f"{mean(h2d):.4f}",
        "mean_kernel_ms": f"{mean(kern):.4f}",
        "mean_d2h_ms": f"{mean(d2h):.4f}",
        "mean_wall_ms": f"{mean(wall):.4f}",
        "mean_qps": f"{mean(qps):.2f}",
        "mean_recall@10": f"{mean(rec):.6f}" if rec else "",
        "tag": tag,
        "out_dir": str(out),
        "status": "ok",
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True, help="$OUT_DIR/loading/<dataset>")
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--batch", type=int, default=10000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--calib-suffix", default="", help="suffix of the calibration directory name")
    args = ap.parse_args()

    rows = []
    for algo in args.systems:
        cal = args.root / algo / f"calib_r95_b{args.batch}_k{args.k}{args.calib_suffix}" / "calib_chosen.txt"
        if cal.exists():
            rows.append(summarize(args.root, algo, args.batch, args.k, args.calib_suffix))
    out = args.root / "summary_resident_search_r95.tsv"
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", restval="", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(out.read_text())


if __name__ == "__main__":
    main()
