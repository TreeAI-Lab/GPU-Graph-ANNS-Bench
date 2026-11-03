"""Merge the Figure 13 runs into the table that plots/fig13_cagra_parameters.py reads.

Reads <root>/<parameter>-<value>/sweep.tsv, whose one row holds the means over the three timed runs, and writes
<root>/fig13_cagra_parameters.csv: parameter, value, recall_at_10, time_us (mean kernel time scaled to 10,000
queries, the time behind the CAGRA QPS of Figure 7), kernel_ms, wall_ms, nq.
Usage: python summarize.py --root $OUT_DIR/cagra_parameters
"""
import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

PARAMETERS = ("search_width", "internal_topk", "degree", "team_size")
FIELDS = ["parameter", "value", "recall_at_10", "time_us", "kernel_ms", "wall_ms", "nq"]


def collect(root: Path) -> List[Dict[str, str]]:
    """One row per setting, in the order of PARAMETERS and increasing value."""
    rows = []
    for param in PARAMETERS:
        runs = sorted(root.glob(f"{param}-*/sweep.tsv"), key=lambda p: int(p.parent.name.rsplit("-", 1)[1]))
        for p in runs:
            with open(p) as f:
                r = next(csv.DictReader(f, delimiter="\t"), None)
            if r is None or not r["kernel_ms"]:
                logger.warning("skip %s: no measured row", p)
                continue
            nq = int(r["nq"])
            rows.append({"parameter": param, "value": p.parent.name.rsplit("-", 1)[1],
                         "recall_at_10": r["recall@10"], "time_us": f"{float(r['kernel_ms']) * 1000 * 10000 / nq:.3f}",
                         "kernel_ms": r["kernel_ms"], "wall_ms": r["wall_ms"], "nq": r["nq"]})
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, help="$OUT_DIR/cagra_parameters")
    args = ap.parse_args()
    root = Path(args.root)
    rows = collect(root)
    if not rows:
        raise SystemExit(f"no sweep.tsv under {root}")
    out = root / "fig13_cagra_parameters.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", out, len(rows))


if __name__ == "__main__":
    main()
