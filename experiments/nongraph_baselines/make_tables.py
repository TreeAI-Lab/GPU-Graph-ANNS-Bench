"""Tables of the non-graph GPU baselines for Figures 1 and 7.

Inputs:
  <k10-root>/<dataset>/{cuvs_ivf_pq,faiss_ivf_pq}/<config>/curve/qps_recall.csv   (run_ivf_k10.sh)
  <bf-root>/<dataset>/cuvs_bf/search/survey_metrics_search.json                  (run_bf.sh)

Outputs in <out>:
  <dataset>_qps_recall.csv  one row per nprobe:
      nprobe,cuvs_config,cuvs_qps,cuvs_recall,faiss_config,faiss_qps,faiss_recall
      (recall = recall_at_10 of the program, QPS = search_qps; a cell is empty when that
      library has no point at this nprobe, e.g. Faiss above nprobe 2048)
  bf.csv                    brute force, one row per dataset (columns of collect_points.py)
  nongraph_points.csv       dataset,data_dir,system,param,recall_at_10,qps for the figure scripts;
                            IVF-PQ QPS = search_qps, brute-force QPS = qps_kernel; dataset is the
                            name of the figure panel (deep1M-annb is shown as deep1M, and so on)

Usage: python make_tables.py --k10-root DIR --bf-root DIR --out DIR
"""
import argparse
import csv
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from collect_points import FIELDS, collect_bf

logger = logging.getLogger(__name__)

LIBS = ("cuvs", "faiss")
# Dataset directory -> name of the figure panel
PANEL = {"deep1M-annb": "deep1M", "deep10M-annb": "deep10M", "nytimes-full": "nytimes"}


def read_curves(k10_root: Path) -> Dict[str, Dict[str, List[Dict]]]:
    """{dataset: {lib: rows of curve/qps_recall.csv}}"""
    out: Dict[str, Dict[str, List[Dict]]] = defaultdict(dict)
    for lib in LIBS:
        for path in sorted(k10_root.glob(f"*/{lib}_ivf_pq/*/curve/qps_recall.csv")):
            ds = path.parts[-5]
            if lib in out[ds]:
                logger.warning("%s: more than one %s configuration, using %s", ds, lib, path.parts[-3])
            with path.open() as f:
                out[ds][lib] = list(csv.DictReader(f))
    return out


def write_dataset_table(path: Path, per_lib: Dict[str, List[Dict]]) -> None:
    by_np: Dict[int, Dict[str, Dict]] = defaultdict(dict)
    for lib, rows in per_lib.items():
        for r in rows:
            by_np[int(r["nprobe"])][lib] = r
    fields = ["nprobe"] + [f"{lib}_{c}" for lib in LIBS for c in ("config", "qps", "recall")]
    cfg = {lib: rows[0]["config"] if rows else "" for lib, rows in per_lib.items()}
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        for nprobe in sorted(by_np):
            row = {"nprobe": nprobe}
            for lib in LIBS:
                r = by_np[nprobe].get(lib)
                row[f"{lib}_config"] = cfg.get(lib, "")
                row[f"{lib}_qps"] = r["search_qps"] if r else ""
                row[f"{lib}_recall"] = r["recall_at_10"] if r else ""
            w.writerow(row)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k10-root", type=Path, required=True)
    ap.add_argument("--bf-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args.out.mkdir(parents=True, exist_ok=True)

    points = []
    curves = read_curves(args.k10_root)
    for ds, per_lib in sorted(curves.items()):
        path = args.out / f"{ds}_qps_recall.csv"
        write_dataset_table(path, per_lib)
        logger.info("wrote %s", path)
        for lib, rows in per_lib.items():
            for r in rows:
                points.append({"dataset": PANEL.get(ds, ds), "data_dir": ds, "system": f"{lib}_ivf_pq",
                               "param": f"{r['config']} nprobe={r['nprobe']}",
                               "recall_at_10": r["recall_at_10"], "qps": r["search_qps"]})

    bf = collect_bf(args.bf_root)
    with (args.out / "bf.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
        w.writeheader()
        w.writerows(bf)
    logger.info("wrote %s (%d datasets)", args.out / "bf.csv", len(bf))
    for r in bf:
        points.append({"dataset": PANEL.get(r["dataset"], r["dataset"]), "data_dir": r["dataset"],
                       "system": "cuvs_bf", "param": f"k={r['k']}", "recall_at_10": r["recall_at_10"],
                       "qps": r["qps_kernel"]})

    with (args.out / "nongraph_points.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "data_dir", "system", "param", "recall_at_10", "qps"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(points)
    logger.info("wrote %s (%d points)", args.out / "nongraph_points.csv", len(points))


if __name__ == "__main__":
    main()
