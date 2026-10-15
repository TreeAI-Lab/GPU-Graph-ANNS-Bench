"""Collect search time and recall of every IVF-PQ and brute-force search JSON into tables.

Inputs (written by run_ivf_sweep.sh or run_ivf_k10.sh, and by run_bf.sh):
  <ivf-root>/<dataset>/{cuvs_ivf_pq,faiss_ivf_pq}/<config>/search/nprobe_*/survey_metrics_search.json
  <bf-root>/<dataset>/cuvs_bf/search/survey_metrics_search.json

kernel_ms is the mean CUDA-event time per search of the search phase, over the timed searches that
follow one untimed warm-up search (n_timed of them, 3 by default); kernel_ms_std is their sample
standard deviation:
  cuvs_bf, faiss_ivf_pq: phase "search"
  cuvs_ivf_pq:           phase "search_e2e" (ANN search plus refine; also ann_kernel_ms, refine_kernel_ms)
qps_kernel = nq / (kernel_ms / 1000); search_qps is the wall-clock value of the program (nq / mean wall
time of the same phase).
Recall columns are those of the programs (fraction of queries whose nearest neighbor is returned
within the first 1, 10, 100 ids).

Outputs in <out>: all_points.csv (with source_json), qps_recall_curve.csv (without), ivf_all.csv, bf_all.csv.

Usage: python collect_points.py --ivf-root DIR [--bf-root DIR] --out DIR
"""
import argparse
import csv
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

FIELDS = [
    "family",
    "dataset",
    "algo",
    "config",
    "nlist",
    "m",
    "nprobe",
    "k",
    "nq",
    "nb",
    "dim",
    "kernel_phase",
    "kernel_ms",
    "ann_kernel_ms",
    "refine_kernel_ms",
    "qps_kernel",
    "search_qps",
    "recall_at_1",
    "recall_at_10",
    "recall_at_100",
    "source_json",
    "kernel_ms_std",
    "n_timed",
]


def phase_ms(j: dict, name: str) -> Optional[float]:
    for p in j.get("phases") or []:
        if p.get("name") == name:
            ms = float(p.get("cuda_event_ms", -1))
            return ms if ms >= 0 else None
    return None


def phase_field(j: dict, name: str, key: str) -> str:
    """Value of `key` in phase `name` as a string; empty if the phase or the key is missing."""
    for p in j.get("phases") or []:
        if p.get("name") == name and key in p:
            return str(p[key])
    return ""


def row_from_json(path: Path, *, dataset: str, family: str, algo: str, config: str = "") -> Optional[Dict]:
    try:
        j = json.loads(path.read_text())
    except (OSError, ValueError) as e:
        logger.warning("skip bad json %s: %s", path, e)
        return None

    nq = int(j.get("nq") or 0)
    k = int(j.get("k") or 0)
    nlist = int(j.get("nlist") or 0)
    m = int(j.get("m") or 0)
    nprobe = int(j.get("nprobe") or 0)
    recall = j.get("recall") or {}

    if algo == "cuvs_ivf_pq":
        kernel_ms = phase_ms(j, "search_e2e")
        ann_ms = phase_ms(j, "search_ann")
        refine_ms = phase_ms(j, "refine")
        kernel_phase = "search_e2e"
    else:
        kernel_ms = phase_ms(j, "search")
        ann_ms = None
        refine_ms = None
        kernel_phase = "search"

    if kernel_ms is None or kernel_ms <= 0 or nq <= 0:
        logger.warning("skip missing kernel time %s", path)
        return None

    qps_kernel = nq / (kernel_ms / 1000.0)
    kernel_std = phase_field(j, kernel_phase, "cuda_event_ms_std")
    return {
        "family": family,
        "dataset": dataset,
        "algo": algo,
        "config": config,
        "nlist": nlist,
        "m": m,
        "nprobe": nprobe,
        "k": k,
        "nq": nq,
        "nb": int(j.get("nb") or 0),
        "dim": int(j.get("d") or 0),
        "kernel_phase": kernel_phase,
        "kernel_ms": f"{kernel_ms:.4f}",
        "ann_kernel_ms": f"{ann_ms:.4f}" if ann_ms is not None else "",
        "refine_kernel_ms": f"{refine_ms:.4f}" if refine_ms is not None else "",
        "qps_kernel": f"{qps_kernel:.3f}",
        "search_qps": f"{float(j.get('search_qps') or 0):.3f}",
        "recall_at_1": f"{float(recall.get('at1') or 0):.6f}",
        "recall_at_10": f"{float(recall.get('at10') or 0):.6f}",
        "recall_at_100": f"{float(recall.get('at100') or 0):.6f}",
        "source_json": str(path),
        "kernel_ms_std": f"{float(kernel_std):.4f}" if kernel_std else "",
        "n_timed": phase_field(j, kernel_phase, "n_timed"),
    }


def collect_bf(bf_root: Path) -> List[Dict]:
    rows = []
    for p in sorted(bf_root.glob("*/cuvs_bf/search/survey_metrics_search.json")):
        r = row_from_json(p, dataset=p.parts[-4], family="bf", algo="cuvs_bf", config="")
        if r:
            rows.append(r)
    return rows


def collect_ivf(ivf_root: Path) -> List[Dict]:
    rows = []
    for algo in ("cuvs_ivf_pq", "faiss_ivf_pq"):
        for p in sorted(ivf_root.glob(f"*/{algo}/*/search/nprobe_*/survey_metrics_search.json")):
            # <root>/<dataset>/<algo>/<config>/search/nprobe_X/survey_metrics_search.json
            r = row_from_json(p, dataset=p.parts[-6], family="ivf", algo=algo, config=p.parts[-4])
            if r:
                rows.append(r)
    return rows


def write_csv(path: Path, rows: List[Dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ivf-root", type=Path, required=True)
    ap.add_argument("--bf-root", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    rows = collect_ivf(args.ivf_root)
    if args.bf_root:
        rows += collect_bf(args.bf_root)
    rows.sort(key=lambda r: (r["dataset"], r["family"], r["algo"], r["config"], int(r["nprobe"])))

    write_csv(args.out / "all_points.csv", rows, FIELDS)
    write_csv(args.out / "qps_recall_curve.csv", rows, [c for c in FIELDS if c != "source_json"])
    by_family: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_family[r["family"]].append(r)
    for fam, rs in sorted(by_family.items()):
        write_csv(args.out / f"{fam}_all.csv", rs, FIELDS)
    logger.info("wrote %s: %d points (%s)", args.out, len(rows),
                ", ".join(f"{fam} {len(rs)}" for fam, rs in sorted(by_family.items())))


if __name__ == "__main__":
    main()
