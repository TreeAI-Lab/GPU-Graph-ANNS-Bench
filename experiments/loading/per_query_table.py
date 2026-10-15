"""Figure 6 table: time per query when the index is loaded once and then serves batches of 10,000 queries.

load_ms (per dataset) = mean host-to-device time of the base vectors + that of the CAGRA graph
(run_h2d_bench.sh; the same load is used for every system of a dataset).
Per steady batch i (batch_id >= 1; batch 0 is the warm-up): t_i = h2d_query_ms + kernel_ms + d2h_topk_ms.
  amortized, after n batches: (load_ms + t_1 + ... + t_n) / (n * batch size), n in 1, 5, 10, 50, 100, all
  resident: mean(t_i) / batch size
The query copies (h2d_query_ms, d2h_topk_ms) carry the same payload for every system. With
--shared-xfer SYSTEM (default cagra) they are taken from the run of SYSTEM on the same dataset and
the kernel time from each system's own run, as in the paper; --shared-xfer none uses each run's own.

Inputs:
  --h2d   summary.tsv of run_h2d_bench.sh
  --root  $OUT_DIR/loading, holding <dataset>/summary_resident_search_r95.tsv and the run directories
Output (--out): dataset,system,kind,queries_served,ms_per_query,load_ms
  (dataset is the figure panel name: deep10M-annb is shown as deep10M)

Usage: python per_query_table.py --h2d H2D.tsv --root DIR --out fig06_amortization.csv
"""
import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM = {"cagra": "CAGRA", "ggnn": "GGNN", "ganns": "GANNS", "song": "SONG"}
PANEL = {"deep10M-annb": "deep10M"}
N_BATCHES = (1, 5, 10, 50, 100)


def load_ms(h2d: Path) -> Dict[str, float]:
    base: Dict[str, float] = {}
    graph: Dict[str, float] = {}
    with h2d.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["kind"] == "dataset":
                base[r["dataset"]] = float(r["mean_ms"])
            elif r["kind"] == "graph" and r["algorithm"] == "cagra":
                graph[r["dataset"]] = float(r["mean_ms"])
    return {ds: base[ds] + graph[ds] for ds in base if ds in graph}


def steady_batches(run_dir: Path) -> List[Dict[str, float]]:
    with (run_dir / "batch_timing.tsv").open() as f:
        return [{"nq": int(r["nq"]), "xfer": float(r["h2d_query_ms"]) + float(r["d2h_topk_ms"]),
                 "kernel": float(r["kernel_ms"])}
                for r in csv.DictReader(f, delimiter="\t") if int(r["batch_id"]) > 0]


def run_dirs(ds_root: Path) -> Dict[str, Path]:
    """{system: run directory} from summary_resident_search_r95.tsv"""
    out: Dict[str, Path] = {}
    path = ds_root / "summary_resident_search_r95.tsv"
    if not path.exists():
        logger.warning("missing %s", path)
        return out
    with path.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r.get("status") == "ok":
                out[r["algo"]] = ds_root / r["algo"] / r["tag"]
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h2d", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["deep1M", "deep10M-annb", "deep100M"])
    ap.add_argument("--shared-xfer", default="cagra", help="system whose query copy times are used, or none")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    load = load_ms(args.h2d)
    rows = []
    for ds in args.datasets:
        if ds not in load:
            logger.warning("%s: no base or CAGRA graph in %s, skipped", ds, args.h2d)
            continue
        runs = {algo: steady_batches(d) for algo, d in run_dirs(args.root / ds).items()}
        shared: Optional[List[Dict[str, float]]] = runs.get(args.shared_xfer)
        if args.shared_xfer != "none" and shared is None:
            logger.warning("%s: no %s run, each system uses its own query copy times", ds, args.shared_xfer)
        for algo in SYSTEM:
            if algo not in runs:
                continue
            batches = runs[algo]
            xfer_src = shared if shared is not None and len(shared) >= len(batches) else batches
            per_batch = [xfer_src[i]["xfer"] + b["kernel"] for i, b in enumerate(batches)]
            bsz = batches[0]["nq"]
            for nb in N_BATCHES + (len(batches),):
                if nb > len(batches):
                    continue
                rows.append({"dataset": PANEL.get(ds, ds), "system": SYSTEM[algo], "kind": "amortized",
                             "queries_served": nb * bsz,
                             "ms_per_query": (load[ds] + sum(per_batch[:nb])) / (nb * bsz), "load_ms": load[ds]})
            rows.append({"dataset": PANEL.get(ds, ds), "system": SYSTEM[algo], "kind": "resident",
                         "queries_served": "", "ms_per_query": sum(per_batch) / len(per_batch) / bsz,
                         "load_ms": load[ds]})
            logger.info("%s %s: %d steady batches, load %.2f ms", ds, SYSTEM[algo], len(batches), load[ds])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "system", "kind", "queries_served", "ms_per_query", "load_ms"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", args.out, len(rows))


if __name__ == "__main__":
    main()
