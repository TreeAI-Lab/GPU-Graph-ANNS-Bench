"""Figure 9 table: time per query when the index is loaded once and then serves batches of 10,000 queries.

load_ms = mean host-to-device time of the base vectors plus a graph (run_h2d_bench.sh).
On deep1M and deep10M each system uses its own graph (GGNN parts are summed). On deep100M
every system uses the CAGRA graph.
Per steady batch i (batch_id >= 1; batch 0 is the warm-up): t_i = h2d_query_ms + kernel_ms + d2h_topk_ms.
  amortized, after n batches: (load_ms + t_1 + ... + t_n) / (n * batch size), n in 1, 5, 10, 50, 100, all
  resident: mean(t_i) / batch size
The query copies (h2d_query_ms, d2h_topk_ms) carry the same payload for every system. With
--shared-xfer SYSTEM (default cagra) they are taken from the run of SYSTEM on the same dataset and
the kernel time from each system's own run (the default); --shared-xfer none uses each run's own.
HNSW_20 stays on the host, so it has no load: amortized and resident are both 1000 / QPS at the smallest
efSearch that reaches 95% Recall@10 with 20 threads, on deep1M and deep10M.

Inputs:
  --h2d   summary.tsv of run_h2d_bench.sh
  --root  $OUT_DIR/loading, holding <dataset>/summary_resident_search_r95.tsv and the run directories
  --hnsw-root  optional, $OUT_DIR/equal_degree/hnsw, holding <dataset>/k10/t20.log of equal_degree/run_hnsw.sh
Output (--out): dataset,system,kind,queries_served,ms_per_query,load_ms

Usage: python per_query_table.py --h2d H2D.tsv --root DIR --out fig09_amortization.csv
"""
import argparse
import csv
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)

SYSTEM = {"cagra": "CAGRA", "ggnn": "GGNN", "ganns": "GANNS", "song": "SONG"}
N_BATCHES = (1, 5, 10, 50, 100)
HNSW_DATASETS = ("deep1M", "deep10M")
HNSW_SERVED = (10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000)  # the queries served of the GPU curves
HNSW_COLS = ["threads", "batch", "efSearch", "k", "nq", "n_batches", "wall_ms", "qps", "recall@10", "wall_ms_std",
             "n_timed"]  # fields after "row" in the output of faiss_hnsw search
TARGET = 0.95


def load_ms(h2d: Path) -> Dict[str, Dict[str, float]]:
    base: Dict[str, float] = {}
    graph: Dict[tuple, float] = {}
    with h2d.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["kind"] == "dataset":
                base[r["dataset"]] = float(r["mean_ms"])
            elif r["kind"] == "graph":
                key = (r["dataset"], r["algorithm"])
                graph[key] = graph.get(key, 0.0) + float(r["mean_ms"])
    out: Dict[str, Dict[str, float]] = {}
    for (ds, algo), ms in graph.items():
        if ds in base:
            out.setdefault(ds, {})[algo] = base[ds] + ms
    return out


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


def hnsw_ms_per_query(log: Path) -> Optional[float]:
    """Time per query at the smallest efSearch that reaches TARGET in a faiss_hnsw log, or None."""
    rows = [dict(zip(HNSW_COLS, line.split("\t")[1:])) for line in log.read_text().splitlines()
            if line.startswith("row\t")]
    ok = [r for r in rows if float(r["recall@10"]) >= TARGET]
    if not ok:
        return None
    return 1000.0 / float(min(ok, key=lambda r: int(r["efSearch"]))["qps"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--h2d", type=Path, required=True)
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--datasets", nargs="+", default=["deep1M", "deep10M", "deep100M"])
    ap.add_argument("--shared-xfer", default="cagra", help="system whose query copy times are used, or none")
    ap.add_argument("--hnsw-root", type=Path, help="$OUT_DIR/equal_degree/hnsw, adds HNSW_20")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    load = load_ms(args.h2d)
    rows = []
    for ds in args.datasets:
        if ds not in load:
            logger.warning("%s: no base copy in %s, skipped", ds, args.h2d)
            continue
        runs = {algo: steady_batches(d) for algo, d in run_dirs(args.root / ds).items()}
        shared: Optional[List[Dict[str, float]]] = runs.get(args.shared_xfer)
        if args.shared_xfer != "none" and shared is None:
            logger.warning("%s: no %s run, each system uses its own query copy times", ds, args.shared_xfer)
        for algo in SYSTEM:
            if algo not in runs:
                continue
            if algo in load[ds]:
                one_load = load[ds][algo]
            elif "cagra" in load[ds]:
                one_load = load[ds]["cagra"]
            else:
                logger.warning("%s %s: no graph copy in %s, skipped", ds, algo, args.h2d)
                continue
            batches = runs[algo]
            xfer_src = shared if shared is not None and len(shared) >= len(batches) else batches
            per_batch = [xfer_src[i]["xfer"] + b["kernel"] for i, b in enumerate(batches)]
            bsz = batches[0]["nq"]
            for nb in N_BATCHES + (len(batches),):
                if nb > len(batches):
                    continue
                rows.append({"dataset": ds, "system": SYSTEM[algo], "kind": "amortized",
                             "queries_served": nb * bsz,
                             "ms_per_query": (one_load + sum(per_batch[:nb])) / (nb * bsz), "load_ms": one_load})
            rows.append({"dataset": ds, "system": SYSTEM[algo], "kind": "resident",
                         "queries_served": "", "ms_per_query": sum(per_batch) / len(per_batch) / bsz,
                         "load_ms": one_load})
            logger.info("%s %s: %d steady batches, load %.2f ms", ds, SYSTEM[algo], len(batches), one_load)
        log = args.hnsw_root / ds / "k10" / "t20.log" if args.hnsw_root else None
        if log is None or ds not in HNSW_DATASETS:
            continue
        ms = hnsw_ms_per_query(log) if log.exists() else None
        if ms is None:
            logger.warning("%s HNSW_20: no point reaches Recall@10 %.2f in %s, skipped", ds, TARGET, log)
            continue
        for served in HNSW_SERVED:
            rows.append({"dataset": ds, "system": "HNSW_20", "kind": "amortized", "queries_served": served,
                         "ms_per_query": ms, "load_ms": 0.0})
        rows.append({"dataset": ds, "system": "HNSW_20", "kind": "resident", "queries_served": "",
                     "ms_per_query": ms, "load_ms": 0.0})

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "system", "kind", "queries_served", "ms_per_query", "load_ms"],
                           lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    logger.info("wrote %s (%d rows)", args.out, len(rows))


if __name__ == "__main__":
    main()
