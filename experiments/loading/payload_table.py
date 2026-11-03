"""Table 8: shares of the one-time loading payload.

A system's payload is the base-vector file plus that system's graph, the files copied by
run_h2d_bench.sh. base_share and graph_share are those sizes divided by their sum.
graph_bytes_per_vertex is the graph size divided by the number of base vectors.

Usage: python payload_table.py --out payload.tsv [--dataset deep1M]
Requires DATA_DIR and INDEX_DIR (source env.sh).
"""
import argparse
import csv
import os
import subprocess
from pathlib import Path

FIELDS = ["dataset", "system", "n", "base_bytes", "graph_bytes", "total_bytes",
          "base_share", "graph_share", "graph_bytes_per_vertex"]


def cfg_graph(ds: str) -> list:
    sh = Path(__file__).resolve().parents[1] / "qps_recall" / "datasets.sh"
    out = subprocess.check_output(
        ["bash", "-c", 'source "$1" && cfg_graph "$2"', "cfg", str(sh), ds], text=True)
    return out.split()


def base_file(data_dir: Path, ds: str) -> Path:
    name = "base.fbin" if ds == "deep100M" else "base.fvecs"
    return data_dir / ds / name


def graph_files(index_dir: Path, ds: str, fields: list) -> dict:
    _data, _n, _dim, _deg, _nq, ctag, gtag, atag, stag, _shift = fields
    root = index_dir / ds
    files = {"cagra": [root / "cagra" / ctag / "index.cagra"]}
    if gtag != "-":
        parts = sorted((root / "ggnn" / gtag).glob("part_*.ggnn"))
        if not parts:
            raise SystemExit(f"missing GGNN graph {root / 'ggnn' / gtag}")
        files["ggnn"] = parts
    if atag != "-":
        files["ganns"] = [root / "ganns" / atag / "graph.nsw"]
    if stag != "-":
        files["song"] = [root / "song" / stag / "bfsg.graph"]
    return files


def nbytes(paths: list) -> int:
    total = 0
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"missing {path}")
        total += path.stat().st_size
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", default="deep1M")
    ap.add_argument("--data-dir", type=Path, default=None)
    ap.add_argument("--index-dir", type=Path, default=None)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    data_dir = args.data_dir or Path(os.environ["DATA_DIR"])
    index_dir = args.index_dir or Path(os.environ["INDEX_DIR"])
    fields = cfg_graph(args.dataset)
    n = int(fields[1])
    base = nbytes([base_file(data_dir, args.dataset)])
    rows = []
    for system, paths in graph_files(index_dir, args.dataset, fields).items():
        graph = nbytes(paths)
        total = base + graph
        rows.append({
            "dataset": args.dataset, "system": system, "n": n,
            "base_bytes": base, "graph_bytes": graph, "total_bytes": total,
            "base_share": f"{base / total:.6f}", "graph_share": f"{graph / total:.6f}",
            "graph_bytes_per_vertex": f"{graph / n:.4f}",
        })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
