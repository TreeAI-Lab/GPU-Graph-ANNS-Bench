"""Merge the sweep tables of the QPS-recall runs into one summary and derive the figure data.

Reads <root>/<dataset>/<system>/sweep.tsv (and gpu_mem_cuda.tsv) for CAGRA, GGNN, GANNS, SONG, BANG and PilotANN.
Writes
  <root>/summary.tsv            one row per point: dataset, algo, it, itopk, nq, warmup_kernel_ms, h2d_query_ms,
                                kernel_ms, d2h_topk_ms, wall_ms, qps, recall@10, gpu_used_mib, host_rss_mib,
                                kernel_ms_std, wall_ms_std, n_timed (times are means over the n_timed timed runs;
                                it is L for BANG and ef for PilotANN; empty cells: not measured)
  <root>/<dataset>/summary.tsv  the rows of one dataset, without the dataset column
  <root>/fig07_points.tsv       dataset, system, param, recall_at_10, qps, excluded as plotted in Figures 1 and 7:
                                CAGRA, GGNN, GANNS, SONG: nq / kernel time; BANG, PilotANN: the qps column
                                (nq / wall time of the whole search); excluded = song_noise_filter for the SONG
                                points that Figure 7 leaves out (song_noise_filter below), empty otherwise
  <root>/fig10.tsv              system, dataset, time_us_per_10k: time per 10,000 queries of the fastest plotted
                                point with Recall@10 >= 0.9, for the six systems on DEEP1M, DEEP10M, DEEP100M
                                (empty: no point reaches 0.9 or no run)
Usage: python summarize.py --root $OUT_DIR/qps_recall
"""
import argparse
import csv
from pathlib import Path
from typing import Dict, List

DATASETS = ("deep1M", "deep10M", "glove200", "nytimes", "sift1M", "gist", "deep100M")
GPU_ALGOS = ("cagra", "ggnn", "ganns", "song")
HYBRID_ALGOS = ("pilotann", "bang")
FIELDS = ["algo", "it", "itopk", "nq", "warmup_kernel_ms", "h2d_query_ms", "kernel_ms", "d2h_topk_ms", "wall_ms",
          "qps", "recall@10", "gpu_used_mib", "host_rss_mib", "kernel_ms_std", "wall_ms_std", "n_timed"]
SYSTEM = {"cagra": "CAGRA", "ggnn": "GGNN", "ganns": "GANNS", "song": "SONG", "bang": "BANG", "pilotann": "PilotANN"}
PARAM = {"cagra": "it", "ggnn": "it", "ganns": "it", "song": "it", "bang": "L", "pilotann": "ef"}
MIN_RECALL = 0.6  # lowest recall plotted in Figure 7
DIP_FRAC, DIP_RECALL_MAX = 0.8, 0.9


def gpu_rows(root: Path, ds: str, algo: str) -> List[Dict[str, str]]:
    """Sweep rows of a GPU system; memory columns from the row or from the last gpu_mem_cuda.tsv sample."""
    p = root / ds / algo / "sweep.tsv"
    if not p.exists():
        return []
    mem = {}
    mp = root / ds / algo / "gpu_mem_cuda.tsv"
    if mp.exists():
        with mp.open() as f:
            for r in csv.DictReader(f, delimiter="\t"):
                mem[r["stage"]] = r
    last = mem.get("after_last_it") or mem.get("after_load") or {}
    rows = []
    with p.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            rec = dict(r)
            rec["algo"] = algo
            rec["gpu_used_mib"] = rec.get("gpu_used_mib") or last.get("gpu_used_mib", "")
            rec["host_rss_mib"] = last.get("host_rss_mib", "")
            if algo == "song":
                # SONG reserves a 16 GB device heap, which cudaMemGetInfo counts as used: not comparable
                rec["gpu_used_mib"] = ""
                rec["host_rss_mib"] = ""
            rows.append(rec)
    return rows


def hybrid_rows(root: Path, ds: str, algo: str) -> List[Dict[str, str]]:
    """Sweep rows of BANG or PilotANN that have a recall and a QPS."""
    p = root / ds / algo / "sweep.tsv"
    if not p.is_file():
        return []
    rows = []
    with p.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if not r.get("it") or str(r.get("recall@10", "")).startswith("FAIL"):
                continue
            try:
                float(r["recall@10"])
                float(r["qps"])
            except (TypeError, ValueError):
                continue
            rows.append({
                "algo": algo,
                "it": r["it"],
                "itopk": r.get("itopk", r["it"]),
                "nq": r.get("nq", ""),
                "warmup_kernel_ms": r.get("warmup_kernel_ms", ""),
                "h2d_query_ms": r.get("h2d_query_ms", ""),
                "kernel_ms": r.get("kernel_ms", ""),
                "d2h_topk_ms": r.get("d2h_topk_ms", ""),
                "wall_ms": r.get("wall_ms", ""),
                "qps": r["qps"],
                "recall@10": r["recall@10"],
                "gpu_used_mib": r.get("gpu_used_mib", ""),
                "host_rss_mib": r.get("host_rss_mib", ""),
                "kernel_ms_std": r.get("kernel_ms_std", ""),
                "wall_ms_std": r.get("wall_ms_std", ""),
                "n_timed": r.get("n_timed", ""),
            })
    return rows


def write(path: Path, fields: List[str], rows: List[Dict[str, str]]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} n={len(rows)}")


def plotted_qps(r: Dict[str, str]) -> float:
    if r["algo"] in GPU_ALGOS:
        return float(r["nq"]) / float(r["kernel_ms"]) * 1000.0
    return float(r["qps"])


def song_noise_filter(points: List[Dict[str, float]]) -> List[int]:
    """Indices of the SONG points of one dataset that Figure 7 plots; points holds recall and qps.

    Among the points with recall >= MIN_RECALL, sorted by recall: below recall 0.9 drop isolated dips (QPS lower
    than 0.8x the smaller neighbour), then keep only points on the running minimum of QPS. Points are dropped,
    never changed. SONG's kernel time varies between runs (it allocates memory inside the kernel).
    """
    order = sorted((i for i, p in enumerate(points) if p["recall"] >= MIN_RECALL), key=lambda i: points[i]["recall"])
    q = [points[i]["qps"] for i in order]
    r = [points[i]["recall"] for i in order]
    order = [i for j, i in enumerate(order)
             if not (r[j] < DIP_RECALL_MAX and 0 < j < len(q) - 1 and q[j] < DIP_FRAC * min(q[j - 1], q[j + 1]))]
    kept, run_min = [], float("inf")
    for i in order:
        if points[i]["qps"] <= run_min:
            kept.append(i)
        run_min = min(run_min, points[i]["qps"])
    return kept


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", required=True, type=Path, help="output folder of the runs ($OUT_DIR/qps_recall)")
    args = ap.parse_args()
    root = args.root

    all_rows = []
    for ds in DATASETS:
        rows = []
        for algo in GPU_ALGOS:
            rows += gpu_rows(root, ds, algo)
        for algo in HYBRID_ALGOS:
            rows += hybrid_rows(root, ds, algo)
        if rows:
            write(root / ds / "summary.tsv", FIELDS, rows)
            all_rows += [{"dataset": ds, **r} for r in rows]
    if not all_rows:
        raise SystemExit(f"no sweep.tsv under {root}")
    write(root / "summary.tsv", ["dataset"] + FIELDS, all_rows)

    points = []
    for r in all_rows:
        if not r.get("recall@10"):
            continue
        points.append({"dataset": r["dataset"], "system": SYSTEM[r["algo"]],
                       "param": f"{PARAM[r['algo']]}={r['it']}", "recall_at_10": r["recall@10"],
                       "qps": f"{plotted_qps(r):.2f}", "excluded": ""})
    for ds in DATASETS:
        song = [p for p in points if p["dataset"] == ds and p["system"] == "SONG"]
        kept = set(song_noise_filter([{"recall": float(p["recall_at_10"]), "qps": float(p["qps"])} for p in song]))
        for i, p in enumerate(song):
            if float(p["recall_at_10"]) >= MIN_RECALL and i not in kept:
                p["excluded"] = "song_noise_filter"
    write(root / "fig07_points.tsv", ["dataset", "system", "param", "recall_at_10", "qps", "excluded"], points)

    fig10 = []
    for algo in ("song", "ggnn", "ganns", "cagra", "bang", "pilotann"):
        for ds in ("deep1M", "deep10M", "deep100M"):
            qps = [float(p["qps"]) for p in points if p["dataset"] == ds and p["system"] == SYSTEM[algo]
                   and not p["excluded"] and float(p["recall_at_10"]) >= 0.9]
            t = f"{1e4 / max(qps) * 1e6:.0f}" if qps else ""
            fig10.append({"system": SYSTEM[algo], "dataset": ds, "time_us_per_10k": t})
    write(root / "fig10.tsv", ["system", "dataset", "time_us_per_10k"], fig10)


if __name__ == "__main__":
    main()
