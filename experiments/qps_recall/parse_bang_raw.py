"""Parse the output of a bang_search L sweep (raw.log) into sweep.tsv with the columns of the other sweeps.

bang_search runs every L once as an untimed warm-up and then three times timed. It prints a tagged line after each
run ("[warmup 1/1] ..." or "[timed i/3] ..."), each preceded by the STATS block of that search call, and after the
timed runs the summary row: L, mean wall ms, QPS = nq / mean wall time, mean Recall@10 in percent, standard
deviation of the wall time, number of timed runs. One output row per L: the summary values, and the stage times and
counters of the STATS blocks averaged over the timed runs (the warm-up block is discarded).
Usage: python parse_bang_raw.py --raw raw.log --out <dir> --nq <number of queries>
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

# Summary row per L: L, mean wall ms, QPS, recall (percent), wall ms standard deviation, number of timed runs
ROW_RE = re.compile(
    r"^\s*(\d+)\s+([\d.]+)\s+([\d.]+|inf)\s+([\d.]+)\s+([\d.]+)\s+(\d+)\s*$"
)
# Line printed after each search call
TAG_RE = re.compile(r"^\[(warmup|timed) \d+/\d+\] L=(\d+) ")
# Stage times of the STATS block printed by every search call
STAT_KEYS = {
    "PD Dist Table Construction": "pq_table_ms",
    "Distance Computations": "gpu_distance_ms",
    "Sort and Merge": "sort_merge_ms",
    "total neighbor_filtering_time": "neighbor_filter_ms",
    "Pre-fetch time": "prefetch_ms",
    "Time elapsed in L2 Dist computation (GPU)": "l2_rerank_gpu_ms",
    "total transfer_time (CPU <--> GPU)": "transfer_ms",
    "total neigbbour seek time": "neighbor_seek_cpu_ms",
    "Total Search iterations": "iterations",
}
# Counters, averaged over the timed runs like the times
COUNTERS = ("iterations", "calc_count", "path_length")


def mean_of(values: list) -> float | int:
    """Mean of the values; an int when all values are ints and the mean is a whole number."""
    m = sum(values) / len(values)
    if all(isinstance(v, int) for v in values) and m == int(m):
        return int(m)
    return m


def average_runs(runs: list[dict]) -> dict:
    """Per key, the mean over the timed runs that report it."""
    keys = {k for r in runs for k in r}
    return {k: mean_of([r[k] for r in runs if k in r]) for k in keys}


def parse(raw: str, nq: int) -> list[dict]:
    rows = []
    pending: dict = {}
    timed: list[dict] = []
    for line in raw.splitlines():
        if "now in bang_query" in line:
            pending = {}
        for needle, key in STAT_KEYS.items():
            if needle in line and "=" in line:
                try:
                    pending[key] = float(line.split("=")[-1].strip().split()[0])
                except (IndexError, ValueError):
                    pass
        if "calc_count=" in line:
            m = re.search(r"calc_count=(\d+),\s*path_length=(\d+)", line)
            if m:
                pending["calc_count"] = int(m.group(1))
                pending["path_length"] = int(m.group(2))
        t = TAG_RE.match(line)
        if t:
            if t.group(1) == "timed":
                timed.append(pending)
            pending = {}
            continue
        m = ROW_RE.match(line)
        if not m:
            continue
        L = int(m.group(1))
        rec = float(m.group(4))
        if rec > 1.5:  # percent
            rec /= 100.0
        row = {
            "it": L,
            "itopk": L,
            "nq": nq,
            "wall_ms": float(m.group(2)),
            "qps": float(m.group(3)),
            "recall@10": rec,
            **(average_runs(timed) if timed else {}),
            "wall_ms_std": float(m.group(5)),
            "n_timed": int(m.group(6)),
        }
        if len(timed) != row["n_timed"]:
            print(f"warning: L={L}: {len(timed)} STATS blocks for {row['n_timed']} timed runs")
        rows.append(row)
        timed = []
        pending = {}
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--nq", type=int, required=True)
    args = ap.parse_args()

    raw = Path(args.raw).read_text(errors="ignore")
    rows = parse(raw, args.nq)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sweep = out / "sweep.tsv"
    fields = [
        "it",
        "itopk",
        "nq",
        "warmup_kernel_ms",
        "h2d_query_ms",
        "kernel_ms",
        "d2h_topk_ms",
        "wall_ms",
        "qps",
        "recall@10",
        "gpu_used_mib",
        "host_rss_mib",
        "iterations",
        "calc_count",
        "path_length",
        "pq_table_ms",
        "gpu_distance_ms",
        "sort_merge_ms",
        "neighbor_filter_ms",
        "prefetch_ms",
        "l2_rerank_gpu_ms",
        "transfer_ms",
        "neighbor_seek_cpu_ms",
        "wall_ms_std",
        "n_timed",
    ]
    with sweep.open("w") as f:
        f.write("\t".join(fields) + "\n")
        for r in rows:
            vals = []
            for k in fields:
                v = r.get(k, "")
                if k == "recall@10" and v != "":
                    vals.append(f"{float(v):.6f}")
                elif isinstance(v, float) and k.endswith(("_ms", "_ms_std")):
                    vals.append(f"{v:.4f}")
                elif isinstance(v, float) and k in COUNTERS:
                    vals.append(f"{v:.0f}" if v == int(v) else f"{v:.2f}")
                elif isinstance(v, float):
                    vals.append(f"{v}")
                else:
                    vals.append("" if v == "" else str(v))
            f.write("\t".join(vals) + "\n")
    print(f"parsed {len(rows)} rows -> {sweep}")


if __name__ == "__main__":
    main()
