"""Parse the output of bang_search for one batch size (raw.log) into sweep.tsv with BANG's stage times.

bang_search runs every L once as an untimed warm-up and then three times timed. The log is cut at every
"now in bang_query" (one search call); each call ends with a tagged line, "[warmup 1/1] ..." or "[timed i/3] ...",
and the call of the last timed run also holds the summary row of the L (L, mean wall ms, QPS, mean recall in
percent, wall ms standard deviation, number of timed runs). One output row per L, from the timed calls only:
wall_ms is the mean of the "Wall Clock Time" that bang_search.cu prints for each call, qps = nq / mean wall_ms,
qps_no_transfer = nq / (mean wall_ms - mean transfer_ms), the stage times and counters are means over the timed
calls, recall is that of the summary row, wall_ms_std is the sample standard deviation of the timed wall times and
n_timed the number of timed calls.
Columns: L nq recall wall_ms qps qps_no_transfer iterations calc_count path_length pq_table_ms gpu_distance_ms
sort_merge_ms neighbor_filter_ms prefetch_ms l2_rerank_gpu_ms transfer_ms neighbor_seek_cpu_ms timer_sum_ms
result_d2h_us wall_ms_std n_timed
Usage: python parse_bang_batch.py --raw raw.log --out sweep.tsv --nq <batch size>
"""
import argparse
import re
import statistics
from pathlib import Path
from typing import Dict, List

PATS = [
    ("iterations", r"Total Search iterations = (\d+)"),
    ("pq_table_ms", r"\(1\) PD Dist Table Construction = ([0-9.]+)"),
    ("gpu_distance_ms", r"\(2\) Distance Computations = ([0-9.]+)"),
    ("sort_merge_ms", r"\(3\) Sort and Merge = ([0-9.]+)"),
    ("neighbor_filter_ms", r"\(4\) total neighbor_filtering_time = ([0-9.]+)"),
    ("prefetch_ms", r"\(5\) Pre-fetch time = ([0-9.]+)"),
    ("l2_rerank_gpu_ms", r"\(6\) Time elapsed in L2 Dist computation \(GPU\)= ([0-9.]+)"),
    ("transfer_ms", r"\(7\) total transfer_time \(CPU <--> GPU\) = ([0-9.]+)"),
    ("neighbor_seek_cpu_ms", r"\(8\) total neigbbour seek time = ([0-9.]+)"),
    ("timer_sum_ms", r"Total time from timers = .* = ([0-9.]+) ms"),
    ("wall_ms", r"Wall Clock Time = ([0-9.]+)"),
    ("calc_count", r"calc_count=(\d+)"),
    ("path_length", r"path_length=(\d+)"),
    ("result_d2h_us", r"Time to transfer final results back to CPU = (\d+)"),
]
COUNTS = ("iterations", "calc_count", "path_length", "result_d2h_us")
TAG_RE = re.compile(r"^\[(warmup|timed) \d+/\d+\] L=(\d+) ", re.M)
SUM_RE = re.compile(r"^(\d+)\t([0-9.]+)\t([0-9.]+|inf)\t([0-9.]+)\t([0-9.]+)\t(\d+)\s*$", re.M)
COLS = ["L", "nq", "recall", "wall_ms", "qps", "qps_no_transfer", "iterations", "calc_count", "path_length",
        "pq_table_ms", "gpu_distance_ms", "sort_merge_ms", "neighbor_filter_ms", "prefetch_ms",
        "l2_rerank_gpu_ms", "transfer_ms", "neighbor_seek_cpu_ms", "timer_sum_ms", "result_d2h_us",
        "wall_ms_std", "n_timed"]


def fmt(key: str, values: List[float]) -> str:
    """Mean of one field over the timed calls; empty when a call did not print it."""
    if not values:
        return ""
    m = statistics.mean(values)
    if key in COUNTS:
        return f"{m:.0f}" if m == int(m) else f"{m:.2f}"
    return f"{m:.4f}"


def summary_row(timed: List[Dict[str, str]], L: str, nq: int, recall_pct: str) -> List[str]:
    vals = {k: [float(r[k]) for r in timed if r.get(k)] for k, _ in PATS}
    if len(vals["wall_ms"]) != len(timed):
        return []
    wall = statistics.mean(vals["wall_ms"])
    transfer = statistics.mean(vals["transfer_ms"]) if vals["transfer_ms"] else 0.0
    qps = f"{nq * 1000.0 / wall:.2f}" if wall > 0 else ""
    qps_nt = f"{nq * 1000.0 / (wall - transfer):.2f}" if wall - transfer > 0 else ""
    std = statistics.stdev(vals["wall_ms"]) if len(timed) > 1 else 0.0
    return [L, str(nq), f"{float(recall_pct) / 100.0:.6f}", f"{wall:.4f}", qps, qps_nt,
            fmt("iterations", vals["iterations"]), fmt("calc_count", vals["calc_count"]),
            fmt("path_length", vals["path_length"]), fmt("pq_table_ms", vals["pq_table_ms"]),
            fmt("gpu_distance_ms", vals["gpu_distance_ms"]), fmt("sort_merge_ms", vals["sort_merge_ms"]),
            fmt("neighbor_filter_ms", vals["neighbor_filter_ms"]), fmt("prefetch_ms", vals["prefetch_ms"]),
            fmt("l2_rerank_gpu_ms", vals["l2_rerank_gpu_ms"]), fmt("transfer_ms", vals["transfer_ms"]),
            fmt("neighbor_seek_cpu_ms", vals["neighbor_seek_cpu_ms"]), fmt("timer_sum_ms", vals["timer_sum_ms"]),
            fmt("result_d2h_us", vals["result_d2h_us"]), f"{std:.4f}", str(len(timed))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--nq", required=True, type=int)
    args = ap.parse_args()
    text = args.raw.read_text(errors="replace")
    lines = ["\t".join(COLS)]
    timed: List[Dict[str, str]] = []
    for part in text.split("now in bang_query")[1:]:
        tag = TAG_RE.search(part)
        if tag is None:
            continue
        if tag.group(1) == "timed":
            rec = {}
            for k, pat in PATS:
                m = re.search(pat, part, re.M)
                rec[k] = m.group(1) if m else ""
            timed.append(rec)
        m = SUM_RE.search(part)
        if not m:
            continue
        L, _, _, recp, _, n_timed = m.groups()
        if len(timed) != int(n_timed):
            print(f"warning: L={L}: {len(timed)} timed calls in the log, the summary row says {n_timed}")
        row = summary_row(timed, L, args.nq, recp)
        if row:
            lines.append("\t".join(row))
        timed = []
    args.out.write_text("\n".join(lines) + "\n")
    print(args.out, len(lines) - 1, lines[-1][:80])


if __name__ == "__main__":
    main()
