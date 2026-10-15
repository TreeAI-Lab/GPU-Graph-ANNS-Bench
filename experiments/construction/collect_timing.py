"""Summary of run_timing.sh, generated from the graph directories (never edited by hand).

Reads selected_configs.tsv and <graphs_root>/<dataset>/<system>/<tag>_verify and <tag>_rep<N>, and writes
  verify.tsv  per verified configuration: distance computations at Recall@10 = 0.9 in the parameter search and on
              the rebuilt graph, whether the latter does not exceed B, and the CPU QPS at that point
  timing.tsv  per configuration that passed: self-reported build time and process wall time (mean and standard
              deviation over the timed builds), peak host and GPU memory, graph statistics, and the distance
              computations and CPU QPS at Recall@10 = 0.9
The QPS comes from the L sweep of the verification build: the common searcher with THREADS CPU threads over all
queries, QPS = nq / search_s, interpolated like the distance computations between the two L around Recall 0.9.
search_s is the mean time of the timed passes after one warm-up pass; qps_n_timed is the number of timed passes
(NA if the sweep does not record it).
Usage: python collect_timing.py <selected_configs.tsv> <graphs_root> <out_dir>
"""
import argparse
import csv
import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src" / "build"))
sys.path.insert(0, str(ROOT / "src" / "unified_search"))
from operating_point import ndist_at, qps_at  # noqa: E402
from tags import tag_of  # noqa: E402

# Build time reported by each builder (seconds): from the data in host memory to the finished graph, without file
# I/O. SONG reports none; its process wall time is used.
SELF_REPORT = {
    "cagra": r"CAGRA_BUILD_SECONDS ([\d.]+)",
    "ggnn": r"GGNN_BUILD_SECONDS ([\d.]+)",
    "pilotann": r"PILOT_BUILD_SECONDS ([\d.]+)",
    "hnsw": r"HNSW_BUILD_SECONDS ([\d.]+)",
    "ganns": r"Running time: ([\d.]+) seconds",
    "vamana": r"Indexing time: ([\d.]+)",
}


def wall_seconds(log: str) -> float | None:
    m = re.search(r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)", log)
    if not m:
        return None
    parts = [float(x) for x in m.group(1).split(":")]
    return sum(p * 60 ** i for i, p in enumerate(reversed(parts)))


def build_seconds(system: str, log: str) -> float | None:
    if system == "song":
        return wall_seconds(log)
    m = re.search(SELF_REPORT[system], log)
    return float(m.group(1)) if m else None


def n_timed(search_tsv: Path) -> str:
    """Number of timed passes per L recorded in a sweep of the common searcher."""
    first = next(csv.DictReader(open(search_tsv), delimiter="\t"), None)
    return (first or {}).get("n_timed") or "NA"


def peak_gpu_mb(d: Path) -> float | None:
    f, b = d / "gpu_mem.txt", d / "gpu_mem_baseline.txt"
    if not f.exists():
        return None
    vals = [float(x) for x in f.read_text().split() if x.strip()]
    base = float(b.read_text().split()[0]) if b.exists() and b.read_text().strip() else 0.0
    return max(vals) - base if vals else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("configs", type=Path)
    ap.add_argument("graphs", type=Path)
    ap.add_argument("out", type=Path)
    a = ap.parse_args()
    cfg, root, out = a.configs, a.graphs, a.out
    ver, tim = [], []
    for r in csv.DictReader(open(cfg), delimiter="\t"):
        ds, sysn, b = r["dataset"], r["system"], float(r["B"])
        chosen = None
        for label, args in (("selected", r["build_args"]), ("fallback", r["fallback_args"])):
            if args == "-":
                continue
            d = root / ds / sysn / (tag_of(sysn, args) + "_verify")
            if not (d / "search.tsv").exists():
                continue
            nd = ndist_at(d / "search.tsv")
            ok = nd is not None and nd <= b
            qps = qps_at(d / "search.tsv")
            ver.append([ds, sysn, label, args, r["ndist_at_r90"] if label == "selected" else "", f"{nd:.0f}" if nd else "NA",
                        f"{b:.0f}", "PASS" if ok else "FAIL", f"{qps:.0f}" if qps else "NA",
                        n_timed(d / "search.tsv")])
            if ok and chosen is None:
                chosen = (args, d)
        if chosen is None:
            continue
        args, vdir = chosen
        st = list(csv.DictReader(open(vdir / "stats.tsv"), delimiter="\t"))[0]
        reps = sorted(root.glob(f"{ds}/{sysn}/{tag_of(sysn, args)}_rep*"))
        bs, ws, rss, gpu = [], [], [], []
        for d in reps:
            log = (d / "build.log").read_text(errors="ignore") if (d / "build.log").exists() else ""
            x = build_seconds(sysn, log)
            if x is None:
                continue
            bs.append(x)
            ws.append(wall_seconds(log) or float("nan"))
            m = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
            rss.append(int(m.group(1)) / 2**20 if m else float("nan"))
            g = peak_gpu_mb(d)
            if g is not None:
                gpu.append(g / 1024)
        fmt = lambda v: f"{statistics.mean(v):.2f}" if v else "NA"
        sd = lambda v: f"{statistics.stdev(v):.2f}" if len(v) > 1 else "NA"
        tim.append([ds, sysn, args, str(len(bs)), fmt(bs), sd(bs), fmt(ws), sd(ws),
                    fmt(rss), fmt(gpu), f"{float(st['avg_unique_out_degree']):.2f}", st["weak_cc"],
                    f"{ndist_at(vdir / 'search.tsv'):.0f}", f"{qps_at(vdir / 'search.tsv'):.0f}",
                    n_timed(vdir / "search.tsv")])
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "verify.tsv", "w") as f:
        f.write("dataset\tsystem\twhich\tbuild_args\tndist_sweep\tndist_verify\tB\tresult\tqps_verify\tqps_n_timed\n")
        f.writelines("\t".join(x) + "\n" for x in ver)
    with open(out / "timing.tsv", "w") as f:
        f.write("dataset\tsystem\tbuild_args\treps\tbuild_s_mean\tbuild_s_std\twall_s_mean\twall_s_std\tpeak_host_gb\tpeak_gpu_gb\tuniq_out_degree\tweak_cc\tndist_at_r90\tqps_at_r90\tqps_n_timed\n")
        f.writelines("\t".join(x) + "\n" for x in tim)
    print(f"wrote {out}/verify.tsv ({len(ver)} rows) and timing.tsv ({len(tim)} rows)")


if __name__ == "__main__":
    main()
