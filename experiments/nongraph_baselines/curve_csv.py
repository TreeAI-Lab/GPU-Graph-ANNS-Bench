"""Collect the search JSON files of one IVF-PQ index (one per nprobe) into one curve CSV.

Usage: python curve_csv.py <search_root> <out_csv> <nlist> <m> <k> <algo> <config>
  search_root/nprobe_<n>/survey_metrics_search.json  ->  out_csv with columns
  algo,config,nprobe,recall_at_1,recall_at_10,recall_at_100,search_qps,nlist,m,k,search_wall_ms_std,n_timed
search_qps = nq / mean wall time of the timed searches (after one untimed warm-up); search_wall_ms_std
is the sample standard deviation of their wall time and n_timed their number, both from the phase that
gives search_qps (search_e2e for cuvs_ivf_pq, search otherwise).
"""
import glob
import json
import os
import sys
from typing import List, Tuple


def qps_phase(j: dict, algo: str) -> dict:
    """The phase whose mean wall time gives search_qps; empty dict if missing."""
    name = "search_e2e" if algo == "cuvs_ivf_pq" else "search"
    for p in j.get("phases") or []:
        if p.get("name") == name:
            return p
    return {}


def collect(search_root: str, nlist: int, m: int, k: int, algo: str, cfg: str) -> List[Tuple]:
    rows = []
    for path in sorted(glob.glob(os.path.join(search_root, "nprobe_*", "survey_metrics_search.json"))):
        try:
            with open(path) as f:
                j = json.load(f)
        except (OSError, ValueError) as e:
            print("skip bad json", path, e, file=sys.stderr)
            continue
        nprobe = j.get("nprobe")
        if nprobe is None:
            nprobe = int(os.path.basename(os.path.dirname(path)).split("_")[-1])
        r = j.get("recall", {})
        p = qps_phase(j, algo)
        wall_std = f"{float(p['wall_s_std']) * 1000.0:.4f}" if "wall_s_std" in p else ""
        rows.append((
            algo, cfg, int(nprobe),
            float(r.get("at1", 0)), float(r.get("at10", 0)), float(r.get("at100", 0)),
            float(j.get("search_qps", 0)),
            nlist, m, k,
            wall_std, p.get("n_timed", ""),
        ))
    rows.sort(key=lambda x: x[2])
    return rows


def main() -> None:
    search_root, out_csv, nlist, m, k, algo, cfg = sys.argv[1:8]
    rows = collect(search_root, int(nlist), int(m), int(k), algo, cfg)
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w") as f:
        f.write("algo,config,nprobe,recall_at_1,recall_at_10,recall_at_100,search_qps,nlist,m,k,"
                "search_wall_ms_std,n_timed\n")
        for row in rows:
            f.write(",".join(map(str, row)) + "\n")
    print("wrote", out_csv, "rows", len(rows))


if __name__ == "__main__":
    main()
