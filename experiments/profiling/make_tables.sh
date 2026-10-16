#!/bin/bash
# Write the LaTeX body rows of Table 9 (tab:profiling) and Table 10 (tab:fp16) from summary.tsv (Nsight Compute)
# and table10_qps/qps.tsv (mean QPS). All numbers come from these two files.
# Table 9: one group row per dataset; the last column is the share of the long-scoreboard stall. The script stops
# with an error if long scoreboard is not the main stall of a row.
# Table 10: the FP16 rows give the speedup in parentheses (FP16 mean QPS / FP32 mean QPS of the same system);
# the last column is the long-scoreboard share, again checked to be the main stall.
# usage: bash make_tables.sh > $OUT_DIR/profiling/tables.tex
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/../../env.sh"
S=${SUMMARY:-$OUT_DIR/profiling/summary.tsv}
Q=${QPS:-$OUT_DIR/profiling/table10_qps/qps.tsv}

awk -F'\t' -v Q="$Q" '
function cite(x) { return (x in NAME) ? NAME[x] : x }
function stall(x) { gsub("_", " ", x); return toupper(substr(x, 1, 1)) substr(x, 2) }
function thousands(n,   s, out) {
  s = sprintf("%d", n); out = ""
  while (length(s) > 3) { out = "," substr(s, length(s) - 2) out; s = substr(s, 1, length(s) - 3) }
  return s out
}
function mb(b) { b = b / 1e6; return (b >= 10) ? sprintf("%.1f", b) : sprintf("%.2f", b) }
BEGIN {
  NAME["gist"] = "GIST~\\cite{herve2011pq}"; NAME["deep1M"] = "DEEP1M~\\cite{DEEP}"
  NAME["song"] = "SONG~\\cite{2020SONG}"; NAME["ggnn"] = "GGNN~\\cite{2022GGNN}"
  NAME["ganns"] = "GANNS~\\cite{2022GANNS}"; NAME["cagra"] = "CAGRA~\\cite{2024CAGRA}"
  NAME["bang"] = "BANG~\\cite{2024BANG}"; NAME["pilotann"] = "PilotANN~\\cite{2025PilotANN}"
  while ((getline line < Q) > 0) {
    split(line, f, "\t")
    if (f[1] == "dataset") continue
    k = f[1] SUBSEP f[2] SUBSEP f[3]; qps[k] = f[6]; rec[k] = f[5]
  }
}
NR == 1 { for (i = 1; i <= NF; i++) c[$i] = i; next }
{ k = $c["dataset"] SUBSEP $c["system"] SUBSEP $c["precision"]; for (i = 1; i <= NF; i++) v[k, i] = $i; have[k] = 1 }
END {
  nd = split("gist deep1M", D, " ")
  print "% Table 9 (tab:profiling)"
  ns = split("song ggnn ganns cagra bang pilotann", T, " ")
  for (d = 1; d <= nd; d++) {
    printf "\t\t\\multicolumn{7}{@{}l}{\\textit{%s}} \\\\\n\t\t\\hline\n", cite(D[d])
    for (s = 1; s <= ns; s++) {
      k = D[d] SUBSEP T[s] SUBSEP "fp32"
      if (!have[k]) continue
      if (v[k, c["stall"]] != "long_scoreboard") { print "table 9: main stall of " D[d] " " T[s] " is " v[k, c["stall"]] > "/dev/stderr"; exit 1 }
      printf "\t\t%s & %s\\%% & %s\\%% & %s\\%% & %s & %s\\%% & %s\\%% \\\\\n", cite(T[s]),
        v[k, c["SM"]], v[k, c["GPU DRAM"]], v[k, c["Occ."]], v[k, c["Thr."]], v[k, c["L2"]], v[k, c["stall_share_pct"]]
    }
    print "\t\t\\hline"
  }
  print "% Table 10 (tab:fp16)"
  ns = split("song ggnn ganns cagra", T, " ")
  np = split("fp32 fp16", P, " ")
  for (d = 1; d <= nd; d++) {
    for (s = 1; s <= ns; s++) {
      for (p = 1; p <= np; p++) {
        k = D[d] SUBSEP T[s] SUBSEP P[p]
        lead = (s == 1 && p == 1) ? "\\multirow{8}{*}{" cite(D[d]) "}" : ""
        sysc = (p == 1) ? "\\multirow{2}{*}{" cite(T[s]) "}" : ""
        k32 = D[d] SUBSEP T[s] SUBSEP "fp32"
        q = thousands(qps[k]) ((P[p] == "fp16") ? sprintf(" (%.2f$\\times$)", qps[k] / qps[k32]) : "")
        if (v[k, c["stall"]] != "long_scoreboard") { print "table 10: main stall of " D[d] " " T[s] " " P[p] " is " v[k, c["stall"]] > "/dev/stderr"; exit 1 }
        printf "\t\t%s & %s & %s & %.4f & %s & %s & %s\\%% & %s\\%% & %s\\%% & %s\\%% \\\\\n", lead, sysc, toupper(P[p]),
          rec[k], q, mb(v[k, c["dram_read_bytes_per_query"]]),
          v[k, c["SM"]], v[k, c["GPU DRAM"]], v[k, c["L2"]], v[k, c["stall_share_pct"]]
      }
      print (s < ns) ? "\t\t\\cline{2-10}" : "\t\t\\hline"
    }
  }
}' "$S"
