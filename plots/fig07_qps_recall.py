"""Figure 7: QPS versus Recall@10 on eight datasets, graph indexes (solid) and non-graph GPU baselines (dashed).

Input: results/fig07_qps_recall.csv. Each curve drops dominated points (a point is not drawn when a point with
higher recall is at least as fast) and draws markers at least MARK_GAP apart; values are never changed.
Output: fig07_<dataset>.pdf, one panel each, and fig07_legend.pdf.
"""
import logging
import math
import os

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MultipleLocator

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATASETS = ['sift1M', 'deep1M', 'deep10M', 'deep100M', 'gist', 'glove200', 'nytimes', 'mnist8M']
MIN_RECALL = {'mnist8M': 0.84}  # every other panel starts at 0.6


def panel(d: pd.DataFrame, ds: str, out: str) -> None:
    min_recall = MIN_RECALL.get(ds, 0.6)
    d = d[(d['dataset'] == ds) & (d['recall_at_10'] >= min_recall)]
    graph = d[d['system'].isin(style.SYSTEMS)]
    curves = {}
    for system in style.SYSTEMS:
        sub = graph[graph['system'] == system].sort_values(by='recall_at_10')
        sub = sub[pd.Series(style.pareto_mask(list(sub['qps'])), index=sub.index, dtype=bool)]
        if len(sub):
            curves[system] = (list(sub['recall_at_10']), list(sub['qps']))
    base_curves = {}
    for system in style.BASELINES:
        sub = d[d['system'] == system]
        pts = sorted(zip(sub['recall_at_10'], sub['qps']))
        pts = [p for p, k in zip(pts, style.pareto_mask([q for _, q in pts])) if k]
        if pts:
            base_curves[system] = ([x for x, _ in pts], [y for _, y in pts])
    bf = d[d['system'] == 'cuvs_bf']
    x_right = graph['recall_at_10'].max()
    if len(bf):
        x_right = max(x_right, 1.005)
    all_y = [y for _, ys in list(curves.values()) + list(base_curves.values()) for y in ys]
    log_y_range = (math.log10(min(all_y)), math.log10(max(all_y)))
    x_range = (min_recall, x_right)

    plt.figure(figsize=(8, 6))
    for system, (xs, ys) in curves.items():
        plt.plot(xs, ys, marker=style.MARKER[system], markersize=11,
                 markevery=style.mark_index(xs, ys, x_range, log_y_range), markerfacecolor='none',
                 markeredgecolor=style.COLOR[system], markeredgewidth=1.5, color=style.COLOR[system], alpha=0.95,
                 linewidth=2)
    for system, (xs, ys) in base_curves.items():
        s = style.BASELINES[system]
        plt.plot(xs, ys, linestyle='--', linewidth=2, color=s['color'], marker=s['marker'], markersize=11,
                 markeredgewidth=2, markevery=style.mark_index(xs, ys, x_range, log_y_range), alpha=0.9)
    if len(bf):
        r = bf.iloc[0]
        plt.plot([r['recall_at_10']], [r['qps']], linestyle='none', marker=style.BRUTE_FORCE['marker'], markersize=20,
                 color=style.BRUTE_FORCE['color'], clip_on=False, zorder=5)
    plt.yscale('log')
    plt.xlabel('Recall@10')
    plt.ylabel('QPS')
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.gca().xaxis.set_major_locator(MultipleLocator(0.05 if min_recall > 0.8 else 0.1))
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(min_recall, x_right)
    style.fit_text(plt.gcf(), 0.23 * style.TEXT_W)  # each panel is 0.23 textwidth wide in the paper
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info('wrote %s', out)


def legend(out: str, rows: int = 2) -> None:
    """Shared legend: graph systems, then the non-graph baselines, two rows."""
    handles = [plt.Line2D([0], [0], marker=style.MARKER[s], markersize=6, markerfacecolor='none',
                          markeredgecolor=style.COLOR[s], markeredgewidth=1, color=style.COLOR[s], linewidth=1.2,
                          alpha=0.85, label=s) for s in style.SYSTEMS]
    handles += [plt.Line2D([0], [0], linestyle='--', marker=s['marker'], markersize=6, markeredgewidth=1,
                           color=s['color'], linewidth=1.2, label=s['label']) for s in style.BASELINES.values()]
    handles.append(plt.Line2D([0], [0], linestyle='none', marker=style.BRUTE_FORCE['marker'], markersize=8,
                              color=style.BRUTE_FORCE['color'], label=style.BRUTE_FORCE['label']))
    style.save_legend(handles, out, ncol=-(-len(handles) // rows), prop={'family': 'Times New Roman', 'size': 9},
                      columnspacing=0.8, handletextpad=0.3, handlelength=1.6)
    logger.info('wrote %s', out)


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig07_qps_recall.csv')
    for ds in DATASETS:
        panel(d, ds, os.path.join(args.out, f'fig07_{ds}.pdf'))
    legend(os.path.join(args.out, 'fig07_legend.pdf'))


if __name__ == '__main__':
    main()
