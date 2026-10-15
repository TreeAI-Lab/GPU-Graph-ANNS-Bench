"""Figure 5: traversal depth at 90% Recall@10. (a) search iterations of each system's own search;
(b) hops of one common search procedure run on every system's graph.

Input: results/fig05_traversal_depth.csv. Output: fig05a_search_iterations.pdf, fig05b_common_search_hops.pdf.
Both panels share the y range so that bars can be compared side by side.
"""
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

DATASETS = ['GIST', 'DEEP1M', 'DEEP10M', 'SIFT1M']
# (system in the results file, legend label, color)
BARS_A = [('SONG', 'SONG', '#8A95D5'), ('GGNN', 'GGNN', '#F27066'), ('GANNS', 'GANNS', '#A446E8'),
          ('CAGRA', 'CAGRA', '#808080'), ('BANG', 'BANG', '#3DBDD6'), ('PilotANN', 'PilotANN', '#B983CD'),
          ('Vamana', 'Vamana', '#F4A3D0'), ('HNSW', 'HNSW', '#5CB8B0')]
# BANG searches the Vamana graph, so the common search gives one bar for both
BARS_B = [('SONG', 'SONG', '#8A95D5'), ('GGNN', 'GGNN', '#F27066'), ('GANNS', 'GANNS', '#A446E8'),
          ('CAGRA', 'CAGRA', '#808080'), ('BANG/Vamana', 'BANG / Vamana', '#3DBDD6'),
          ('PilotANN', 'PilotANN', '#B983CD'), ('HNSW', 'HNSW', '#5CB8B0')]
BAR_W = 0.11
STAR = dict(marker='*', color='gold', markersize=19, markeredgecolor='orange', markeredgewidth=1.5)


def draw(d: pd.DataFrame, bars: list, ylabel: str, ncol: int, out: str) -> None:
    values = {(r.dataset, r.system): r.hops_per_query for r in d.itertuples()}
    x = np.arange(len(DATASETS))
    plt.figure(figsize=(12, 7))
    ax = plt.gca()
    per_ds = {ds: [] for ds in DATASETS}
    for i, (system, label, color) in enumerate(bars):
        for j, ds in enumerate(DATASETS):
            if (ds, system) not in values:
                continue
            v = values[(ds, system)]
            xpos = x[j] + (i - (len(bars) - 1) / 2) * BAR_W
            ax.bar(xpos, v, width=BAR_W, color=color, label=label if j == 0 else '', alpha=1, edgecolor='black',
                   linewidth=2.5)
            per_ds[ds].append((v, xpos))
    for ds in DATASETS:  # shortest bar per dataset: red arrow and a star
        v, xpos = min(per_ds[ds])
        ax.annotate('', xy=(xpos, v), xytext=(xpos, v + 8), arrowprops=dict(arrowstyle='->', color='red', lw=4),
                    zorder=10)
        ax.plot(xpos, v + 10, zorder=11, **STAR)
    ax.set_xticks(x)
    ax.set_xticklabels(DATASETS)
    ax.set_ylim(0, 120)
    ax.set_ylabel(ylabel, fontdict={'family': 'Times New Roman'})
    star = Line2D([], [], linestyle='none', label='Shortest', **STAR)
    handles, labels = ax.get_legend_handles_labels()
    h, lab = handles + [star], labels + ['Shortest']
    order = style.row_major(list(range(len(h))), ncol)
    style.axes_legend(ax, [h[i] for i in order], [lab[i] for i in order], ncol=ncol, loc='lower center',
                      bbox_to_anchor=(0.5, 1.0), frameon=False, columnspacing=0.5, handlelength=0.6,
                      handletextpad=0.2, borderaxespad=0.2, prop={'family': 'Times New Roman'})
    ax.grid(False)
    ax.minorticks_off()
    ax.tick_params(which='minor', length=0)
    ax.tick_params(axis='y', which='both', length=5)
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    style.fit_text(plt.gcf(), 0.47 * style.TEXT_W, tight_layout=True)
    plt.savefig(out, bbox_inches='tight')
    plt.close()
    logger.info('wrote %s', out)


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig05_traversal_depth.csv')
    draw(d[d['panel'] == 'a'], BARS_A, 'Search Iterations (SI)', 5,
         os.path.join(args.out, 'fig05a_search_iterations.pdf'))
    draw(d[d['panel'] == 'b'], BARS_B, 'Common-Search Hops (CH)', 4,
         os.path.join(args.out, 'fig05b_common_search_hops.pdf'))


if __name__ == '__main__':
    main()
