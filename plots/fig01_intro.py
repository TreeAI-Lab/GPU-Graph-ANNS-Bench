"""Figure 1: QPS versus Recall@10 on SIFT1M, CAGRA against the non-graph GPU indexes.

Input: results/fig01_intro_sift1m.csv. Output: fig01_intro.pdf.
"""
import logging
import os

import matplotlib.pyplot as plt

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

MIN_RECALL = 0.6


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig01_intro_sift1m.csv')
    d = d[d['recall_at_10'] >= MIN_RECALL]
    cagra = d[d['system'] == 'CAGRA'].sort_values(by='recall_at_10')
    plt.figure(figsize=(8, 5))
    plt.plot(cagra['recall_at_10'], cagra['qps'], marker=style.MARKER['CAGRA'], markersize=9, markerfacecolor='none',
             markeredgecolor=style.COLOR['CAGRA'], markeredgewidth=1.5, color=style.COLOR['CAGRA'], linewidth=2,
             label='CAGRA (graph)')
    for system, s in style.BASELINES.items():
        sub = d[d['system'] == system]
        xs, ys = zip(*sorted(zip(sub['recall_at_10'], sub['qps'])))
        plt.plot(xs, ys, linestyle='--', linewidth=2, color=s['color'], marker=s['marker'], markersize=10,
                 markeredgewidth=2, markevery=max(1, len(xs) // 8), label=s['label'])
    bf = d[d['system'] == 'cuvs_bf'].iloc[0]
    plt.plot([bf['recall_at_10']], [bf['qps']], linestyle='none', marker=style.BRUTE_FORCE['marker'], markersize=18,
             color=style.BRUTE_FORCE['color'], clip_on=False, zorder=5, label=style.BRUTE_FORCE['label'])
    plt.yscale('log')
    plt.xlabel('Recall@10')
    plt.ylabel('QPS')
    plt.xticks(fontproperties='Times New Roman')
    plt.yticks(fontproperties='Times New Roman')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.xlim(MIN_RECALL, 1.005)
    h, labels = plt.gca().get_legend_handles_labels()  # two rows above the axes
    order = style.row_major(list(range(len(h))), 2)
    style.axes_legend(plt.gca(), [h[i] for i in order], [labels[i] for i in order], loc='lower center',
                      bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False, prop={'family': 'Times New Roman'},
                      columnspacing=1.0, handletextpad=0.4, handlelength=1.8)
    style.fit_text(plt.gcf(), 0.9 * style.COL_W)  # 0.9 columnwidth in the paper
    out = os.path.join(args.out, 'fig01_intro.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
