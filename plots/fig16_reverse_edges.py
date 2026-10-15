"""Figure 16: CAGRA on SIFT1M built with (R) and without (NR) reverse edges.

(a) Recall@10 against the number of search iterations; (b) share of short (SE) and long (LE) edges.
Input: results/fig16_reverse_edges.csv. Output: fig16a_iter_recall.pdf, fig16b_edge_share.pdf.
"""
import logging
import os

import matplotlib.pyplot as plt

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

BUILDS = {'R': dict(color='#E63946', marker='o'), 'NR': dict(color='#457B9D', marker='s')}
ITERATIONS = [15, 20, 32, 48, 64]  # iteration budgets shown in panel (a)
EDGE_COLORS = {'SE': '#D6EEF6', 'LE': '#B5DCE6'}
WIDTH = 0.48 * style.COL_W  # two panels side by side in one column


def times(ax) -> None:
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_fontfamily('Times New Roman')
    ax.spines[['top', 'right']].set_visible(False)


def iter_recall(d, out: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    for build, s in BUILDS.items():
        sub = d[(d['build'] == build) & d['it'].isin(ITERATIONS)].sort_values('it')
        ax.plot(sub['it'], sub['recall_at_10'], color=s['color'], marker=s['marker'], markersize=10,
                markerfacecolor='white', markeredgewidth=2, linewidth=3, label=build)
    ax.set_xticks(ITERATIONS[1:])
    ax.set_ylim(0.6, 1.0)
    ax.set_yticks([0.6, 0.8, 1.0])
    ax.set_xlabel('Iterations', fontfamily='Times New Roman')
    ax.set_ylabel('Recall@10', fontfamily='Times New Roman')
    ax.grid(True, linestyle='--', alpha=0.4)
    times(ax)
    style.axes_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                      prop={'family': 'Times New Roman'}, handlelength=1.8, columnspacing=1.5)
    style.fit_text(fig, WIDTH, tight_layout=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out)


def edge_share(d, out: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    builds = list(BUILDS)
    short = [float(d[d['build'] == b]['short_edge_share'].iloc[0]) * 100 for b in builds]
    long_ = [float(d[d['build'] == b]['long_edge_share'].iloc[0]) * 100 for b in builds]
    x = range(len(builds))
    ax.bar(x, short, width=0.5, color=EDGE_COLORS['SE'], edgecolor='#7FB3C8', label='SE')
    ax.bar(x, long_, width=0.5, bottom=short, color=EDGE_COLORS['LE'], edgecolor='#6A9FB5', label='LE')
    label = dict(ha='center', va='center', color='#2C3E50', fontweight='bold', fontfamily='Times New Roman')
    for i in x:
        ax.text(i, short[i] / 2, f'{short[i]:.1f}%', **label)
        ax.text(i, short[i] + long_[i] / 2, f'{long_[i]:.1f}%', **label)
    ax.set_xticks(list(x))
    ax.set_xticklabels(builds)
    ax.set_xlim(-0.6, len(builds) - 0.4)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels([f'{v}%' for v in (0, 25, 50, 75, 100)])
    ax.axhline(50, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.set_xlabel('Build Type', fontfamily='Times New Roman')
    ax.set_ylabel('Edge Ratio', fontfamily='Times New Roman')
    ax.grid(True, axis='y', linestyle='--', alpha=0.4)
    times(ax)
    style.axes_legend(ax, loc='lower center', bbox_to_anchor=(0.5, 1.0), ncol=2, frameon=False,
                      prop={'family': 'Times New Roman'}, handlelength=1.8, columnspacing=1.5)
    style.fit_text(fig, WIDTH, tight_layout=True)
    fig.savefig(out, bbox_inches='tight')
    plt.close(fig)
    logger.info('wrote %s', out)


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig16_reverse_edges.csv')
    iter_recall(d[d['panel'] == 'a'], os.path.join(args.out, 'fig16a_iter_recall.pdf'))
    edge_share(d[d['panel'] == 'b'], os.path.join(args.out, 'fig16b_edge_share.pdf'))


if __name__ == '__main__':
    main()
