"""Figure 13: Recall@10 and QPS of CAGRA on SIFT1M when one parameter varies.

Input: results/fig13_cagra_parameters.csv (time_us: search time of 10,000 queries).
Output: fig13_<parameter>.pdf for search_width, internal_topk, degree and team_size.
"""
import logging
import os

import matplotlib.pyplot as plt

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

XLABEL = {'search_width': 'Search Width', 'internal_topk': 'Internal TopK', 'degree': 'Degree',
          'team_size': 'Team Size'}


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig13_cagra_parameters.csv')
    for param, g in d.groupby('parameter'):
        g = g.sort_values('value')
        x = range(len(g))
        fig, ax1 = plt.subplots(figsize=(7, 6))
        line1, = ax1.plot(x, g['recall_at_10'], marker='o', markersize=14, markerfacecolor='none', linewidth=3,
                          color='#000000', label='Recall')
        ax1.set_xlabel(XLABEL[param], fontdict={'family': 'Times New Roman'})
        ax1.set_ylabel('Recall', color='#000000', fontdict={'family': 'Times New Roman'})
        ax1.tick_params(axis='y', labelcolor='#000000')
        ax1.set_xticks(x)
        ax1.set_xticklabels(g['value'].astype(str), rotation=45, ha='right', rotation_mode='anchor')
        ax2 = ax1.twinx()
        qps_millions = 1e4 / g['time_us']  # 10,000 queries in time_us microseconds, in millions of queries/s
        line2, = ax2.plot(x, qps_millions, marker='^', markersize=14, markerfacecolor='none', linewidth=3,
                          color='tab:blue', label='QPS')
        ax2.set_ylabel('QPS (1e6)', color='tab:blue', fontdict={'family': 'Times New Roman'})
        ax2.tick_params(axis='y', labelcolor='tab:blue')
        style.axes_legend(ax1, [line1, line2], ['Recall', 'QPS'], loc='lower center', bbox_to_anchor=(0.5, 1.0),
                          ncol=2, frameon=False, handlelength=1.5, handletextpad=0.4, columnspacing=1.0,
                          borderaxespad=0.2, prop={'family': 'Times New Roman'})
        plt.grid(True, linestyle='--', alpha=0.6)
        style.fit_text(fig, 0.47 * style.COL_W, tight_layout=True)
        out = os.path.join(args.out, f'fig13_{param}.pdf')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
