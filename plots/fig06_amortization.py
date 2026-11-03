"""Figure 6: time per query when the index is loaded once and then serves batches of 10,000 queries.

Solid: (one-time load + all batches so far) / queries served. Dashed: per-query time with a resident index.
Input: results/fig06_amortization.csv. Output: fig06_<dataset>.pdf and fig06_legend.pdf.
"""
import logging
import os

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SYSTEMS = ['CAGRA', 'GGNN', 'GANNS', 'SONG']


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig06_amortization.csv')
    for ds in ['deep1M', 'deep10M', 'deep100M']:
        plt.figure(figsize=(8, 6))
        for system in SYSTEMS:
            sub = d[(d['dataset'] == ds) & (d['system'] == system)]
            if sub.empty:
                continue  # DEEP100M has only CAGRA
            pts = sorted(zip(sub[sub['kind'] == 'amortized']['queries_served'].astype(int),
                             sub[sub['kind'] == 'amortized']['ms_per_query']))
            c = style.COLOR[system]
            plt.plot([p[0] for p in pts], [p[1] for p in pts], marker=style.MARKER[system], markersize=11,
                     markerfacecolor='none', markeredgecolor=c, markeredgewidth=1.5, color=c, linewidth=2)
            plt.axhline(sub[sub['kind'] == 'resident']['ms_per_query'].iloc[0], color=c, linestyle='--', linewidth=1.5)
        plt.xscale('log')
        plt.yscale('log')
        plt.xlabel('Queries served', fontfamily='Times New Roman')
        plt.ylabel('Time per query (ms)', fontfamily='Times New Roman')
        plt.xticks(fontproperties='Times New Roman')
        plt.yticks(fontproperties='Times New Roman')
        plt.grid(True, which='major', alpha=0.3)
        style.fit_text(plt.gcf(), 0.3 * style.TEXT_W, tight_layout=True)
        out = os.path.join(args.out, f'fig06_{ds}.pdf')
        plt.savefig(out, bbox_inches='tight')
        plt.close()
        logger.info('wrote %s', out)
    handles = [Line2D([], [], marker=style.MARKER[s], markersize=6, markerfacecolor='none',
                      markeredgecolor=style.COLOR[s], markeredgewidth=1, color=style.COLOR[s], linewidth=1.2, label=s)
               for s in SYSTEMS]
    handles.append(Line2D([], [], color='gray', linestyle='--', linewidth=1, label='Resident index'))
    out = os.path.join(args.out, 'fig06_legend.pdf')
    style.save_legend(handles, out, ncol=len(handles), prop={'family': 'Times New Roman', 'size': 9})
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
