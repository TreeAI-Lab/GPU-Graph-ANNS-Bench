"""Figure 9: share of search time in candidate locating (CL), distance calculation (DC) and data maintenance (DM),
on the GPU and on the CPU, and the total time per 10,000 queries.

Left: stacked shares. Right: bar length is log10 of the total time in microseconds, split by the same shares.
Input: results/fig09_search_components.csv. Output: fig09_<dataset>.pdf and fig09_legend.pdf.
"""
import logging
import os

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

import style

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

SYSTEMS = ['SONG', 'GGNN', 'GANNS', 'CAGRA', 'BANG', 'PilotANN']
STAGES = ['gpu_cl', 'cpu_cl', 'gpu_dc', 'cpu_dc', 'gpu_dm', 'cpu_dm']
LABEL = {'gpu_cl': 'GPU CL', 'cpu_cl': 'CPU CL', 'gpu_dc': 'GPU DC', 'cpu_dc': 'CPU DC', 'gpu_dm': 'GPU DM',
         'cpu_dm': 'CPU DM'}
COLOR = {'gpu_cl': (120, 150, 180), 'cpu_cl': (200, 215, 225), 'gpu_dc': (100, 180, 220),
         'cpu_dc': (210, 235, 245), 'gpu_dm': (240, 140, 90), 'cpu_dm': (255, 200, 160)}
COLOR = {k: tuple(c / 255.0 for c in v) for k, v in COLOR.items()}


def main() -> None:
    args = style.parse_args(__doc__.splitlines()[0])
    d = style.read(args, 'fig09_search_components.csv')
    for ds in d['dataset'].unique():
        sub = d[d['dataset'] == ds]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), width_ratios=[1.5, 1.8])
        share = np.zeros((len(SYSTEMS), len(STAGES)))
        log_total = np.zeros(len(SYSTEMS))
        for i, system in enumerate(SYSTEMS):
            row = sub[sub['system'] == system]
            if row.empty:
                continue
            v = row[STAGES].astype(float).mean(axis=0).values
            share[i] = v / v.sum()
            log_total[i] = np.log10(float(row['total_time_us'].mean()))
        x = np.arange(len(SYSTEMS))
        bottom = np.zeros(len(SYSTEMS))
        for j, stage in enumerate(STAGES):
            ax1.bar(x, share[:, j], bottom=bottom, color=COLOR[stage], label=stage, width=0.8)
            bottom += share[:, j]
        ax1.set_xticks(x)
        ax1.set_xticklabels(SYSTEMS, rotation=60, ha='right', rotation_mode='anchor')
        present = set(sub['system'])
        for i, system in enumerate(SYSTEMS):
            if system not in present:
                ax1.get_xticklabels()[i].set_color('lightgray')
        ax1.set_ylim(0, 1)
        ax1.set_ylabel('Time Ratio', fontdict={'family': 'Times New Roman'})
        widths = (share.T * log_total).T
        left = np.zeros(len(SYSTEMS))
        y = np.arange(len(SYSTEMS))
        for j, stage in enumerate(STAGES):
            ax2.barh(y, widths[:, j], left=left, color=COLOR[stage], edgecolor='k', height=0.8, label=stage)
            left += widths[:, j]
        ax2.set_yticks(y)
        ax2.set_yticklabels(SYSTEMS)
        ax2.invert_yaxis()
        ax2.set_xlabel('Log-scaled Total\nTime (us)', fontdict={'family': 'Times New Roman'})
        style.fit_text(fig, 0.48 * style.TEXT_W, tight_layout=True)
        out = os.path.join(args.out, f'fig09_{ds}.pdf')
        plt.savefig(out, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info('wrote %s', out)
    handles = [mpatches.Patch(color=COLOR[s], label=LABEL[s]) for s in STAGES]
    out = os.path.join(args.out, 'fig09_legend.pdf')
    style.save_legend(handles, out, ncol=6, prop={'family': 'Times New Roman', 'size': 9}, columnspacing=1.5)
    logger.info('wrote %s', out)


if __name__ == '__main__':
    main()
