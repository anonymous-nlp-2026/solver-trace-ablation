import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.8,
})

COLORS = {
    'exec_only': '#0072B2',
    'isr_beta01': '#D55E00',
    'random_beta01': '#009E73',
}

exec_steps = [100, 500, 1000]
exec_isr = [0.6423, 0.1324, 0.0027]

isr_500_seeds = [0.4154, 0.8301, 0.9022]
isr_500_mean = np.mean(isr_500_seeds)
isr_500_std = np.std(isr_500_seeds, ddof=1)

random_500_seeds = [0.1314, 0.1495, 0.0695]
random_500_mean = np.mean(random_500_seeds)
random_500_std = np.std(random_500_seeds, ddof=1)

fig, ax = plt.subplots(figsize=(5, 3.5))

ax.plot(exec_steps, exec_isr, '-o', color=COLORS['exec_only'],
        markersize=7, zorder=5, label='Exec-only GRPO')

for i, (x, y) in enumerate(zip(exec_steps, exec_isr)):
    offset_y = 0.04
    if i == 2:
        offset_y = 0.035
    ax.annotate(f'{y:.3f}', (x, y), textcoords='offset points',
                xytext=(0, 10), ha='center', fontsize=9,
                color=COLORS['exec_only'], fontweight='bold')

ax.errorbar(500, isr_500_mean, yerr=isr_500_std, fmt='s',
            color=COLORS['isr_beta01'], markersize=8, capsize=4,
            capthick=1.5, zorder=5, label=r'ISR $\beta$=0.1')
ax.annotate(f'{isr_500_mean:.3f}', (500, isr_500_mean), textcoords='offset points',
            xytext=(45, -5), ha='center', fontsize=9,
            color=COLORS['isr_beta01'], fontweight='bold')

ax.errorbar(500, random_500_mean, yerr=random_500_std, fmt='^',
            color=COLORS['random_beta01'], markersize=8, capsize=4,
            capthick=1.5, zorder=5, label=r'Random $\beta$=0.1')
ax.annotate(f'{random_500_mean:.3f}', (500, random_500_mean), textcoords='offset points',
            xytext=(45, -5), ha='center', fontsize=9,
            color=COLORS['random_beta01'], fontweight='bold')

ax.set_xlabel('GRPO Training Steps')
ax.set_ylabel('ISR (proved-only)')
ax.set_xticks(exec_steps)
ax.set_xlim(0, 1100)
ax.set_ylim(-0.05, 0.85)

ax.legend(loc='upper right', framealpha=0.9, edgecolor='none')

outdir = '/root/solver-trace-ablation/paper/figures'
os.makedirs(outdir, exist_ok=True)
fig.savefig(f'{outdir}/isr_collapse.pdf')
fig.savefig(f'{outdir}/isr_collapse.png')
print(f'Saved to {outdir}/isr_collapse.pdf and .png')
plt.close()
