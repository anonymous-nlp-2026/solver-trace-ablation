#!/usr/bin/env python3
"""Generate all 4 paper figures for solver-trace-ablation."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import os

OUT = '/root/solver-trace-ablation/paper/figures'
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.labelsize': 9,
    'xtick.labelsize': 8,
    'ytick.labelsize': 8,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
    'axes.spines.top': False,
    'lines.linewidth': 1.8,
})

CB = '#0072B2'   # blue
CR = '#D55E00'   # red
CG = '#009E73'   # green
CO = '#E69F00'   # orange
CP = '#CC79A7'   # purple
CY = '#999999'   # gray


def fig1_sta_overview():
    fig = plt.figure(figsize=(6.75, 2.4))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.3, 0.95], wspace=0.32)

    def box(ax, cx, cy, w, h, txt, fc='#F0F0F0', ec='black', fs=8, fw='normal', lw=1.0):
        p = FancyBboxPatch((cx - w/2, cy - h/2), w, h,
                           boxstyle="round,pad=0.06", fc=fc, ec=ec, lw=lw, zorder=2)
        ax.add_patch(p)
        ax.text(cx, cy, txt, ha='center', va='center', fontsize=fs,
                fontweight=fw, zorder=3, linespacing=1.3)

    def arr(ax, x1, y1, x2, y2, c='black', lw=1.2):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle='->', color=c, lw=lw), zorder=1)

    # (a) Formalization pipeline
    a = fig.add_subplot(gs[0])
    a.set_xlim(0, 1); a.set_ylim(0, 1); a.axis('off')
    a.set_title('(a) Formalization Pipeline', fontsize=9, fontweight='bold', pad=6)
    box(a, 0.5, 0.88, 0.72, 0.11, 'NL Problem', fc='#E3F2FD')
    box(a, 0.5, 0.66, 0.72, 0.11, 'LLM', fc='#FFF3E0', fw='bold')
    box(a, 0.5, 0.44, 0.72, 0.13, 'Formal Logic\n(FOL / Prolog)', fc='#E3F2FD', fs=7)
    box(a, 0.5, 0.22, 0.72, 0.11, 'Solver', fc='#F3E5F5', fw='bold')
    box(a, 0.22, 0.03, 0.33, 0.09, 'Proved', fc='#E8F5E9', fs=7, ec=CG)
    box(a, 0.78, 0.03, 0.33, 0.09, 'Failed', fc='#FFEBEE', fs=7, ec=CR)
    arr(a, 0.5, 0.825, 0.5, 0.72)
    arr(a, 0.5, 0.605, 0.5, 0.505)
    arr(a, 0.5, 0.375, 0.5, 0.28)
    arr(a, 0.34, 0.165, 0.26, 0.08)
    arr(a, 0.66, 0.165, 0.74, 0.08)

    # (b) Solver Trace Ablation
    b = fig.add_subplot(gs[1])
    b.set_xlim(0, 1); b.set_ylim(0, 1); b.axis('off')
    b.set_title('(b) Formalization\nLeave-One-Out Ablation', fontsize=8, fontweight='bold', pad=8)
    box(b, 0.5, 0.88, 0.88, 0.11,
        r'Proved: $p_1, p_2, \ldots, p_n \rightarrow c$', fc='#E8F5E9', fs=8)
    box(b, 0.5, 0.62, 0.88, 0.20, '', fc='#FAFAFA', ec=CY, lw=0.8)
    b.text(0.5, 0.69, 'Leave-One-Out Ablation', ha='center', va='center',
           fontsize=8, fontweight='bold')
    b.text(0.5, 0.58, r'For each $p_i$: remove & re-check solver', ha='center',
           va='center', fontsize=7, color=CY)
    arr(b, 0.5, 0.825, 0.5, 0.72)
    box(b, 0.5, 0.35, 0.88, 0.14, '', fc='#FFF8E1')
    b.text(0.5, 0.39, r'ISR $= n_{\rm necessary}\, /\, n_{\rm total}$',
           ha='center', va='center', fontsize=9, fontweight='bold')
    b.text(0.5, 0.31, 'Indispensable Step Ratio', ha='center', va='center',
           fontsize=7, style='italic', color=CY)
    arr(b, 0.5, 0.52, 0.5, 0.42)
    box(b, 0.25, 0.1, 0.38, 0.12, 'ISR = 1\nAll necessary', fc='#E8F5E9', fs=7, ec=CG)
    box(b, 0.75, 0.1, 0.38, 0.12, 'ISR = 0\nAll redundant', fc='#FFEBEE', fs=7, ec=CR)
    arr(b, 0.35, 0.28, 0.28, 0.16, c=CG)
    arr(b, 0.65, 0.28, 0.72, 0.16, c=CR)

    # (c) Dual role
    c = fig.add_subplot(gs[2])
    c.set_xlim(0, 1); c.set_ylim(0, 1); c.axis('off')
    c.set_title('(c) Dual Role of ISR', fontsize=9, fontweight='bold', pad=6)
    box(c, 0.5, 0.82, 0.7, 0.12, 'ISR Metric', fc='#E3F2FD', fw='bold')
    box(c, 0.5, 0.53, 0.82, 0.18, 'As Diagnostic\n(post-hoc evaluation)', fc='#E8F5E9',
        fs=7.5, ec=CG, lw=1.2)
    c.text(0.95, 0.53, '✓', ha='center', va='center', fontsize=14,
           color=CG, fontweight='bold')
    box(c, 0.5, 0.22, 0.82, 0.18, 'As RL Reward\n(training signal)', fc='#FFEBEE',
        fs=7.5, ec=CR, lw=1.2)
    c.text(0.95, 0.22, '✗', ha='center', va='center', fontsize=14,
           color=CR, fontweight='bold')
    arr(c, 0.5, 0.76, 0.5, 0.62)
    arr(c, 0.5, 0.44, 0.5, 0.31, c=CR)
    c.text(0.5, 0.05, "Goodhart's Law:\nmodel games the metric",
           ha='center', va='center', fontsize=7, color=CR, style='italic')

    fig.savefig(os.path.join(OUT, 'fig1_sta_overview.pdf'))
    plt.close(fig)
    print('Saved fig1_sta_overview.pdf')


def fig2_isr_collapse():
    steps = [100, 500, 1000]
    pr = [32, 93, 96]
    isr = [0.6423, 0.1324, 0.003]

    h7_frac = [0.281, 0.0, 0.0]
    h1_frac = [0.031, 0.871, 1.0]
    nh_frac = [0.688, 0.129, 0.0]

    fig = plt.figure(figsize=(3.31, 4.2))
    gs_main = fig.add_gridspec(2, 1, height_ratios=[1, 0.75], hspace=0.35)

    ax1 = fig.add_subplot(gs_main[0])
    ax1.set_xlabel('Training Steps')
    ax1.set_ylabel('Prove Rate (%)', color=CB)
    l1 = ax1.plot(steps, pr, 'o-', color=CB, label='Prove Rate', markersize=6, zorder=3)
    ax1.tick_params(axis='y', labelcolor=CB)
    ax1.set_ylim(0, 108)
    ax1.set_xticks(steps)
    for s, p in zip(steps, pr):
        ax1.text(s, p + 4, f'{p}%', ha='center', va='bottom', fontsize=7, color=CB)

    ax2 = ax1.twinx()
    ax2.set_ylabel('ISR (proved only)', color=CR)
    l2 = ax2.plot(steps, isr, 's--', color=CR, label='ISR', markersize=6, zorder=3)
    ax2.tick_params(axis='y', labelcolor=CR)
    ax2.set_ylim(-0.05, 0.80)
    ax2.spines['right'].set_visible(True)
    ax2.spines['right'].set_color(CR)
    ax2.spines['right'].set_linewidth(0.8)

    ax2.annotate('ISR=0.003\n94.8% at ISR=0\n18.6 comp./proof',
                 xy=(1000, 0.003), xytext=(550, 0.45),
                 fontsize=6.5, color=CR, ha='center',
                 arrowprops=dict(arrowstyle='->', color=CR, lw=0.8))

    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc='center left',
               fontsize=7, framealpha=0.9)
    ax1.grid(True, alpha=0.15, zorder=0)

    ax3 = fig.add_subplot(gs_main[1])
    x = np.arange(3)
    w = 0.55

    b_nh = ax3.bar(x, nh_frac, w, label='Non-hacking', color=CG, zorder=2)
    b_h1 = ax3.bar(x, h1_frac, w, bottom=nh_frac, label='H1 (conclusion embed.)', color=CO, zorder=2)
    bot = [a + b for a, b in zip(nh_frac, h1_frac)]
    b_h7 = ax3.bar(x, h7_frac, w, bottom=bot, label='H7 (premise=conclusion)', color=CR, zorder=2)

    for i in range(3):
        if nh_frac[i] >= 0.07:
            ax3.text(i, nh_frac[i] / 2, f'{nh_frac[i]:.0%}',
                     ha='center', va='center', fontsize=6.5,
                     color='white' if nh_frac[i] > 0.15 else 'black', fontweight='bold')
        if h1_frac[i] >= 0.07:
            ax3.text(i, nh_frac[i] + h1_frac[i] / 2, f'{h1_frac[i]:.0%}',
                     ha='center', va='center', fontsize=6.5,
                     color='white' if h1_frac[i] > 0.15 else 'black', fontweight='bold')
        if h7_frac[i] >= 0.07:
            ax3.text(i, bot[i] + h7_frac[i] / 2, f'{h7_frac[i]:.0%}',
                     ha='center', va='center', fontsize=6.5,
                     color='white' if h7_frac[i] > 0.15 else 'black', fontweight='bold')

    ax3.set_ylabel('Fraction of Proved')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'Step {s}\n(n={n})' for s, n in zip(steps, pr)], fontsize=7)
    ax3.set_ylim(0, 1.08)
    ax3.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax3.legend(fontsize=6, loc='upper right', framealpha=0.9)
    ax3.grid(True, alpha=0.15, axis='y', zorder=0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig2_isr_collapse.pdf'))
    plt.close(fig)
    print('Saved fig2_isr_collapse.pdf')


def fig3_hacking_composition():
    conditions = ['Exec-only', 'ISR\n(β=0.1)', 'Random\n(β=0.1)', 'SFT']
    # Weighted 3-seed means from registry
    h7      = [0,     0.645, 0,     0]
    h1      = [0.871, 0.182, 0.715, 0.071]
    genuine = [0.129, 0.173, 0.285, 0.929]

    x = np.arange(len(conditions))
    w = 0.55
    fig, ax = plt.subplots(figsize=(3.31, 2.8))

    b_h7 = ax.bar(x, h7, w, label='H7 (premise=conclusion)', color=CR, zorder=2)
    b_h1 = ax.bar(x, h1, w, bottom=h7, label='H1 (conclusion embed.)', color=CO, zorder=2)
    bot = [a + b for a, b in zip(h7, h1)]
    b_gn = ax.bar(x, genuine, w, bottom=bot, label='Non-hacking', color=CG, zorder=2)

    def lbl(bars, vals, bots):
        for r, v, bt in zip(bars, vals, bots):
            if v >= 0.07:
                ax.text(r.get_x() + r.get_width()/2, bt + v/2,
                        f'{v:.0%}', ha='center', va='center', fontsize=6.5,
                        color='white' if v > 0.15 else 'black', fontweight='bold')

    lbl(b_h7, h7, [0]*4)
    lbl(b_h1, h1, h7)
    lbl(b_gn, genuine, bot)

    ax.set_ylabel('Fraction of Proved Samples')
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, fontsize=7)
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax.legend(fontsize=6.5, loc='upper right', framealpha=0.9)
    ax.grid(True, alpha=0.15, axis='y', zorder=0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig3_hacking_composition.pdf'))
    plt.close(fig)
    print('Saved fig3_hacking_composition.pdf')


def fig4_prolog_comparison():
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(6.75, 2.8),
                                      gridspec_kw={'width_ratios': [1, 1], 'wspace': 0.35})
    seeds = ['s42', 's43', 's44']

    # (a) FOL vs Prolog exec-only ISR (proved_only_isr)
    fol_isr    = [0.1324, 0,     0.5816]
    prolog_isr = [0.037,  0.015, 0.3866]
    x = np.arange(3)
    w = 0.3
    ax_a.bar(x - w/2, fol_isr, w, label='FOL (Prover9)', color=CB, zorder=2)
    ax_a.bar(x + w/2, prolog_isr, w, label='Prolog (SWI)', color=CO, zorder=2)
    for i, (f, p) in enumerate(zip(fol_isr, prolog_isr)):
        ax_a.text(i - w/2, f + 0.015, f'{f:.3f}', ha='center', va='bottom',
                  fontsize=6, color=CB)
        ax_a.text(i + w/2, p + 0.015, f'{p:.3f}', ha='center', va='bottom',
                  fontsize=6, color=CO)
    ax_a.annotate('H7 collapse\n(tautological)', xy=(2 - w/2, 0.58),
                  xytext=(1.3, 0.52), fontsize=6, color=CY, ha='center',
                  arrowprops=dict(arrowstyle='->', color=CY, lw=0.6))
    ax_a.set_ylabel('ISR (proved only)')
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(seeds)
    ax_a.set_ylim(0, 0.72)
    ax_a.legend(fontsize=7, loc='upper left')
    ax_a.set_title('(a) Exec-only: FOL vs Prolog ISR', fontsize=9, fontweight='bold')
    ax_a.grid(True, alpha=0.15, axis='y', zorder=0)

    # (b) Prolog STA gaming modes
    h7_b   = [1.0,   0,     0]
    h1_b   = [0,     0.974, 0]
    qaf_b  = [0,     0,     0.979]
    gen_b  = [0,     0.026, 0.021]
    x_b = np.arange(3)
    wb = 0.5
    ax_b.bar(x_b, h7_b, wb, label='H7 (single premise)', color=CR, zorder=2)
    b2 = [a + b for a, b in zip(h7_b, h1_b)]
    ax_b.bar(x_b, h1_b, wb, bottom=h7_b, label='H1 (degen. syntax)', color=CO, zorder=2)
    b3 = [a + b for a, b in zip(b2, qaf_b)]
    ax_b.bar(x_b, qaf_b, wb, bottom=b2, label='Query-as-fact', color=CP, zorder=2)
    ax_b.bar(x_b, gen_b, wb, bottom=b3, label='Non-hacking', color=CG, zorder=2)

    mode_lbl = ['H7', 'H1 / degen.', 'Query-as-fact']
    pr_lbl = ['PR=50%', 'PR=38%', 'PR=97%']
    for i, (ml, pl) in enumerate(zip(mode_lbl, pr_lbl)):
        ax_b.text(i, 1.07, ml, ha='center', va='bottom', fontsize=6.5,
                  fontweight='bold', color=CY)
        ax_b.text(i, -0.07, pl, ha='center', va='top', fontsize=6, color=CY)

    ax_b.set_ylabel('Fraction of Proved Samples')
    ax_b.set_xticks(x_b)
    ax_b.set_xticklabels(seeds)
    ax_b.set_ylim(-0.02, 1.18)
    ax_b.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
    ax_b.legend(fontsize=6, loc='center right', framealpha=0.9)
    ax_b.set_title('(b) Prolog + STA: Gaming Modes', fontsize=9, fontweight='bold')
    ax_b.grid(True, alpha=0.15, axis='y', zorder=0)

    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'fig4_prolog_comparison.pdf'))
    plt.close(fig)
    print('Saved fig4_prolog_comparison.pdf')


if __name__ == '__main__':
    fig1_sta_overview()
    fig2_isr_collapse()
    fig3_hacking_composition()
    fig4_prolog_comparison()
    print(f'\nAll figures saved to {OUT}')
