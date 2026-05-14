"""
Summary results table figure for presentation slides.
Numbers from eval_full_comparison run (2026-05-09).
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

out_dir = os.path.join(os.path.dirname(__file__), "output", "slides")
os.makedirs(out_dir, exist_ok=True)

# ── Data ─────────────────────────────────────────────────────────────────────
methods = [
    "Log-Gabor",
    "Steerable",
    "Adaptive IIR",
    "Adap. IIR BD",
    "DT-CWT",
]

# Real data
real_f1     = [1.000, 0.991, 0.965, 0.925, 0.983]

# Sim data (avg F across all PPP)
sim_f1      = [0.836, 0.828, 0.817, 0.753, 0.836]

# Memory (edge phase, GB)
edge_mem_gb = [12.0,  12.0,  4.2,   4.2,   20.3]
sim_mem_mb  = [488.6, 488.6, 171.4, 171.4, 829.6]

# Runtime ratio vs Log-Gabor (sim)
rt_ratio    = [1.00, 0.34, 4.09, 1.48, 2.38]

# ── Figure ────────────────────────────────────────────────────────────────────
BG   = '#0d0d0d'
CELL = '#181818'
HEAD = '#1e2a3a'

ACCENT  = '#4e9af1'   # blue — reference
GREEN   = '#4ef18a'
ORANGE  = '#f1834e'
YELLOW  = '#f1d44e'
PURPLE  = '#b04ef1'

method_colors = [ACCENT, GREEN, ORANGE, ORANGE, PURPLE]

fig, axes = plt.subplots(1, 4, figsize=(15, 4.0), facecolor=BG)
fig.subplots_adjust(left=0.04, right=0.99, top=0.82, bottom=0.05, wspace=0.35)

n = len(methods)
x = np.arange(n)
bar_kw = dict(width=0.6, zorder=3)

def style_ax(ax, title, ylabel, ymax, ref_line=None):
    ax.set_facecolor(CELL)
    ax.set_title(title, color='#cccccc', fontsize=11, fontweight='bold', pad=6)
    ax.set_ylabel(ylabel, color='#999999', fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha='right', fontsize=9, color='white')
    ax.tick_params(axis='y', colors='#888888', labelsize=8)
    ax.set_ylim(0, ymax)
    ax.set_xlim(-0.5, n - 0.5)
    for spine in ax.spines.values():
        spine.set_color('#333333')
    ax.yaxis.grid(True, color='#2a2a2a', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if ref_line is not None:
        ax.axhline(ref_line, color='#555555', lw=1, ls='--', zorder=2)

# ── Panel 1: Real F1 score ─────────────────────────────────────────────────
ax = axes[0]
bars = ax.bar(x, real_f1, color=method_colors, **bar_kw)
style_ax(ax, "F-Score  (Real data)", "Relative F-score vs Log-Gabor", 1.12)
ax.axhline(1.0, color='#555555', lw=1, ls='--', zorder=2)
for i, (b, v) in enumerate(zip(bars, real_f1)):
    ax.text(b.get_x() + b.get_width()/2, v + 0.008, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')

# ── Panel 2: Sim F1 score ─────────────────────────────────────────────────
ax = axes[1]
bars = ax.bar(x, sim_f1, color=method_colors, **bar_kw)
style_ax(ax, "F-Score  (Simulated data)", "Absolute F-score vs GT", 0.92)
for i, (b, v) in enumerate(zip(bars, sim_f1)):
    ax.text(b.get_x() + b.get_width()/2, v + 0.003, f'{v:.3f}',
            ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')

# ── Panel 3: Edge memory (GB) ──────────────────────────────────────────────
ax = axes[2]
bars = ax.bar(x, edge_mem_gb, color=method_colors, **bar_kw)
style_ax(ax, "Edge-Phase Memory (Real)", "Peak memory  [GB]", 24)
for i, (b, v) in enumerate(zip(bars, edge_mem_gb)):
    ax.text(b.get_x() + b.get_width()/2, v + 0.3, f'{v:.1f}',
            ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
# IIR saving annotation
ax.annotate('−65%\nvs LG', xy=(2, 4.2), xytext=(2, 10),
            arrowprops=dict(arrowstyle='->', color=GREEN, lw=1.4),
            ha='center', color=GREEN, fontsize=9, fontweight='bold')

# ── Panel 4: Runtime ratio ─────────────────────────────────────────────────
ax = axes[3]
bars = ax.bar(x, rt_ratio, color=method_colors, **bar_kw)
style_ax(ax, "Runtime Ratio  (Sim, vs Log-Gabor)", "×  Log-Gabor runtime", 5.5, ref_line=1.0)
for i, (b, v) in enumerate(zip(bars, rt_ratio)):
    label = f'{v:.2f}×'
    ax.text(b.get_x() + b.get_width()/2, v + 0.07, label,
            ha='center', va='bottom', fontsize=8, color='white', fontweight='bold')
ax.text(0, 1.08, 'baseline', ha='center', color='#555555', fontsize=7)

# ── Legend ────────────────────────────────────────────────────────────────────
patches = [mpatches.Patch(color=c, label=m)
           for c, m in zip(method_colors, methods)]
fig.legend(handles=patches, loc='upper center', ncol=5,
           bbox_to_anchor=(0.5, 1.01),
           framealpha=0.15, labelcolor='white', fontsize=9,
           facecolor='#1a1a1a', edgecolor='#444444')

fig.suptitle(
    "Single-Photon Filter Bank Comparison  —  Summary Results",
    color='white', fontsize=13, fontweight='bold', y=1.09
)

out_path = os.path.join(out_dir, "results_table.png")
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.close(fig)
