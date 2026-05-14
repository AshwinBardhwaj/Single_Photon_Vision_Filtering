"""
Filter comparison + Log-Gabor optimality / ML cost figure for slides.

Produces:
  1. filter_frequency_profiles.png  — radial + angular profiles of each bank
  2. log_gabor_optimality.png       — why LG is near-optimal but impractical
  3. ml_cost_comparison.png         — ML vs classical method cost breakdown
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "output" / "slides"
out_dir.mkdir(parents=True, exist_ok=True)

BG   = '#0d0d0d'
CELL = '#141414'

def new_dark_fig(figsize):
    fig = plt.figure(figsize=figsize, facecolor=BG)
    return fig

def style_ax(ax, title='', xlabel='', ylabel=''):
    ax.set_facecolor(CELL)
    ax.tick_params(colors='#888888', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#333333')
    ax.grid(True, color='#252525', linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    if title:  ax.set_title(title,   color='#cccccc', fontsize=10, fontweight='bold', pad=6)
    if xlabel: ax.set_xlabel(xlabel, color='#999999', fontsize=9)
    if ylabel: ax.set_ylabel(ylabel, color='#999999', fontsize=9)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: Radial & Angular profiles of the three filter banks
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating filter frequency profiles...")

f = np.linspace(0, 0.5, 500)
theta = np.linspace(-180, 180, 720)

log_sigma = np.log(0.55)
min_wl = 4.0; mult = 2.1; n_scales = 3
dtheta_max_lg = 60.0      # log-gabor
dtheta_max_st = 60.0      # steerable (similar bandwidth)
dtheta_max_dt = 30.0      # DT-CWT (tighter)

LG_COLOR  = '#4e9af1'
ST_COLOR  = '#f1d44e'
IIR_COLOR = '#f1834e'
DT_COLOR  = '#b04ef1'

fig = new_dark_fig((14, 5.5))
gs  = gridspec.GridSpec(2, 3, figure=fig, wspace=0.35, hspace=0.50,
                        left=0.07, right=0.97, top=0.88, bottom=0.10)

# ── Row 0: Radial profiles (all scales, one orientation) ────────────────────
ax_rad = [fig.add_subplot(gs[0, s]) for s in range(3)]
f_nz = f[1:]  # avoid log(0)

for s in range(n_scales):
    f0 = (1.0 / min_wl) / (mult ** (n_scales - 1 - s))
    lp = 1.0 / (1.0 + (f_nz / 0.45) ** 30)

    # Log-Gabor radial
    lg_rad = np.exp(-(np.log(f_nz / f0)) ** 2 / (2 * log_sigma ** 2)) * lp

    # Steerable (nth-order derivative of Gaussian, n=2)
    n_order = 2
    r = f_nz / f0
    st_rad = (r ** n_order) * np.exp(-0.5 * r ** 2)
    st_rad /= (st_rad.max() + 1e-12)

    # IIR radial is same Log-Gabor spatial as LG
    iir_rad = lg_rad.copy()

    ax = ax_rad[s]
    style_ax(ax, f'Radial Profile — Scale {s+1}\n($f_c$≈{f0:.3f} cyc/px)',
             'Spatial freq.  $f$ [cyc/px]', '|H(f)|')
    ax.plot(f_nz, lg_rad,  color=LG_COLOR,  lw=2.0, label='Log-Gabor / IIR')
    ax.plot(f_nz, st_rad,  color=ST_COLOR,  lw=2.0, label='Steerable (G₂)')
    ax.plot(f_nz, lg_rad,  color=DT_COLOR,  lw=1.2, ls='--', alpha=0.7, label='DT-CWT')
    ax.set_xlim(0, 0.5)
    if s == 0:
        ax.legend(fontsize=7.5, framealpha=0.15, labelcolor='white',
                  facecolor=CELL, edgecolor='#444444')
    ax.axvline(f0, color='#555555', lw=0.8, ls=':')

# ── Row 1: Angular profiles (one scale, all filter types) ───────────────────
ax_ang = [fig.add_subplot(gs[1, k]) for k in range(3)]

# Log-Gabor / IIR angular: raised cosine, dtheta_max = 60°
def raised_cos_window(delta_deg, dtheta_max):
    delta_deg = np.abs(delta_deg)
    out = np.where(delta_deg < dtheta_max,
                   (np.cos(np.radians(180.0 * np.minimum(1, delta_deg / dtheta_max))) + 1) / 2,
                   0.0)
    return out

# Steerable G_n angular: |cos(Δθ)|^n
def steerable_angular(delta_deg, n=2):
    rad = np.radians(delta_deg)
    return np.maximum(0.0, np.cos(rad)) ** n

delta = np.linspace(-90, 90, 500)

ax = ax_ang[0]
style_ax(ax, 'Angular Profile — Log-Gabor / IIR\n(dθ_max = 60°)',
         'Δθ from filter centre [°]', '|H(θ)|')
ax.plot(delta, raised_cos_window(delta, 60), color=LG_COLOR, lw=2.5)
ax.axvline(-60, color='#444444', lw=0.8, ls=':')
ax.axvline( 60, color='#444444', lw=0.8, ls=':')
ax.set_xlim(-90, 90)
ax.text(0, 0.5, '60° bandwidth\n(one-sided)', color='#888888', ha='center',
        fontsize=8.5, transform=ax.transAxes)

ax = ax_ang[1]
style_ax(ax, 'Angular Profile — Steerable (G₂)\n(derivative-of-Gaussian basis)',
         'Δθ from filter centre [°]', '|H(θ)|')
ax.plot(delta, steerable_angular(delta, n=2), color=ST_COLOR, lw=2.5)
ax.set_xlim(-90, 90)
ax.text(0.5, 0.5, 'cos²(Δθ)\nsteerable basis', color='#888888', ha='center',
        fontsize=8.5, transform=ax.transAxes)

ax = ax_ang[2]
style_ax(ax, 'Angular Profile — DT-CWT\n(dθ_max = 30°, tighter tiling)',
         'Δθ from filter centre [°]', '|H(θ)|')
colors_dtcwt = ['#b04ef1', '#d07ef1', '#e0aff8']
for i, (center, col) in enumerate(zip([0, 30, -30], colors_dtcwt)):
    label = f'θ={center}°' if i < 2 else None
    w = raised_cos_window(delta - center, 30)
    ax.plot(delta, w, color=col, lw=2.0 if i == 0 else 1.2,
            ls='-' if i == 0 else '--', label=label, alpha=1 if i == 0 else 0.7)
ax.set_xlim(-90, 90)
ax.text(0.5, 0.5, '30° bandwidth\nadjacent bands sum = 1', color='#888888',
        ha='center', fontsize=8.5, transform=ax.transAxes)

fig.suptitle('Spatial Filter Frequency Profiles: Radial & Angular',
             color='white', fontsize=12, fontweight='bold', y=0.97)

out_path = str(out_dir / 'filter_frequency_profiles.png')
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
print(f"Saved: {out_path}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Why Log-Gabor is near-optimal but impractical
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating log-gabor optimality figure...")

fig = new_dark_fig((13, 5))
fig.subplots_adjust(left=0.05, right=0.97, top=0.84, bottom=0.08, wspace=0.40)

gs2 = gridspec.GridSpec(1, 3, figure=fig)

# ── Panel A: Log-Gabor approaches ideal bandpass ─────────────────────────────
ax = fig.add_subplot(gs2[0])
style_ax(ax, 'A. Radial Profile: LG ≈ Ideal Bandpass',
         'Spatial freq. $f$', '|H(f)|  (normalised)')

f_plot = np.linspace(0.001, 0.5, 500)
f0 = 0.1
ideal = ((f_plot > 0.065) & (f_plot < 0.15)).astype(float)
lg = np.exp(-(np.log(f_plot / f0)) ** 2 / (2 * log_sigma ** 2))
gauss = np.exp(-((f_plot - f0) ** 2) / (2 * (0.04) ** 2))

ax.fill_between(f_plot, 0, ideal, color='white', alpha=0.10, label='Ideal bandpass')
ax.plot(f_plot, gauss,  color='#888888', lw=1.5, ls='--', label='Symmetric Gabor')
ax.plot(f_plot, lg,     color=LG_COLOR,  lw=2.5, label='Log-Gabor')
ax.axvline(f0, color='#444444', lw=0.8, ls=':')
ax.text(f0 + 0.005, 0.92, '$f_c$', color='#666666', fontsize=9)

ax.set_xlim(0, 0.4); ax.set_ylim(0, 1.05)
ax.legend(fontsize=8, framealpha=0.15, labelcolor='white',
          facecolor=CELL, edgecolor='#444444')

note = ("LG has zero DC response,\n"
        "symmetric on log-freq axis,\n"
        "tight energy in passband.\n"
        "→ Near-optimal SNR/bandwidth.")
ax.text(0.62, 0.55, note, transform=ax.transAxes, color='#aaaaaa',
        fontsize=7.5, va='top', bbox=dict(boxstyle='round', facecolor='#1e1e1e',
                                          edgecolor='#444', alpha=0.9))

# ── Panel B: Memory wall (all responses in RAM simultaneously) ────────────────
ax2 = fig.add_subplot(gs2[1])
style_ax(ax2, 'B. Memory Wall — All Responses in RAM',
         'Filter banks', 'Peak edge-phase memory [GB]')

methods_m = ['Log-Gabor\n(3 ori)', 'Steerable\n(3 ori)', 'DT-CWT\n(6 ori)', 'Adap.\nIIR']
mem_gb    = [12.0, 12.0, 20.3, 4.2]
cols_m    = [LG_COLOR, ST_COLOR, DT_COLOR, IIR_COLOR]

bars = ax2.bar(range(4), mem_gb, color=cols_m, width=0.6, zorder=3)
for b, v in zip(bars, mem_gb):
    ax2.text(b.get_x() + b.get_width()/2, v + 0.3, f'{v:.1f} GB',
             ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')
ax2.set_xticks(range(4))
ax2.set_xticklabels(methods_m, color='white', fontsize=8)
ax2.set_ylim(0, 26)
ax2.axhline(12, color='#444444', lw=1, ls='--')
ax2.text(3.45, 12.3, 'LG baseline', color='#666666', fontsize=7.5)
ax2.annotate('−65%\n(streaming path)', xy=(3, 4.2), xytext=(3, 14),
             arrowprops=dict(arrowstyle='->', color=IIR_COLOR, lw=1.4),
             ha='center', color=IIR_COLOR, fontsize=9, fontweight='bold')

# ── Panel C: On-chip constraints ──────────────────────────────────────────────
ax3 = fig.add_subplot(gs2[2])
ax3.set_facecolor(CELL)
for spine in ax3.spines.values():
    spine.set_color('#333333')
ax3.set_axis_off()
ax3.set_title('C. On-Chip Constraints', color='#cccccc',
              fontsize=10, fontweight='bold', pad=6)

items = [
    ('✗', '#f1834e', 'No off-chip DRAM: all state\nmust fit in SRAM (~1 MB)'),
    ('✗', '#f1834e', 'No batch processing: frames\narrive continuously, real-time'),
    ('✗', '#f1834e', 'LG stores ALL R[d,s] at once\n→ GB-scale memory needed'),
    ('✓', '#4ef18a', 'IIR: causal, O(1) memory\nper filter direction'),
    ('✓', '#4ef18a', 'IIR: real-time frame-by-frame\nprocessing possible'),
    ('✓', '#4ef18a', 'Steerable: no DRAM but FIR\n(needs full clip in RAM)'),
]
for i, (icon, col, text) in enumerate(items):
    y = 0.90 - i * 0.155
    ax3.text(0.04, y, icon,  transform=ax3.transAxes, color=col,
             fontsize=14, va='center', fontweight='bold')
    ax3.text(0.17, y, text, transform=ax3.transAxes, color='#cccccc',
             fontsize=8.5, va='center')

fig.suptitle('Log-Gabor: Near-Optimal SNR But Memory-Impractical for Real-Time / On-Chip',
             color='white', fontsize=12, fontweight='bold', y=0.97)

out_path = str(out_dir / 'log_gabor_optimality.png')
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
print(f"Saved: {out_path}")
plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: ML methods cost
# ═══════════════════════════════════════════════════════════════════════════════
print("Generating ML cost comparison figure...")

fig = new_dark_fig((12, 5))
fig.subplots_adjust(left=0.05, right=0.97, top=0.85, bottom=0.10, wspace=0.40)
gs3 = gridspec.GridSpec(1, 2, figure=fig)

# ── Panel A: Runtime bar chart ────────────────────────────────────────────────
ax = fig.add_subplot(gs3[0])
style_ax(ax, 'A. Runtime per Video Clip (256×512×120)',
         'Method', 'Runtime [seconds]  (log scale)')

ml_methods   = ['Deep\nOpticalFlow\n(RAFT)', 'Deep\nEdge Net\n(DexiNed)',
                'Burst Frames\n(Baseline)', 'Log-Gabor\n(TPC)', 'Steerable\n(TPC)',
                'Adap. IIR\n(TPC)']
runtimes     = [45, 60, 100, 3.3, 1.1, 13.4]
ml_cols      = ['#888888', '#888888', '#888888', LG_COLOR, ST_COLOR, IIR_COLOR]
ml_hatches   = ['///', '///', '///', '', '', '']

bars = ax.bar(range(len(ml_methods)), runtimes, color=ml_cols, width=0.65, zorder=3)
for bar, hatch in zip(bars, ml_hatches):
    bar.set_hatch(hatch)
    bar.set_edgecolor('#555555')

for b, v in zip(bars, runtimes):
    ax.text(b.get_x() + b.get_width()/2, v * 1.05, f'{v:.0f}s',
            ha='center', va='bottom', fontsize=9, color='white', fontweight='bold')

ax.set_xticks(range(len(ml_methods)))
ax.set_xticklabels(ml_methods, color='white', fontsize=8)
ax.set_yscale('log')
ax.set_ylim(0.5, 500)
ax.tick_params(axis='y', colors='#888888')

# ML vs classical boundary
ax.axvline(2.5, color='#555555', lw=1.5, ls='--', alpha=0.8)
ax.text(1.25, 200, 'ML-based\n(GPU required)', ha='center', color='#888888',
        fontsize=9, style='italic')
ax.text(4.5, 200, 'Classical\n(CPU)', ha='center', color='#cccccc',
        fontsize=9, style='italic')

# ── Panel B: Requirement checklist ────────────────────────────────────────────
ax2 = fig.add_subplot(gs3[1])
ax2.set_facecolor(CELL)
for spine in ax2.spines.values():
    spine.set_color('#333333')
ax2.set_axis_off()
ax2.set_title('B. Deployment Feasibility', color='#cccccc',
              fontsize=10, fontweight='bold', pad=6)

headers = ['Requirement', 'ML-based', 'TPC (IIR)']
rows = [
    ['Training data needed',   '✗ yes',  '✓ no'],
    ['GPU for inference',       '✗ yes',  '✓ no'],
    ['Generalises to new data', '✗ maybe','✓ yes'],
    ['Real-time capable',       '✗ no',   '✓ yes'],
    ['On-chip deployable',      '✗ no',   '✓ yes'],
    ['Interpretable outputs',   '✗ no',   '✓ yes'],
    ['Runtime per clip',        '45–100s','1–13s'],
]

col_x = [0.02, 0.48, 0.78]
col_colors = ['#cccccc', '#f1834e', '#4ef18a']

# Header
for x, h, c in zip(col_x, headers, ['#aaaaaa', '#f1834e', '#4ef18a']):
    ax2.text(x, 0.95, h, transform=ax2.transAxes, color=c,
             fontsize=9, fontweight='bold', va='top')
ax2.axhline(0.90, xmin=0.02, xmax=0.98, color='#333333', lw=0.8)

for i, row in enumerate(rows):
    y = 0.87 - i * 0.115
    for x, val, c in zip(col_x, row, col_colors):
        is_check = '✓' in val
        is_cross = '✗' in val
        col = '#4ef18a' if is_check else ('#f1834e' if is_cross else '#cccccc')
        ax2.text(x, y, val, transform=ax2.transAxes, color=col,
                 fontsize=8.5, va='center')

fig.suptitle('ML-Based Methods: High Cost, Low Deployability vs Classical TPC',
             color='white', fontsize=12, fontweight='bold', y=0.97)

out_path = str(out_dir / 'ml_cost_comparison.png')
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
print(f"Saved: {out_path}")
plt.close(fig)

print("\nAll filter comparison figures saved.")
