import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "output" / "slides"
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
})

# ── Filter parameters (match LogGaborBank3DSepT defaults) ────────────────────
num_scales    = 3
min_wavelength = 4.0
mult          = 2.1
sigma_on_f    = 0.55
dtheta_max    = 60.0
orientations  = [0, 60, 120]

f_centers = [(1.0 / min_wavelength) / (mult ** (num_scales - 1 - s))
             for s in range(num_scales)]
log_sigma = np.log(sigma_on_f)

SCALE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c']
SCALE_LABELS = [f'Scale {s+1}  ($f_c$ = {fc:.3f})' for s, fc in enumerate(f_centers)]

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(13, 5), facecolor='white')
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38,
                        left=0.07, right=0.97, top=0.88, bottom=0.14)

def style(ax, xlabel, ylabel, title):
    ax.set_facecolor('white')
    ax.tick_params(labelsize=10, colors='#333333')
    ax.spines['left'].set_color('#bbbbbb')
    ax.spines['bottom'].set_color('#bbbbbb')
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=11, fontweight='bold', color='#222222', pad=8)
    ax.grid(True, color='#eeeeee', linewidth=0.8)

# ── Panel 1: Radial profiles (linear frequency axis) ────────────────────────
ax1 = fig.add_subplot(gs[0])
style(ax1, 'Spatial frequency  $f$  [cyc/px]', 'Filter response  $|H(f)|$',
      'Radial Profile')

f_lin = np.linspace(0.001, 0.5, 800)
lp = 1.0 / (1.0 + (f_lin / 0.45) ** 30)

for s, (fc, col, lbl) in enumerate(zip(f_centers, SCALE_COLORS, SCALE_LABELS)):
    H = np.exp(-(np.log(f_lin / fc)) ** 2 / (2 * log_sigma ** 2)) * lp
    ax1.plot(f_lin, H, color=col, lw=2.2, label=lbl)
    # Mark centre frequency
    ax1.axvline(fc, color=col, lw=0.8, ls='--', alpha=0.5)

ax1.set_xlim(0, 0.5)
ax1.set_ylim(0, 1.08)
ax1.legend(fontsize=9, framealpha=0.9, edgecolor='#cccccc')

# Half-power bandwidth annotation for scale 1
fc0 = f_centers[0]
H0  = np.exp(-(np.log(f_lin / fc0)) ** 2 / (2 * log_sigma ** 2)) * lp
half_idx = np.where(H0 > 0.5)[0]
if len(half_idx) > 1:
    f_lo, f_hi = f_lin[half_idx[0]], f_lin[half_idx[-1]]
    ax1.annotate('', xy=(f_hi, 0.5), xytext=(f_lo, 0.5),
                 arrowprops=dict(arrowstyle='<->', color=SCALE_COLORS[0], lw=1.4))
    ax1.text((f_lo + f_hi) / 2, 0.55, '−3 dB BW',
             ha='center', fontsize=8, color=SCALE_COLORS[0])

# ── Panel 2: Radial profiles (log frequency axis) ───────────────────────────
ax2 = fig.add_subplot(gs[1])
style(ax2, 'Spatial frequency  $f$  [cyc/px]', 'Filter response  $|H(f)|$',
      'Radial Profile  (log scale)')

for s, (fc, col, lbl) in enumerate(zip(f_centers, SCALE_COLORS, SCALE_LABELS)):
    H = np.exp(-(np.log(f_lin / fc)) ** 2 / (2 * log_sigma ** 2)) * lp
    ax2.plot(f_lin, H, color=col, lw=2.2, label=lbl)
    ax2.axvline(fc, color=col, lw=0.8, ls='--', alpha=0.5)

ax2.set_xscale('log')
ax2.set_xlim(0.01, 0.5)
ax2.set_ylim(0, 1.08)
ax2.legend(fontsize=9, framealpha=0.9, edgecolor='#cccccc')

# Show symmetry on log axis
ax2.text(0.42, 0.88, 'Symmetric on\nlog-frequency axis',
         transform=ax2.transAxes, fontsize=8.5, color='#555555',
         ha='right', style='italic')

# ── Panel 3: Angular profiles ─────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[2])
style(ax3, 'Δθ  from filter centre  [°]', 'Angular response  $|G(θ)|$',
      'Angular Profile  (all orientations)')

theta_deg = np.linspace(-180, 180, 720)
ORI_COLORS = ['#9467bd', '#8c564b', '#e377c2']

for ori, ori_col in zip(orientations, ORI_COLORS):
    delta = np.abs(theta_deg - ori)
    delta = np.where(delta > 180, 360 - delta, delta)   # wrap
    spread = np.where(delta < dtheta_max,
                      (np.cos(np.radians(180.0 * delta / dtheta_max)) + 1) / 2,
                      0.0)
    ax3.plot(theta_deg, spread, color=ori_col, lw=2.2, label=f'{ori}°')

ax3.set_xlim(-180, 180)
ax3.set_ylim(0, 1.12)
ax3.set_xticks([-180, -120, -60, 0, 60, 120, 180])
ax3.legend(title='Orientation', fontsize=9, title_fontsize=9,
           framealpha=0.9, edgecolor='#cccccc')

# Mark dtheta_max
ax3.axvline( dtheta_max, color='#aaaaaa', lw=1, ls=':')
ax3.axvline(-dtheta_max, color='#aaaaaa', lw=1, ls=':')
ax3.text(dtheta_max + 3, 0.92, f'±{dtheta_max:.0f}°',
         fontsize=8.5, color='#888888')

fig.suptitle('Log-Gabor Filter Bank — Frequency Bandwidth',
             fontsize=13, fontweight='bold', color='#111111', y=0.98)

out_path = str(out_dir / 'log_gabor_bandwidth.png')
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor='white')
print(f"Saved: {out_path}")
plt.close(fig)
