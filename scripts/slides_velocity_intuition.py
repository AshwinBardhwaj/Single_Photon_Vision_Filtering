import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.fft import fft2, fftshift
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "output" / "slides"
out_dir.mkdir(parents=True, exist_ok=True)

# ── Synthetic scenes ──────────────────────────────────────────────────────────
NX, NT = 200, 200

def make_edge(v, width=6):
    x = np.arange(NX)
    t = np.arange(NT)
    XX, TT = np.meshgrid(x, t, indexing='ij')   # shape (NX, NT)
    cx = NX // 2
    pos = cx + v * (TT - NT // 2)
    scene = np.clip(1.0 - np.abs(XX - pos) / width, 0, 1) ** 0.5
    return scene

def make_spectrum(scene):
    F  = fftshift(fft2(scene))
    Fm = np.abs(F)
    Fm = np.log1p(Fm * 30)
    Fm /= Fm.max()
    return Fm

scene_0 = make_edge(v=0)
scene_1 = make_edge(v=1)
spec_0  = make_spectrum(scene_0)
spec_1  = make_spectrum(scene_1)

# ── Figure ────────────────────────────────────────────────────────────────────
BG   = '#1a1a1a'
DARK = '#0d0d0d'
ROW_BG = '#2a2a2a'

fig = plt.figure(figsize=(10, 6.2), facecolor=BG)

# Manual axes positions: [left, bottom, width, height]
pad   = 0.02
gap   = 0.06   # gap between columns
rh    = 0.36   # row height
rw    = 0.40   # panel width
left0 = 0.13
left1 = left0 + rw + gap
bot0  = 0.54   # top row bottom
bot1  = 0.08   # bottom row bottom

axes = {
    'a_xt': fig.add_axes([left0, bot0, rw, rh]),
    'a_ft': fig.add_axes([left1, bot0, rw, rh]),
    'b_xt': fig.add_axes([left0, bot1, rw, rh]),
    'b_ft': fig.add_axes([left1, bot1, rw, rh]),
}

for ax in axes.values():
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)

# ── Images ────────────────────────────────────────────────────────────────────
axes['a_xt'].imshow(scene_0.T, origin='lower', cmap='gray', aspect='auto',
                    vmin=0, vmax=1, interpolation='bilinear')
axes['a_ft'].imshow(spec_0.T,  origin='lower', cmap='afmhot', aspect='auto',
                    vmin=0, vmax=1, interpolation='bilinear')
axes['b_xt'].imshow(scene_1.T, origin='lower', cmap='gray', aspect='auto',
                    vmin=0, vmax=1, interpolation='bilinear')
axes['b_ft'].imshow(spec_1.T,  origin='lower', cmap='afmhot', aspect='auto',
                    vmin=0, vmax=1, interpolation='bilinear')

# ── Axis arrows on row (a) spacetime panel ────────────────────────────────────
ax = axes['a_xt']
ax.annotate('', xy=(0.80, 0.18), xytext=(0.62, 0.18),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='#f5c542', lw=1.8))
ax.text(0.84, 0.16, 'j', color='#f5c542', fontsize=12, fontweight='bold',
        transform=ax.transAxes, va='center')
ax.annotate('', xy=(0.62, 0.32), xytext=(0.62, 0.14),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='#f5c542', lw=1.8))
ax.text(0.60, 0.36, 'n', color='#f5c542', fontsize=12, fontweight='bold',
        transform=ax.transAxes, ha='center')

# ── Axis arrows on row (a) freq panel ────────────────────────────────────────
ax = axes['a_ft']
ax.annotate('', xy=(0.80, 0.18), xytext=(0.62, 0.18),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='#f5c542', lw=1.8))
ax.text(0.84, 0.16, r'$k_x$', color='#f5c542', fontsize=12, fontweight='bold',
        transform=ax.transAxes, va='center')
ax.annotate('', xy=(0.62, 0.32), xytext=(0.62, 0.14),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='#f5c542', lw=1.8))
ax.text(0.58, 0.36, r'$k_t$', color='#f5c542', fontsize=12, fontweight='bold',
        transform=ax.transAxes, ha='center')

# Dashed horizontal line on v=0 spectrum (energy at kt=0)
axes['a_ft'].axhline(y=NX // 2 - 0.5, color='#f5c542', lw=1.2,
                     ls='--', alpha=0.7, xmin=0.05, xmax=0.95)

# ── Velocity arrow on row (b) spacetime ──────────────────────────────────────
ax = axes['b_xt']
ax.annotate('', xy=(0.75, 0.24), xytext=(0.42, 0.24),
            xycoords='axes fraction',
            arrowprops=dict(arrowstyle='->', color='#d62728', lw=2.0))
ax.text(0.585, 0.30, '1 px/fr.', color='#d62728', fontsize=10,
        transform=ax.transAxes, ha='center', fontweight='bold')

# Dashed diagonal line on v=1 spectrum (energy ridge kt = -kx)
cx, ct = NX // 2, NT // 2
xs = np.array([0, NX - 1])
ys = ct - (xs - cx)                  # slope -1 → kt = -kx
axes['b_ft'].plot(xs, ys, color='#f5c542', lw=1.2, ls='--', alpha=0.6)

# ── Column headers ────────────────────────────────────────────────────────────
fig.text(left0 + rw / 2, bot0 + rh + 0.03, r'$f\,[j,\,n]$',
         ha='center', va='bottom', fontsize=13, color='#dddddd', fontweight='bold')
fig.text(left1 + rw / 2, bot0 + rh + 0.03, r'$|F\,[k_x,\,k_t]|$',
         ha='center', va='bottom', fontsize=13, color='#dddddd', fontweight='bold')

# ── Row background rounded rectangles ─────────────────────────────────────────
for bot in (bot0, bot1):
    rect = mpatches.FancyBboxPatch(
        (left0 - 0.015, bot - 0.025),
        rw * 2 + gap + 0.03, rh + 0.05,
        boxstyle='round,pad=0.01', linewidth=1.2,
        edgecolor='#555555', facecolor=ROW_BG,
        transform=fig.transFigure, zorder=0, clip_on=False
    )
    fig.add_artist(rect)

# ── Row labels (brown pill) ───────────────────────────────────────────────────
for bot, label in [(bot0, r'(a)  $v = 0$'), (bot1, r'(b)  $v = 1$')]:
    cy = bot + rh / 2
    pill = mpatches.FancyBboxPatch(
        (0.01, cy - 0.045), 0.10, 0.09,
        boxstyle='round,pad=0.015', linewidth=0,
        facecolor='#6b3a1f', transform=fig.transFigure,
        zorder=10, clip_on=False
    )
    fig.add_artist(pill)
    fig.text(0.06, cy, label, ha='center', va='center',
             fontsize=11, color='white', fontweight='bold', zorder=11)

out_path = str(out_dir / 'velocity_intuition.png')
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
print(f"Saved: {out_path}")
plt.close(fig)
