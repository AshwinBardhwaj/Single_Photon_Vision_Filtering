import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "output" / "slides" / "paper"
out_dir.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    'font.family':          'serif',
    'mathtext.fontset':     'cm',
    'font.size':            12,
    'axes.linewidth':       1.0,
    'axes.spines.top':      True,
    'axes.spines.right':    True,
    'axes.grid':            False,
    'lines.linewidth':      2.5,
    'xtick.direction':      'in',
    'ytick.direction':      'in',
    'xtick.top':            True,
    'ytick.right':          True,
    'xtick.labelsize':      11,
    'ytick.labelsize':      11,
    'legend.frameon':       True,
    'legend.framealpha':    1.0,
    'legend.edgecolor':     'black',
    'legend.fontsize':      10,
    'figure.facecolor':     'white',
    'axes.facecolor':       'white',
})

f_c = 0.12
sig = 0.55

# ── Figure 1: linear scale ────────────────────────────────────────────────────
f = np.linspace(0, 0.5, 1000)
bw_gabor = f_c * 0.5
gabor = np.exp(-((f - f_c) ** 2) / (2 * bw_gabor ** 2))
with np.errstate(divide='ignore', invalid='ignore'):
    lg = np.exp(-(np.log(f / f_c)) ** 2 / (2 * np.log(sig) ** 2))
lg[f == 0] = 0
ideal = np.where((f > 0.07) & (f < 0.21), 1.0, 0.0)

fig1, ax1 = plt.subplots(figsize=(5.5, 4.0), facecolor='white')
fig1.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)

ax1.fill_between(f, 0, ideal, color='#e0e0e0', zorder=0, label='Ideal bandpass')
ax1.plot(f, gabor, color='#d62728', lw=2.5, ls='--',   label='Gabor')
ax1.plot(f, lg,    color='black',   lw=2.5, ls='-',    label='Log-Gabor')

ax1.set_xlim(0, 0.45)
ax1.set_ylim(0, 1.28)
ax1.set_xlabel('Spatial frequency $f$ [cyc/px]', fontsize=12)
ax1.set_ylabel('$|H(f)|$', fontsize=12)
ax1.set_title('Radial Profile — Linear Scale', fontsize=13, fontweight='bold', pad=8)
ax1.legend(loc='upper right')

fig1.savefig(str(out_dir / 'log_gabor_linear.png'), dpi=180,
             bbox_inches='tight', facecolor='white')
print("Saved: log_gabor_linear.png")
plt.close(fig1)

# ── Figure 2: log scale ───────────────────────────────────────────────────────
f_log = np.linspace(0.008, 0.48, 2000)
gabor_log = np.exp(-((f_log - f_c) ** 2) / (2 * bw_gabor ** 2))
with np.errstate(divide='ignore'):
    lg_log = np.exp(-(np.log(f_log / f_c)) ** 2 / (2 * np.log(sig) ** 2))

fig2, ax2 = plt.subplots(figsize=(5.5, 4.0), facecolor='white')
fig2.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)

ax2.plot(f_log, gabor_log, color='#d62728', lw=2.5, ls='--', label='Gabor')
ax2.plot(f_log, lg_log,    color='black',   lw=2.5, ls='-',  label='Log-Gabor')
ax2.axvline(f_c, color='#aaaaaa', lw=1.0, ls=':')

ax2.set_xscale('log')
ax2.set_xlim(0.008, 0.48)
ax2.set_ylim(0, 1.28)
ax2.set_xlabel('Spatial frequency $f$ [cyc/px]', fontsize=12)
ax2.set_ylabel('$|H(f)|$', fontsize=12)
ax2.set_title('Radial Profile — Log Scale', fontsize=13, fontweight='bold', pad=8)
ax2.legend(loc='upper left')

fig2.savefig(str(out_dir / 'log_gabor_log_scale.png'), dpi=180,
             bbox_inches='tight', facecolor='white')
print("Saved: log_gabor_log_scale.png")
plt.close(fig2)
