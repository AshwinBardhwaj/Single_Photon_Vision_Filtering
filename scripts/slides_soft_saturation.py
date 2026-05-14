"""
Soft saturation figure for presentation slides.

Shows the nonlinear mapping from photon flux c (photons-per-pixel-per-frame)
to observed SPAD count fraction B/M, the saturation regime, and the IPPD
correction that recovers the linear signal.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os

out_dir = os.path.join(os.path.dirname(__file__), "output", "slides")
os.makedirs(out_dir, exist_ok=True)

# ── Physics ──────────────────────────────────────────────────────────────────
c = np.linspace(0, 5, 500)          # true photon flux (ppp per sub-frame)
p = 1.0 - np.exp(-c)                 # Pr(detection) = 1 - e^{-c}
# After M binary frames: B/M -> p (in expectation)
# IPPD: recover c_hat = -log(1 - B/M)
c_hat = -np.log(1.0 - p + 1e-12)    # = c  (exact inverse, shown for clarity)

# Noise std for a single pixel, M frames  (binomial std / M)
M_vals = [10, 50, 100]
colors = ['#4e9af1', '#f1834e', '#4ef18a']

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(12, 4.5), facecolor='#0d0d0d')
gs  = gridspec.GridSpec(1, 3, figure=fig, wspace=0.38, left=0.07, right=0.97,
                        top=0.88, bottom=0.15)

TEXT_KW = dict(color='white', fontsize=10)
TITLE_KW = dict(color='#cccccc', fontsize=11, fontweight='bold', pad=8)
AX_STYLE = dict(facecolor='#1a1a1a', tick_params=dict(colors='#888888'))

def style_ax(ax, title):
    ax.set_facecolor('#1a1a1a')
    ax.tick_params(colors='#888888', labelsize=9)
    for spine in ax.spines.values():
        spine.set_color('#444444')
    ax.set_title(title, **TITLE_KW)
    ax.grid(True, color='#2a2a2a', linewidth=0.6)

# ── Panel 1: raw count fraction vs flux ──────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
style_ax(ax1, "Binarisation Nonlinearity")

ax1.plot(c, p, color='#4e9af1', lw=2.5, label=r'$\langle B/M \rangle = 1 - e^{-c}$')
ax1.plot(c, c / (1 + c / 5) * (1 - np.exp(-5)) / 5 * 5,  # just to show linear region
         color='white', lw=1, ls='--', alpha=0.35)
# Show linear approximation at low flux
c_lin = np.linspace(0, 1, 100)
ax1.plot(c_lin, c_lin, color='#cccccc', lw=1.5, ls='--', alpha=0.7,
         label=r'Linear: $B/M \approx c$')

ax1.axvspan(0, 0.5,  alpha=0.08, color='#4ef18a')
ax1.axvspan(0.5, 2,  alpha=0.08, color='#f1d44e')
ax1.axvspan(2,   5,  alpha=0.08, color='#f1834e')

ax1.text(0.25,  0.65, 'Linear\nregime', color='#4ef18a', fontsize=8,
         ha='center', transform=ax1.transAxes, va='center')
ax1.text(0.52,  0.40, 'Soft\nsat.', color='#f1d44e', fontsize=8,
         ha='center', transform=ax1.transAxes, va='center')
ax1.text(0.85,  0.25, 'Hard\nsat.', color='#f1834e', fontsize=8,
         ha='center', transform=ax1.transAxes, va='center')

ax1.set_xlabel("True flux  $c$  [ppp/sub-frame]", **TEXT_KW)
ax1.set_ylabel(r"Observed $\langle B / M \rangle$", **TEXT_KW)
ax1.legend(fontsize=8, framealpha=0.15, labelcolor='white',
           facecolor='#1a1a1a', edgecolor='#444444')
ax1.set_xlim(0, 5); ax1.set_ylim(0, 1)

# ── Panel 2: IPPD correction recovers linear signal ──────────────────────────
ax2 = fig.add_subplot(gs[1])
style_ax(ax2, "IPPD Linearisation")

p_meas = np.linspace(0.01, 0.97, 400)
c_rec  = -np.log(1.0 - p_meas)

ax2.plot(p_meas, p_meas,  color='#f1834e', lw=2, label='Raw $B/M$ (biased)')
ax2.plot(p_meas, c_rec,   color='#4e9af1', lw=2.5,
         label=r'IPPD: $\hat{c} = -\ln(1 - B/M)$')
ax2.plot([0, 1], [0, 1],  color='#cccccc', lw=1, ls='--', alpha=0.5,
         label='Ideal linear')

ax2.set_xlabel(r"Observed fraction  $B/M$", **TEXT_KW)
ax2.set_ylabel(r"Recovered flux  $\hat{c}$", **TEXT_KW)
ax2.legend(fontsize=8, framealpha=0.15, labelcolor='white',
           facecolor='#1a1a1a', edgecolor='#444444')
ax2.set_xlim(0, 0.97); ax2.set_ylim(0, 3.5)

# Mark the regime used in our experiments (~0.03-0.65 ppp)
ax2.axvspan(0.03, 0.6, alpha=0.10, color='#4e9af1')
ax2.text(0.28, 0.88, 'Our\nexperiments', color='#4e9af1', fontsize=8,
         ha='center', transform=ax2.transAxes)

# ── Panel 3: noise std vs flux for different M ────────────────────────────────
ax3 = fig.add_subplot(gs[2])
style_ax(ax3, "Noise Std vs Flux (Binomial)")

c_range = np.linspace(0.001, 3, 400)
for M, col in zip(M_vals, colors):
    p_r = 1.0 - np.exp(-c_range)
    sigma = np.sqrt(p_r * (1 - p_r) / M)   # std of B/M per pixel
    # Convert to flux domain via IPPD Jacobian: dc/dp = 1/(1-p)
    sigma_c = sigma / (1 - p_r)
    ax3.plot(c_range, sigma_c, color=col, lw=2, label=f'M = {M}')

ax3.set_xlabel("True flux  $c$  [ppp/sub-frame]", **TEXT_KW)
ax3.set_ylabel(r"$\sigma_{\hat{c}}$  (std of IPPD estimate)", **TEXT_KW)
ax3.legend(fontsize=8, framealpha=0.15, labelcolor='white',
           facecolor='#1a1a1a', edgecolor='#444444')
ax3.set_xlim(0, 3); ax3.set_ylim(0, 0.5)

# Mark typical dataset range
ax3.axvspan(0, 0.8, alpha=0.08, color='#4e9af1')
ax3.text(0.18, 0.88, 'Our flux\nrange', color='#4e9af1', fontsize=8,
         ha='center', transform=ax3.transAxes)

fig.suptitle(
    "SPAD Sensing: Soft Saturation & Linearisation",
    color='white', fontsize=13, fontweight='bold', y=0.98
)

out_path = os.path.join(out_dir, "soft_saturation.png")
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=fig.get_facecolor())
print(f"Saved: {out_path}")
plt.close(fig)
