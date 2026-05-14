"""
Separate professional figures for steerable and IIR filters.
Style matches the reference academic figure (serif, box frame, inward ticks).
Output: scripts/output/slides/paper/
"""

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

# ── Shared filter parameters ──────────────────────────────────────────────────
min_wl     = 4.0
mult       = 2.1
n_scales   = 3
sigma_on_f = 0.55
log_sigma  = np.log(sigma_on_f)
b_val      = 0.8
order      = 2

f0s    = [(1.0 / min_wl) / (mult ** (n_scales - 1 - s)) for s in range(n_scales)]
f0_mid = f0s[1]
f0_fine = f0s[2]

# Line styles matching reference figure (solid/dotted/dashed)
STYLES = [
    dict(color='black',   ls='-',  lw=2.5),
    dict(color='#1f77b4', ls=':',  lw=2.5),
    dict(color='#d62728', ls='--', lw=2.5),
]
REF_STYLE = dict(color='#888888', ls='--', lw=1.5)

SCALE_LABELS = [f'Scale {s+1}  ($f_c={f0:.3f}$)' for s, f0 in enumerate(f0s)]


# ══════════════════════════════════════════════════════════════════════════════
# 1. Steerable G2 — radial profile (linear scale)
# ══════════════════════════════════════════════════════════════════════════════
f_lin = np.linspace(0.001, 0.5, 800)
lp    = 1.0 / (1.0 + (f_lin / 0.45) ** 30)

fig, ax = plt.subplots(figsize=(5.5, 4.0), facecolor='white')
fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)

for s, (f0, sty, lbl) in enumerate(zip(f0s, STYLES, SCALE_LABELS)):
    r  = f_lin / f0
    H  = (r ** order) * np.exp(-0.5 * r ** 2)
    H /= H.max() + 1e-12
    ax.plot(f_lin, H, label=lbl, **sty)

lg_ref = np.exp(-(np.log(f_lin / f0_mid)) ** 2 / (2 * log_sigma ** 2)) * lp
ax.plot(f_lin, lg_ref, label='Log-Gabor (ref)', **REF_STYLE)

ax.set_xlim(0, 0.5)
ax.set_ylim(0, 1.25)
ax.set_xlabel('Spatial frequency $f$ [cyc/px]', fontsize=12)
ax.set_ylabel('$|H(f)|$', fontsize=12)
ax.set_title('Steerable G$_2$ — Radial Profile', fontsize=13, fontweight='bold', pad=8)
ax.legend(loc='upper right')

fig.savefig(str(out_dir / 'steerable_radial_profile.png'), dpi=180,
            bbox_inches='tight', facecolor='white')
print("Saved: steerable_radial_profile.png")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 2. Steerable G2 — spatial kernels (analytical G2, 3 orientations)
# ══════════════════════════════════════════════════════════════════════════════
sigma_sp = 6.0
half_win = 24
g2_pts   = 400
coords   = np.linspace(-half_win, half_win, g2_pts)
X, Y     = np.meshgrid(coords, coords)
G        = np.exp(-(X ** 2 + Y ** 2) / (2 * sigma_sp ** 2))

orientations_deg = [0, 60, 120]
ORI_LABELS       = [r'$\theta = 0^\circ$', r'$\theta = 60^\circ$', r'$\theta = 120^\circ$']

for theta_deg, label in zip(orientations_deg, ORI_LABELS):
    theta_rad = np.radians(theta_deg)
    X_n    = X * np.cos(theta_rad) + Y * np.sin(theta_rad)
    kernel = ((X_n ** 2 - sigma_sp ** 2) / sigma_sp ** 4) * G
    kernel /= np.abs(kernel).max() + 1e-12

    fig, ax = plt.subplots(figsize=(4.2, 4.0), facecolor='white')
    fig.subplots_adjust(left=0.14, right=0.88, top=0.92, bottom=0.13)

    im = ax.imshow(kernel, cmap='RdBu_r', vmin=-1, vmax=1,
                   extent=[-half_win, half_win, -half_win, half_win],
                   origin='lower', aspect='equal', interpolation='bilinear')

    for sp in ax.spines.values():
        sp.set_color('black')
        sp.set_linewidth(1.0)
    ax.tick_params(direction='in', top=True, right=True, labelsize=10)
    ax.set_xlabel('$x$ [px]', fontsize=11)
    ax.set_ylabel('$y$ [px]', fontsize=11)
    ax.set_title(f'Steerable G$_2$ — {label}', fontsize=13, fontweight='bold', pad=8)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_ticks([-1, 0, 1])
    cbar.ax.tick_params(labelsize=9, direction='in')
    cbar.outline.set_edgecolor('black')
    cbar.outline.set_linewidth(1.0)

    fname = f'steerable_spatial_kernel_{theta_deg}deg.png'
    fig.savefig(str(out_dir / fname), dpi=180, bbox_inches='tight', facecolor='white')
    print(f"Saved: {fname}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 3. IIR — temporal frequency response
# ══════════════════════════════════════════════════════════════════════════════
N_t = 512
ft  = np.fft.fftfreq(N_t)
z   = np.exp(-1j * 2 * np.pi * ft)

velocities_show = [1.0, 1.0 / 3.0]
vel_styles      = [STYLES[0], STYLES[2]]
vel_labels      = ['$v = 1$', '$v = 1/3$']

fig, ax = plt.subplots(figsize=(5.5, 4.0), facecolor='white')
fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)

for vel, sty, lbl in zip(velocities_show, vel_styles, vel_labels):
    ft_c    = -f0_fine * vel
    omega_0 = 2.0 * np.tan(np.pi * ft_c)
    q       = b_val / (b_val - 1j * omega_0 + 2)
    r       = (b_val - 1j * omega_0 - 2) / (b_val - 1j * omega_0 + 2)
    num     = (q ** 3) * (1 + 3 * z + 3 * z ** 2 + z ** 3)
    den     = 1 + 3 * r * z + 3 * (r ** 2) * z ** 2 + (r ** 3) * z ** 3
    H       = np.abs(num / den)
    H      /= H.max() + 1e-12
    idx     = np.argsort(ft)
    ax.plot(ft[idx], H[idx], label=lbl, **sty)

ft_ref  = np.linspace(-0.5, 0.5, 1000)
ft_c_ref = -f0_fine * 1.0
mask    = ft_ref / ft_c_ref > 0
lg_t    = np.zeros_like(ft_ref)
lg_t[mask] = np.exp(-(np.log(ft_ref[mask] / ft_c_ref)) ** 2 / (2 * log_sigma ** 2))
ax.plot(ft_ref, lg_t, label='Log-Gabor (ref, $v=1$)', **REF_STYLE)

ax.set_xlim(-0.5, 0.5)
ax.set_ylim(0, 1.25)
ax.set_xlabel('Temporal frequency $f_t$ [cyc/frame]', fontsize=12)
ax.set_ylabel('$|H(f_t)|$', fontsize=12)
ax.set_title('Adaptive IIR — Temporal Freq. Response', fontsize=13, fontweight='bold', pad=8)
ax.legend(loc='upper left')

fig.savefig(str(out_dir / 'iir_freq_response.png'), dpi=180,
            bbox_inches='tight', facecolor='white')
print("Saved: iir_freq_response.png")
plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
# 4. IIR — temporal impulse response
# ══════════════════════════════════════════════════════════════════════════════
N_imp  = 256
ft_imp = np.fft.fftfreq(N_imp)
z_imp  = np.exp(-1j * 2 * np.pi * ft_imp)

fig, ax = plt.subplots(figsize=(5.5, 4.0), facecolor='white')
fig.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.15)

for vel, sty, lbl in zip(velocities_show, vel_styles, vel_labels):
    ft_c    = -f0_fine * vel
    omega_0 = 2.0 * np.tan(np.pi * ft_c)
    q       = b_val / (b_val - 1j * omega_0 + 2)
    r       = (b_val - 1j * omega_0 - 2) / (b_val - 1j * omega_0 + 2)
    num     = (q ** 3) * (1 + 3 * z_imp + 3 * z_imp ** 2 + z_imp ** 3)
    den     = 1 + 3 * r * z_imp + 3 * (r ** 2) * z_imp ** 2 + (r ** 3) * z_imp ** 3
    h       = np.real(np.fft.ifft(num / den))
    h      /= np.abs(h).max() + 1e-12
    n_show  = 64
    ax.plot(np.arange(n_show), h[:n_show], label=lbl, **sty)

ax.axhline(0, color='black', lw=0.8)
ax.set_xlim(0, 63)
ax.set_ylim(-1.15, 1.15)
ax.set_xlabel('Time [frames]', fontsize=12)
ax.set_ylabel('$h[n]$', fontsize=12)
ax.set_title('Adaptive IIR — Impulse Response', fontsize=13, fontweight='bold', pad=8)
ax.legend(loc='upper right')

fig.savefig(str(out_dir / 'iir_impulse_response.png'), dpi=180,
            bbox_inches='tight', facecolor='white')
print("Saved: iir_impulse_response.png")
plt.close(fig)

print("\nAll figures saved to:", out_dir)
