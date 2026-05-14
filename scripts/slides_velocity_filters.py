"""
Velocity-tuned filter visualisation for presentation slides.

Produces two figures:
  1. Frequency-domain view: spatiotemporal filter support on (fx, ft) plane,
     one panel per velocity/scale combination.
  2. Videos (MP4) showing the real part of filter responses at each scale
     when run on the lowest-flux real dataset (ball-3, 0.030 ppp).
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path

root = Path(__file__).resolve().parent.parent
sys.path.append(str(root))

from scipy.io import loadmat
from scipy.ndimage import gaussian_filter
from scipy.fft import fft2, ifft2, fft, ifft
from sensor.sensor import ippd, binom_noise_std
from filtering.log_gabor_filter_bank import LogGaborBank3DSepT
from fileio.fileio import save_video_imageio as save_video
from utils.utils import rescale_prctile

out_dir = Path(__file__).resolve().parent / "output" / "slides"
out_dir.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────────────────────────────────────
# 1. FREQUENCY-DOMAIN SCHEMATIC
# ─────────────────────────────────────────────────────────────────────────────
print("Generating frequency-domain velocity filter schematic...")

BG = '#0d0d0d'
velocities  = [1.0, 1/3, 0.0, -1/3, -1.0]
vel_labels  = ['v = +1', 'v = +1/3', 'v = 0', 'v = −1/3', 'v = −1']
vel_colors  = ['#f1834e', '#f1d44e', '#4ef18a', '#4e9af1', '#b04ef1']

scales      = 3
min_wl      = 4.0
mult        = 2.1
sigma_f     = 0.55
theta_0     = 0.0   # show the 0° orientation slice

fig, axes = plt.subplots(1, scales, figsize=(14, 5), facecolor=BG)
fig.subplots_adjust(wspace=0.28, left=0.07, right=0.97, top=0.85, bottom=0.12)

N_f = 256
fx_1d = np.fft.fftfreq(N_f)
ft_1d = np.fft.fftfreq(N_f)
FX, FT = np.meshgrid(fx_1d, ft_1d, indexing='ij')
f_mag = np.abs(FX)

log_sigma = np.log(sigma_f)

for s in range(scales):
    ax  = axes[s]
    ind = scales - 1 - s
    f0  = (1.0 / min_wl) / (mult ** (scales - 1 - ind))
    lp  = 1.0 / (1.0 + (f_mag / 0.45) ** 30)

    with np.errstate(divide='ignore', invalid='ignore'):
        lg_r = np.exp(-(np.log(f_mag / f0)) ** 2 / (2 * log_sigma ** 2))
    lg_r *= lp
    lg_r[0, :] = 0

    ax.set_facecolor('#111111')
    for spine in ax.spines.values():
        spine.set_color('#333333')
    ax.tick_params(colors='#888888', labelsize=8)
    ax.grid(color='#222222', lw=0.5)
    ax.set_title(f'Scale {s+1}  (λ_c ≈ {min_wl * mult**ind:.1f} px)',
                 color='#cccccc', fontsize=10, fontweight='bold', pad=5)
    ax.set_xlabel('Spatial freq.  $f_x$', color='#999999', fontsize=9)
    if s == 0:
        ax.set_ylabel('Temporal freq.  $f_t$', color='#999999', fontsize=9)
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5)

    for v, label, col in zip(velocities, vel_labels, vel_colors):
        ft_c = -f0 * v
        if abs(ft_c) > 1/3:
            continue
        # Log-Gabor temporal slice along each velocity line
        if ft_c == 0:
            ft_mag = np.abs(np.array([-f0 * vv for vv in velocities if vv != 0]))
            sigmaf = np.min(ft_mag[ft_mag > 0]) / 2 if len(ft_mag) > 0 else 0.05
            lg_t = np.exp(-0.5 * (ft_1d / sigmaf) ** 2)
        else:
            ft_scaled = ft_1d / ft_c
            lg_t = np.zeros(N_f)
            mask = ft_scaled > 0
            lg_t[mask] = np.exp(-(np.log(ft_scaled[mask]) ** 2) / (2 * log_sigma ** 2))

        # 2D filter: outer product of radial spatial × temporal
        G_2d = np.outer(lg_r[:, 0] if lg_r.ndim > 1 else lg_r,
                        lg_t)

        # Draw the iso-contour at 0.3 amplitude on the (fx, ft) plane
        # We compute the filter on the 1D fx axis and the ft axis
        fx_ax = np.fft.fftfreq(N_f)
        ft_ax = np.fft.fftfreq(N_f)
        FX_2, FT_2 = np.meshgrid(fx_ax, ft_ax, indexing='ij')

        f_r = np.abs(FX_2)
        with np.errstate(divide='ignore', invalid='ignore'):
            rad = np.exp(-(np.log(f_r / f0 + 1e-12)) ** 2 / (2 * log_sigma ** 2))
        rad[f_r == 0] = 0

        ft_c_arr = ft_c
        if ft_c_arr == 0:
            temp = np.exp(-0.5 * (FT_2 / sigmaf) ** 2)
        else:
            ft_sc = FT_2 / ft_c_arr
            temp = np.zeros_like(FT_2)
            m = ft_sc > 0
            temp[m] = np.exp(-(np.log(ft_sc[m]) ** 2) / (2 * log_sigma ** 2))

        G = rad * temp
        cs = ax.contour(np.fft.fftshift(FX_2, axes=0)[:, N_f//2],
                        np.fft.fftshift(FT_2, axes=1)[N_f//2, :],
                        np.fft.fftshift(G),
                        levels=[0.25], colors=[col], linewidths=2, alpha=0.9,
                        zorder=5)

        # Label the velocity line
        if v != 0:
            ax.annotate(label, xy=(0.08, -v * 0.08 * N_f / N_f),
                        color=col, fontsize=7.5, fontweight='bold',
                        xycoords='data')

    # Diagonal velocity lines (guide)
    for v, col in zip(velocities, vel_colors):
        if abs(-f0 * v) <= 1/3:
            xline = np.linspace(-0.48, 0.48, 200)
            ax.plot(xline, -v * xline, color=col, lw=0.8, alpha=0.25, ls='--')

# Legend
from matplotlib.lines import Line2D
handles = [Line2D([0], [0], color=c, lw=2, label=l)
           for c, l in zip(vel_colors, vel_labels)
           if abs(0.0) <= 1/3 or True]
fig.legend(handles=handles, loc='upper center', ncol=5,
           bbox_to_anchor=(0.5, 1.01),
           framealpha=0.15, labelcolor='white', fontsize=9,
           facecolor='#1a1a1a', edgecolor='#444444')

fig.suptitle("Velocity-Tuned Filters: Spatiotemporal Frequency Support",
             color='white', fontsize=13, fontweight='bold', y=1.08)

out_path = str(out_dir / "velocity_filter_freq.png")
fig.savefig(out_path, dpi=180, bbox_inches='tight', facecolor=BG)
print(f"Saved: {out_path}")
plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# 2. FILTER RESPONSE VIDEOS on ball-3 (0.030 ppp real data)
# ─────────────────────────────────────────────────────────────────────────────
print("\nGenerating filter response videos on ball-3 (0.030 ppp)...")

data_file = root / "scripts" / "data" / "fig08_vision_0604-ball-3.mat"
data = loadmat(str(data_file))
B = data['B'].astype(np.float32)
M = data['M'].item()
if M != 1:
    B = B / M

H, W, N = B.shape
local_flux = ippd(gaussian_filter(B, sigma=10))
flux_noise_std = binom_noise_std(local_flux, M)

# Build filter bank (only need spatial + temporal, no full response needed)
filts = LogGaborBank3DSepT()
filts.orientations = np.array([0, 60, 120])
filts.velocities   = np.array([1.0, -1.0, 1/3, -1/3, 0.0])
filts.num_scales   = 3
filts.input_size   = (H, W, N)
filts.set_up_filters()

print("  Computing filter responses (this may take ~30s)...")
R, Rz = filts.response(B, flux_noise_std)

n_skip = 15
N_ORIENTATIONS = filts.num_orientations
N_SCALES       = filts.num_scales
N_VELS         = filts.num_velocities

# Save one video per (scale, velocity) showing |response| averaged over orientations
vel_labels_short = ['v+1', 'v-1', 'v+1_3', 'v-1_3', 'v0']

for ind_s in range(N_SCALES):
    for ind_v in range(N_VELS):
        # Accumulate |response| across orientations
        resp_sum = None
        count = 0
        for ind_o in range(N_ORIENTATIONS):
            ind_filt = filts.ind_tuning_to_ind_filt(ind_v + 1, ind_o + 1)
            key = (ind_filt, ind_s)
            if key in R and R[key] is not None:
                mag = np.abs(R[key]).astype(np.float32)
                if resp_sum is None:
                    resp_sum = mag
                else:
                    resp_sum = resp_sum + mag
                count += 1
        if resp_sum is None or count == 0:
            continue
        resp_avg = resp_sum / count

        v_label = vel_labels_short[ind_v]
        out_vid = str(out_dir / f"filter_response_s{ind_s+1}_{v_label}.mp4")

        crop = resp_avg[:, :, n_skip:-n_skip]
        vis  = rescale_prctile(crop)
        vis_frames = (vis * 255).astype(np.uint8)

        # Apply inferno colormap: (H,W,T) → (H,W,T,4) → drop alpha → (T,H,W,3)
        cmap = plt.cm.inferno
        colored = (cmap(vis_frames.astype(np.float32) / 255.0)[..., :3] * 255).astype(np.uint8)
        save_video(np.moveaxis(colored, 2, 0), out_vid)  # (T, H, W, 3)
        print(f"  Saved: {out_vid}")

print("\nDone.")
