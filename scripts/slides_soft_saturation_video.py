import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage import uniform_filter1d, gaussian_filter1d
from matplotlib.lines import Line2D
from pathlib import Path

out_dir = Path(__file__).resolve().parent / "output" / "slides"
out_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(42)

N  = 200
t  = np.linspace(0, 1, N)
f0 = 1.0

A        = 1.4
s_true   = A * np.cos(2 * np.pi * f0 * t)

# Detection probability: shift signal positive, scale to reasonable ppp
c_inst  = np.clip(s_true + A, 0, None) / 1.5
p_det   = 1.0 - np.exp(-c_inst)
samples = (rng.random(N) < p_det).astype(float)

# Recovery: smooth the binary samples heavily → tracks rate ∝ (s_true + A)
# Subtract DC to centre at zero — phase aligns but amplitude is naturally distorted
w           = 20
smoothed    = gaussian_filter1d(samples.astype(float), sigma=w)
s_recovered = smoothed - smoothed.mean()   # centre; do NOT rescale amplitude

RED   = '#d62728'
BLUE  = '#1a1aaa'
GREEN = '#2ca02c'

fig, ax = plt.subplots(figsize=(12, 4), facecolor='white')
fig.subplots_adjust(left=0.08, right=0.97, top=0.93, bottom=0.13)

ax.set_facecolor('white')
ax.set_xlim(0, 1)
ax.set_ylim(-2.2, 2.8)
ax.set_xlabel('t', fontsize=13)
ax.set_ylabel('flux', fontsize=13)
ax.tick_params(labelsize=10, colors='#444444')
for sp in ax.spines.values():
    sp.set_color('#bbbbbb')
ax.axhline(0, color='#dddddd', lw=0.8)

ax.plot(t, s_true, color=RED, lw=2.2, zorder=5)

sample_lines, sample_dots = [], []
for i in range(N):
    col = BLUE if samples[i] == 1 else '#cccccc'
    ln, = ax.plot([], [], color=col, lw=0.9, alpha=0.85, zorder=3)
    dt, = ax.plot([], [], 'o', color=col, ms=3.5, alpha=0.9, zorder=4)
    sample_lines.append(ln)
    sample_dots.append(dt)

rec_line, = ax.plot([], [], color=GREEN, lw=2.2, zorder=6)

ax.legend(handles=[
    Line2D([0], [0], color=RED,   lw=2.2, label='radiance'),
    Line2D([0], [0], color=BLUE,  lw=1.2, marker='o', ms=4, label='single-photon samples'),
    Line2D([0], [0], color=GREEN, lw=2.2, label='recovered signal'),
], fontsize=9, loc='upper right', framealpha=0.9, edgecolor='#cccccc')

SWEEP = 160
HOLD  = 30
TOTAL = SWEEP + HOLD

def animate(af):
    n = min(N, int(af / SWEEP * N) + 1) if af < SWEEP else N
    for i in range(N):
        if i < n:
            sample_lines[i].set_data([t[i], t[i]], [0, samples[i]])
            sample_dots[i].set_data([t[i]], [samples[i]])
        else:
            sample_lines[i].set_data([], [])
            sample_dots[i].set_data([], [])
    rec_line.set_data(t[w:n] if n > w else [], s_recovered[w:n] if n > w else [])
    return sample_lines + sample_dots + [rec_line]

anim = animation.FuncAnimation(fig, animate, frames=TOTAL, interval=40, blit=True)
writer = animation.FFMpegWriter(fps=25, bitrate=3000,
                                 extra_args=['-vcodec', 'libx264', '-pix_fmt', 'yuv420p'])
print("Rendering...")
anim.save(str(out_dir / 'soft_saturation_animation.mp4'), writer=writer,
          dpi=150, savefig_kwargs={'facecolor': 'white'})
print("Done.")
plt.close(fig)
