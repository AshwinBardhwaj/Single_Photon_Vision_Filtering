import os
from pathlib import Path
import sys
import numpy as np
import matplotlib.pyplot as plt

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from sensor.sensor import ppd, quanta_sample_direct
from fisher_information.fisher_information import fi_px_quanta_1bit


def angdiff(a, b):
    return np.angle(np.exp(1j * (a - b)))


def main():
    scriptdir = Path(__file__).resolve().parent
    scriptname = Path(__file__).stem
    plot_dir = scriptdir / "output" / "Fig11"
    plot_dir.mkdir(parents=True, exist_ok=True)

    FIG_W = 4
    FIG_H = 2

    S = 240
    P = 10
    N = P * S
    tn = np.arange(N) / S

    phi = 2 * np.pi / 3

    c = 1.0
    alpha = 0.5
    a = alpha * c

    Kvals = np.arange(1, int(S / 2))
    R = 1000

    I_pp = np.zeros(len(Kvals))
    phi_est = np.zeros((R, len(Kvals)))

    for i1, k in enumerate(Kvals):
        cosvals = np.cos(2 * np.pi * k * tn + phi)
        sinvals = np.sin(2 * np.pi * k * tn + phi)
        sinvals2 = sinvals ** 2

        F = c + a * cosvals
        F_rep = np.broadcast_to(F, (R, N))

        pr = ppd(F_rep)
        B = quanta_sample_direct(F_rep)
        FB = np.fft.fft(B, axis=1)
        coef = FB[:, k * P]
        phi_est[:, i1] = np.angle(coef)

        In = fi_px_quanta_1bit(F)
        I_pp[i1] = (a ** 2) * np.sum(In * sinvals2)

    phi_CRLB = np.minimum(180.0, np.degrees(1.0 / np.sqrt(I_pp)))
    phi_error = np.degrees(angdiff(phi_est, np.full_like(phi_est, phi)))
    phi_RMSE = np.sqrt(np.mean(phi_error ** 2, axis=0))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(Kvals, phi_CRLB, '-o', linewidth=2, label='CRLB')
    ax.plot(Kvals, phi_RMSE, '-o', linewidth=2, label='RMSE from DFT')
    ax.set_xlabel('frequency $k_0$')
    ax.set_ylabel('phase RMSE (deg)')
    ax.set_title(f'N = {S} (x {P} periods), c = {c:.1f}, a = {a:.1f}')
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{scriptname}_1bit.svg")
    plt.close(fig)
    print(f"Saved plot to {plot_dir / f'{scriptname}_1bit.svg'}")


if __name__ == "__main__":
    main()