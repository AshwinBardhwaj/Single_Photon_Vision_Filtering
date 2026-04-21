import os
from pathlib import Path
import sys
import numpy as np
from scipy.io import savemat

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from synthetic_signals.synthetic_signals import synth_vid_circ_edge
from sensor.sensor import quanta_sample_direct, calc_bitdepth
from fileio.fileio import save_video_imageio as save_video


DATA_BASEDIR = root_dir / "scripts" / "data"
DATA_DIR = DATA_BASEDIR / "Fig07_edge_vs_flux_circle_sim"
DATA_DIR.mkdir(parents=True, exist_ok=True)


def main():
    rng = np.random.default_rng(12)

    L = 128
    N = 128
    IND_FRAME = N // 2
    radius = 40
    v = np.array([1.0, 0.0])
    alpha = 0.4
    c0 = 0.5

    X_3D, eX_3D = synth_vid_circ_edge(L, radius, c0, alpha, N, v)
    eX_2D = eX_3D[:, :, IND_FRAME]

    Cvals = np.array([0.01, 0.1, 1.0])
    Mvals = np.array([1, 7, 63])

    Y_3D = np.empty((len(Cvals), len(Mvals)), dtype=object)
    sim_params = np.empty((len(Cvals), len(Mvals)), dtype=object)

    ind_vis = np.arange(IND_FRAME - 10, IND_FRAME + 11)

    for i1, c in enumerate(Cvals):
        Xc = (c / c0) * X_3D
        for i2, M in enumerate(Mvals):
            y = quanta_sample_direct(M * Xc, None, int(M), rng=rng)
            Y_3D[i1, i2] = y
            sim_params[i1, i2] = {'c': float(c), 'M': int(M)}

            if M == 1:
                y_vis = y
            else:
                hi = np.percentile(y, 97)
                y_vis = np.clip(y / max(hi, 1e-12), 0, 1)

            clip = y_vis[:, :, ind_vis]
            out_path = DATA_DIR / f"obs_{calc_bitdepth(int(M))}bit_c{c * M:.2f}.mp4"
            save_video(np.moveaxis(clip, -1, 0), str(out_path))

    save_video(np.moveaxis(X_3D[:, :, ind_vis], -1, 0), str(DATA_DIR / "true_seq.mp4"))

    savemat(
        str(DATA_DIR / "data.mat"),
        {
            'X_3D': X_3D,
            'eX_3D': eX_3D,
            'eX_2D': eX_2D,
            'Cvals': Cvals,
            'Mvals': Mvals,
            'Y_3D': Y_3D,
            'sim_params': sim_params,
            'IND_FRAME': IND_FRAME + 1,
            'L': L,
            'N': N,
            'radius': radius,
            'v': v,
            'alpha': alpha,
            'c0': c0,
        },
    )
    print(f"Saved simulation data to {DATA_DIR / 'data.mat'}")


if __name__ == "__main__":
    main()