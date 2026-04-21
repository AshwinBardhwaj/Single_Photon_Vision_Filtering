import os
from pathlib import Path
import sys
import numpy as np
from scipy.io import loadmat, savemat
from scipy.ndimage import gaussian_filter

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from sensor.sensor import ippd, binom_noise_std, calc_bitdepth
from filtering.log_gabor_filter_bank import LogGaborBank3DSepT
from vision.feature_detection.phase_congruency import phase_congruency_3D_structure_tensor
from vision.feature_detection.structure_tensor import features_3D_structure_tensor
from visualize.visualize import vis_edge
from fileio.fileio import save_video_imageio as save_video


DATA_BASEDIR = root_dir / "scripts" / "data"
OUTPUT_BASEDIR = root_dir / "scripts" / "output"
DATA_DIR = DATA_BASEDIR / "Fig07_edge_vs_flux_circle_sim"
OUTPUT_DIR = OUTPUT_BASEDIR / "Fig07"

TPC_NOISE_THRESH_ZMIN = 2


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw = loadmat(str(DATA_DIR / "data.mat"))
    Cvals = raw['Cvals'].ravel()
    Mvals = raw['Mvals'].ravel()
    Y_3D = raw['Y_3D']
    IND_FRAME = int(raw['IND_FRAME'].item()) - 1
    eX_2D = raw['eX_2D']

    sample = Y_3D[0, 0] if Y_3D.ndim == 2 else Y_3D[0]
    input_size = sample.shape

    filts = LogGaborBank3DSepT()
    filts.input_size = input_size
    filts.min_wavelength = 6
    filts.num_scales = 3
    result = filts.set_up_filters()
    if result is not None:
        filts = result

    n_rows, n_cols = Y_3D.shape
    estr = np.empty((n_rows, n_cols), dtype=object)
    ind_vis = np.arange(IND_FRAME - 10, IND_FRAME + 11)

    for i1 in range(n_rows):
        c = float(Cvals[i1])
        for i2 in range(n_cols):
            M = int(Mvals[i2])
            y = Y_3D[i1, i2].astype(np.float32) / M

            local_flux = ippd(gaussian_filter(y, sigma=10))
            flux_noise_std = binom_noise_std(local_flux, M)

            R, _ = filts.response(y, flux_noise_std)

            num_dirs = filts.num_orientations * filts.num_velocities
            R_formatted = [
                [R.get((d, s), None) for s in range(filts.num_scales)]
                for d in range(num_dirs)
            ]

            PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt = phase_congruency_3D_structure_tensor(
                R_formatted,
                filts.tuning_directions(),
                filter_energies=filts.filter_energies,
                flux_noise_std=flux_noise_std,
                noise_thresh_zmin=TPC_NOISE_THRESH_ZMIN,
            )
            edges, _, _ = features_3D_structure_tensor(PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt)

            estr[i1, i2] = edges['strength']

            clip = edges['strength'][:, :, ind_vis]
            vis = vis_edge(clip)
            out_path = OUTPUT_DIR / f"edges_{calc_bitdepth(M)}bit_c{c * M:.2f}.mp4"
            save_video(np.moveaxis(vis, -1, 0), str(out_path))

    raw['estr'] = estr
    savemat(str(DATA_DIR / "data.mat"), raw)
    print(f"Saved edge strengths back to {DATA_DIR / 'data.mat'}")


if __name__ == "__main__":
    main()