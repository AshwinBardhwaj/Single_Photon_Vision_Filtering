import time
import os
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter

# Adjust these imports based on your exact Python implementations
from sensor.sensor import ippd, binom_noise_std
from filtering.log_gabor_filter_bank import LogGaborBank3DSepT
from vision.feature_detection.phase_congruency import phase_congruency_3D_structure_tensor
from vision.feature_detection.structure_tensor import features_3D_structure_tensor
from vision.motion_estimation.flo_1d_to_uv import flo_1d_to_uv
from vision.motion_estimation.vel_fj1990_approx_withz import vel_fj1990_approx_withz
from visualize.visualize import vis_edge, vis_flo_dense
from utils.utils import rescale_prctile

# Using your specific imageio saver
from fileio.fileio import save_video_imageio as save_video


def execute_pipeline(data_file, output_dir, filts_config, params, crop_bounds):
    """
    Python equivalent of run_base.m
    """
    print(f"Loading data from {data_file}...")
    data = loadmat(data_file)
    B = data['B'].astype(np.float32)
    M = data['M'].item()

    out_imin, out_imax, out_jmin, out_jmax = crop_bounds
    n_skip = params['N_SKIP_OUTPUT']

    # Implementation assumes B is scaled to range [0, 1]
    if M != 1:
        B = B * (1.0 / M)

    # Flux level estimation
    c_est = ippd(np.mean(B[out_imin:out_imax, out_jmin:out_jmax, :]))
    print(f"Flux level: {c_est * M:.4f} ppp.")

    # Local flux and noise
    t0 = time.time()
    # imgaussfilt3 with sigma 10
    local_flux = ippd(gaussian_filter(B, sigma=10))
    flux_noise_std = binom_noise_std(local_flux, M)
    print(f"Estimate local mean flux: {time.time() - t0:.3f} seconds.")

    # Filter initialization
    t0 = time.time()
    H, W, N = B.shape

    filts = LogGaborBank3DSepT()

    # Apply the config parameters as attributes
    for key, value in filts_config.items():
        setattr(filts, key, value)

    # Set the input size and run setup
    filts.input_size = (H, W, N)

    # Check if your Python method returns a new object or modifies in-place
    result = filts.set_up_filters()
    if result is not None:
        filts = result

    print(f"Filter initialization: {time.time() - t0:.3f} seconds.")

    # Filter responses
    t0 = time.time()
    R, Rz = filts.response(B, flux_noise_std)
    print(f"Get filter responses: {time.time() - t0:.3f} seconds.")

    # Edge Detection (TPC)
    t0 = time.time()

    # Phase Congruency function expects a list of lists: R_formatted[direction][scale]
    num_dirs = filts.num_orientations * filts.num_velocities
    R_formatted = [
        [R.get((d, s), None) for s in range(filts.num_scales)]
        for d in range(num_dirs)
    ]

    pc_out = phase_congruency_3D_structure_tensor(
        R_formatted,
        filts.tuning_directions(),
        filter_energies=filts.filter_energies,
        flux_noise_std=flux_noise_std,
        noise_thresh_zmin=params['TPC_NOISE_THRESH_ZMIN']
    )
    PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt = pc_out

    edges, _, _ = features_3D_structure_tensor(PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt)
    print(f"edge_detection_TPC: {time.time() - t0:.3f} seconds.")

    # Edge Velocities (TPC)
    t0 = time.time()
    flo1 = flo_1d_to_uv(edges['normal_velocity'], edges['orientation'])
    strength_thresh = np.percentile(edges['strength'], 75)
    rel_flo1 = (edges['strength'] > strength_thresh) & \
               (np.abs(edges['normal_velocity']) < params['MAX_FLO']) & \
               (edges['coherence'] > 0.5)
    print(f"edge_velocities_TPC: {time.time() - t0:.3f} seconds.")

    # FJ1990 Approximation
    t0 = time.time()

    if Rz:
        Rz_dict = Rz
    else:
        print("Note: Rz is empty, falling back to R dictionary for FJ1990 optical flow.")
        Rz_dict = R

    # Create a wrapper class to bypass the conflicting requirements in vel_fj1990_approx_withz
    class RzDictWrapper:
        def __init__(self, d, nd, ns):
            self.d = d
            self.nd = nd
            self.ns = ns

        def __len__(self):
            return self.nd  # Resolves: len(Rz)

        def __getitem__(self, idx):
            if idx == 0:
                return [None] * self.ns  # Resolves: len(Rz[0])
            raise KeyError(idx)

        def get(self, key, default=None):
            return self.d.get(key, default)  # Resolves: Rz.get((i2, i1))

    Rz_wrapped = RzDictWrapper(Rz_dict, num_dirs, filts.num_scales)

    _, _, flo2_FJ_ms, rel2_FJ_ms = vel_fj1990_approx_withz(
        Rz_wrapped,
        filts.tuning_directions(),
        solve_2d_pxwise_k=params['FJ_SOLVE_2D_PXWISE_K']
    )
    print(f"vel_FJ1990_approx_withz_multiscale: {time.time() - t0:.3f} seconds.")

    # Cropping outputs (removing periodicity artifacts)
    B_crop = B[out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip]
    estr_crop = edges['strength'][out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip]
    flo1_crop = flo1[out_imin:out_imax, out_jmin:out_jmax, :, n_skip:-n_skip]
    rel_flo1_crop = rel_flo1[out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip]

    flo2_FJ_ms_crop = []
    rel2_FJ_ms_crop = []
    for s in range(len(flo2_FJ_ms)):
        flo2_FJ_ms_crop.append(flo2_FJ_ms[s][out_imin:out_imax, out_jmin:out_jmax, :, n_skip:-n_skip])
        rel2_FJ_ms_crop.append(rel2_FJ_ms[s][out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip])

    # Save Visualizations
    os.makedirs(output_dir, exist_ok=True)

    # Use np.moveaxis(..., -1, 0) to shift 'Time' axis from last to first for imageio
    B_v = rescale_prctile(B_crop)
    save_video(np.moveaxis(B_v, -1, 0), os.path.join(output_dir, 'B.mp4'))

    estr_v = vis_edge(estr_crop, False, params['ESTR_VIS_PRCTILE_THRESH'])
    save_video(np.moveaxis(estr_v, -1, 0), os.path.join(output_dir, 'estr_TPC.mp4'))

    flo1_v = vis_flo_dense(flo1_crop, rel_flo1_crop, params['MAX_FLO'], params['FLO1_VIS_THICKEN'])
    save_video(np.moveaxis(flo1_v, -1, 0), os.path.join(output_dir, 'flo1_TPC.mp4'))

    for s in range(len(flo2_FJ_ms_crop)):
        rel_mask = rel2_FJ_ms_crop[s] > params['FLO2_REL_THRESH']
        flo2_FJ_ms_c = vis_flo_dense(flo2_FJ_ms_crop[s], rel_mask, params['MAX_FLO'])
        save_video(np.moveaxis(flo2_FJ_ms_c, -1, 0), os.path.join(output_dir, f'flo2_FJ_ms_{s + 1}.mp4'))

    print(f"Finished processing and saving to {output_dir}")