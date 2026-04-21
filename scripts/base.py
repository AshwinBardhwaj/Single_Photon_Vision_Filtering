import os
import time
import numpy as np
import scipy.io as sio
from scipy.ndimage import gaussian_filter

# Assuming your converted functions/classes are imported
# from log_gabor_filter_bank import LogGaborBank3DSepT
# from structure_tensor import phase_congruency_3d_structure_tensor, features_3d_structure_tensor
# from velocity_utils import flo_1d_to_uv, vel_fj1990_approx_withz, ippd, binom_noise_std
# from visualization_utils import rescale_prctile, vis_edge, vis_flo_dense, save_video

# 1. LOAD DATA
data_dict = sio.loadmat(DATA_FILE)
B = data_dict['B'].astype(np.float32)  # Temporally low-passed binary video
M = data_dict['M'].item()              # Number of frames summed

# Scale B to range [0, 1]
if M != 1:
    B = B * (1.0 / M)

# Estimate global flux (using pre-defined crop indices from config)
crop_B = B[OUT_IMIN:OUT_IMAX, OUT_JMIN:OUT_JMAX, :]
c_est = ippd(np.mean(crop_B))
print(f"Flux level: {c_est * M:.4f} ppp.")

# 2. LOCAL NOISE ESTIMATION
t0 = time.time()
# imgaussfilt3(B, 10) equivalent
local_flux = ippd(gaussian_filter(B, sigma=10))
flux_noise_std = binom_noise_std(local_flux, M)
print(f"Estimate local mean flux: {time.time() - t0:.3f} seconds.")

# 3. FILTER INITIALIZATION
t0 = time.time()
H, W, N = B.shape
filts.input_size = [H, W, N]
filts.set_up_filters()
print(f"Filter initialization: {time.time() - t0:.3f} seconds.")

# 4. GET FILTER RESPONSES
t0 = time.time()
R, Rz = filts.response(B, flux_noise_std)
print(f"Get filter responses: {time.time() - t0:.3f} seconds.")

# 5. STRUCTURE TENSOR & EDGE DETECTION
t0 = time.time()
PC_components = phase_congruency_3d_structure_tensor(
    R, filts.tuning_directions(),
    filter_energies=filts.filter_energies,
    flux_noise_std=flux_noise_std,
    noise_thresh_zmin=TPC_NOISE_THRESH_ZMIN
)

# PC_components = (PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt)
edges, _, _ = features_3d_structure_tensor(*PC_components)
print(f"edge_detection_TPC: {time.time() - t0:.3f} seconds.")

# 6. EDGE VELOCITIES (1D to 2D)
t0 = time.time()
flo1 = flo_1d_to_uv(edges.normal_velocity, edges.orientation)

# Reliability Mask
strength_thresh = np.percentile(edges.strength, 75)
rel_flo1 = (edges.strength > strength_thresh) & \
           (np.abs(edges.normal_velocity) < MAX_FLO) & \
           (edges.coherence > 0.5)
print(f"edge_velocities_TPC: {time.time() - t0:.3f} seconds.")

# 7. FLEET & JEPSON VELOCITY APPROXIMATION
t0 = time.time()
# Returns: V, Rel, V0_list, Rel0_list
_, _, flo2_FJ_ms, rel2_FJ_ms = vel_fj1990_approx_withz(
    Rz, filts.tuning_directions(), 
    solve_2d_pxwise_k=FJ_SOLVE_2D_PXWISE_K
)
print(f"vel_FJ1990_approx_withz_multiscale: {time.time() - t0:.3f} seconds.")

# 8. CROP OUTPUT TO REMOVE PERIODICITY ARTIFACTS
t_slice = slice(N_SKIP_OUTPUT, -N_SKIP_OUTPUT)
spatial_slice = (slice(OUT_IMIN, OUT_IMAX), slice(OUT_JMIN, OUT_JMAX))

B_cropped = B[spatial_slice + (t_slice,)]
estr_cropped = edges.strength[spatial_slice + (t_slice,)]
flo1_cropped = flo1[spatial_slice + (slice(None),) + (t_slice,)] # axis 2 is u,v
rel_flo1_cropped = rel_flo1[spatial_slice + (t_slice,)]

for s in range(len(flo2_FJ_ms)):
    flo2_FJ_ms[s] = flo2_FJ_ms[s][spatial_slice + (slice(None),) + (t_slice,)]
    rel2_FJ_ms[s] = rel2_FJ_ms[s][spatial_slice + (t_slice,)]

# 9. SAVE VISUALIZATIONS
output_dir = os.path.join(OUTPUT_BASEDIR, OUTPUT_NAME)
os.makedirs(output_dir, exist_ok=True)

# Helper for saving videos
save_video(rescale_prctile(B_cropped), os.path.join(output_dir, 'B'))
save_video(vis_edge(estr_cropped, False, ESTR_VIS_PRCTILE_THRESH), os.path.join(output_dir, 'estr_TPC'))
save_video(vis_flo_dense(flo1_cropped, rel_flo1_cropped, MAX_FLO, FLO1_VIS_THICKEN), os.path.join(output_dir, 'flo1_TPC'))

for s, (f_ms, r_ms) in enumerate(zip(flo2_FJ_ms, rel2_FJ_ms)):
    vis_f2 = vis_flo_dense(f_ms, r_ms > FLO2_REL_THRESH, MAX_FLO)
    save_video(vis_f2, os.path.join(output_dir, f'flo2_FJ_ms_{s}'))