import time
import os
import numpy as np
import copy
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from sensor.sensor import ippd, binom_noise_std
from vision.feature_detection.structure_tensor import features_3D_structure_tensor
from vision.motion_estimation.flo_1d_to_uv import flo_1d_to_uv
from vision.motion_estimation.vel_fj1990_approx_withz import vel_fj1990_approx_withz
from visualize.visualize import vis_edge, vis_flo_dense
from utils.utils import rescale_prctile
from fileio.fileio import save_video_imageio as save_video

class SteerableFilterBank3D:
    def __init__(self, input_size, sigma=1.0, tau=1.0):
        self.input_size = input_size
        self.sigma = sigma
        self.tau = tau
        self.basis = {}
        
    def compute_basis(self, I):
        I_smoothed = gaussian_filter(I, sigma=[self.sigma, self.sigma, self.tau])
        
        self.basis['Ix'] = np.gradient(I_smoothed, axis=1)
        self.basis['Iy'] = np.gradient(I_smoothed, axis=0)
        self.basis['It'] = np.gradient(I_smoothed, axis=2)
        
        self.basis['Ixx'] = np.gradient(self.basis['Ix'], axis=1)
        self.basis['Iyy'] = np.gradient(self.basis['Iy'], axis=0)
        self.basis['Ixy'] = np.gradient(self.basis['Ix'], axis=0)
        
    def get_energy_response(self, theta_deg, velocity):
        theta = np.radians(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        R_odd = c * self.basis['Ix'] + s * self.basis['Iy']
        
        R_even = (c**2 * self.basis['Ixx'] + 
                  s**2 * self.basis['Iyy'] + 
                  2*c*s * self.basis['Ixy'])
        
        R_total = (R_even + 1j * R_odd) + (velocity * self.basis['It'])
        
        return R_total.astype(np.complex64)

def execute_pipeline(data_file, output_dir, filts_config, params, crop_bounds=None):
    print(f"--- Optimized Steerable Pipeline Start: {data_file} ---")
    
    # 1. Data Loading
    data = loadmat(data_file)
    B_orig = data['B'].astype(np.float32)
    M = data['M'].item()
    if M != 1: B_orig = B_orig * (1.0 / M)

    if crop_bounds:
        imin, imax, jmin, jmax = crop_bounds
        B_orig = B_orig[imin:imax, jmin:jmax, :]

    # 2. Noise Estimation
    t_noise = time.time()
    local_flux = ippd(gaussian_filter(B_orig, sigma=5)) # Sharper noise mask
    flux_noise_std = binom_noise_std(local_flux, M)
    print(f"Noise estimation: {time.time() - t_noise:.3f}s")

    # Define Parameter Variations
    sigmas = filts_config.get('sigma_variations', [0.8, 1.2]) # Lower sigma for sharpness
    taus = filts_config.get('tau_variations', [1.0, 2.0])
    
    orientations = filts_config.get('orientations', [0, 45, 90, 135])
    velocities = filts_config.get('velocities', [0, 0.5, -0.5])
    n_skip = params['N_SKIP_OUTPUT']

    for sigma in sigmas:
        for tau in taus:
            t_var = time.time()
            subdir_name = f"sigma_{sigma:0.1f}_tau_{tau:0.1f}_sharp"
            output_subdir = os.path.join(output_dir, subdir_name)
            os.makedirs(output_subdir, exist_ok=True)
            
            print(f"\nProcessing variation: {subdir_name}")
            
            # 3. Steerable Basis Computation
            H, W, N = B_orig.shape
            steerable = SteerableFilterBank3D((H, W, N), sigma=sigma, tau=tau)
            steerable.compute_basis(B_orig)
            
            # 4. Feature Detection via Structure Tensor
            # We use the squared gradients of the basis directly for the tensor
            # to maintain the highest spatial frequency possible.
            PC_x2 = steerable.basis['Ix']**2 + steerable.basis['Ixx']**2
            PC_y2 = steerable.basis['Iy']**2 + steerable.basis['Iyy']**2
            PC_t2 = steerable.basis['It']**2
            PC_xy = steerable.basis['Ix'] * steerable.basis['Iy']
            PC_yt = steerable.basis['Iy'] * steerable.basis['It']
            PC_xt = steerable.basis['Ix'] * steerable.basis['It']

            edges, _, _ = features_3D_structure_tensor(PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt)
            
            # 5. Motion Estimation & Visualization
            flo1 = flo_1d_to_uv(edges['normal_velocity'], edges['orientation'])
            # Dynamic thresholding to pick up faint distant lines
            strength_thresh = np.percentile(edges['strength'], 60) 
            rel_flo1 = (edges['strength'] > strength_thresh)
            
            B_crop = B_orig[:, :, n_skip:-n_skip]
            save_video(np.moveaxis(rescale_prctile(B_crop), -1, 0), os.path.join(output_subdir, 'B.mp4'))
            
            # Save edge strength map (normalized for visibility of faint lines)
            estr_norm = edges['strength'] / (np.max(edges['strength']) + 1e-6)
            estr_v = vis_edge(estr_norm[:, :, n_skip:-n_skip], False, params['ESTR_VIS_PRCTILE_THRESH'])
            save_video(np.moveaxis(estr_v, -1, 0), os.path.join(output_subdir, 'estr_sharp.mp4'))
            
            # Save 1D flow
            flo1_v = vis_flo_dense(flo1[:, :, :, n_skip:-n_skip], rel_flo1[:, :, n_skip:-n_skip], params['MAX_FLO'], params['FLO1_VIS_THICKEN'])
            save_video(np.moveaxis(flo1_v, -1, 0), os.path.join(output_subdir, 'flo1_sharp.mp4'))

            print(f"Finished {subdir_name} in {time.time() - t_var:0.2f}s")

    print(f"\nAll variations complete. Root output: {output_dir}")
    return {"status": "success", "method": "steerable_sharp_quadrature"}