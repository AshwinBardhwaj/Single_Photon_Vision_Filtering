import time
import os
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, label, maximum_filter

# Pipeline-specific imports
from sensor.sensor import ippd, binom_noise_std
from vision.feature_detection.structure_tensor import features_3D_structure_tensor
from vision.motion_estimation.flo_1d_to_uv import flo_1d_to_uv
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
        """Vectorized computation of 3D steerable basis gradients."""
        I_smoothed = gaussian_filter(I, sigma=[self.sigma, self.sigma, self.tau])
        
        self.basis['Ix'] = np.gradient(I_smoothed, axis=1)
        self.basis['Iy'] = np.gradient(I_smoothed, axis=0)
        self.basis['It'] = np.gradient(I_smoothed, axis=2)
        
        self.basis['Ixx'] = np.gradient(self.basis['Ix'], axis=1)
        self.basis['Iyy'] = np.gradient(self.basis['Iy'], axis=0)
        self.basis['Ixy'] = np.gradient(self.basis['Ix'], axis=0)
        
    def get_energy_response(self, theta_deg, velocity=0):
        """Computes quadrature energy for a specific orientation."""
        theta = np.radians(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        R_odd = c * self.basis['Ix'] + s * self.basis['Iy']
        R_even = (c**2 * self.basis['Ixx'] + 
                  s**2 * self.basis['Iyy'] + 
                  2*c*s * self.basis['Ixy'])
        
        R_total = (R_even + 1j * R_odd) + (velocity * self.basis['It'])
        return np.abs(R_total).astype(np.float32)

def vectorized_nms_2d_slices(energy):
    """Spatially thins edges without suppressing temporal continuity."""
    # Find local maxima in a 3x3 spatial window
    local_max = maximum_filter(energy, size=(3, 3, 1))
    # Relaxed NMS: allow pixels within 70% of the local peak to survive
    # This prevents the "blank output" caused by strict maximum matching on noisy data
    return np.where(energy >= 0.7 * local_max, energy, 0)

def apply_hysteresis_3d(energy, low_threshold, high_threshold, min_vol=10):
    """Fast 3D hysteresis with volumetric filtering."""
    # Use array-based thresholding
    strong_edges = energy > high_threshold
    weak_edges = energy > low_threshold
    
    # Label connected components in 3D
    labels, num_labels = label(weak_edges, structure=np.ones((3,3,3)))
    
    if num_labels == 0:
        return np.zeros_like(energy)

    # Sum of strong pixels per label
    sums = np.bincount(labels.ravel(), weights=strong_edges.ravel())
    strong_label_mask = sums > 0
    
    # Component size filtering
    sizes = np.bincount(labels.ravel())
    size_mask = sizes >= min_vol
    
    valid_label_mask = strong_label_mask & size_mask
    valid_label_indices = np.where(valid_label_mask)[0]
    
    final_mask = np.isin(labels, valid_label_indices)
    
    result = np.zeros_like(energy)
    result[final_mask] = energy[final_mask]
    return result

def execute_steerable_filter(data_file, output_dir, filts_config, params, crop_bounds=None):
    print(f"--- Robust Optimized Steerable Pipeline ---")
    
    data = loadmat(data_file)
    B_orig = data['B'].astype(np.float32)
    M = data['M'].item()
    if M != 1: B_orig = B_orig * (1.0 / M)

    if crop_bounds:
        imin, imax, jmin, jmax = crop_bounds
        B_orig = B_orig[imin:imax, jmin:jmax, :]

    # Noise Estimation
    t_noise = time.time()
    local_flux = ippd(gaussian_filter(B_orig, sigma=2)) 
    flux_noise_std = binom_noise_std(local_flux, M) 
    print(f"Noise estimation: {time.time() - t_noise:.3f}s")

    sigmas = filts_config.get('sigma_variations', [0.8]) 
    taus = filts_config.get('tau_variations', [2])
    n_skip = params['N_SKIP_OUTPUT']

    for sigma in sigmas:
        for tau in taus:
            t_var = time.time()
            subdir_name = f"sigma_{sigma:0.1f}_tau_{tau:0.1f}_fast"
            output_subdir = os.path.join(output_dir, subdir_name)
            os.makedirs(output_subdir, exist_ok=True)
            
            # Basis Computation
            H, W, N = B_orig.shape
            steerable = SteerableFilterBank3D((H, W, N), sigma=sigma, tau=tau)
            steerable.compute_basis(B_orig)
            
            angles = filts_config['orientations']
            max_energy = np.zeros_like(B_orig)
            for ang in angles:
                max_energy = np.maximum(max_energy, steerable.get_energy_response(ang))
            
            # Structure Tensor
            PC_x2 = steerable.basis['Ix']**2 + steerable.basis['Ixx']**2
            PC_y2 = steerable.basis['Iy']**2 + steerable.basis['Iyy']**2
            PC_t2 = steerable.basis['It']**2
            PC_xy = steerable.basis['Ix'] * steerable.basis['Iy']
            PC_yt = steerable.basis['Iy'] * steerable.basis['It']
            PC_xt = steerable.basis['Ix'] * steerable.basis['It']
            edges, _, _ = features_3D_structure_tensor(PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt)
            
            # Spatial NMS
            sharp_energy = vectorized_nms_2d_slices(max_energy)
            
            # Softened glare penalty to avoid killing signal
            glare_penalty = 1.0 + (2.0 * local_flux) 
            robust_noise_floor = np.maximum(flux_noise_std, 1e-6) * glare_penalty
            snr_map = max_energy / robust_noise_floor
            
            # Using more inclusive percentiles (50th and 80th)
            # We filter out energy=0 to get valid statistics
            valid_snr = snr_map[max_energy > 0]
            if valid_snr.size > 0:
                low_snr_t = np.percentile(valid_snr, 50)
                high_snr_t = np.percentile(valid_snr, 80)
            else:
                low_snr_t, high_snr_t = 0.5, 1.0

            low_t_3d = low_snr_t * robust_noise_floor
            high_t_3d = high_snr_t * robust_noise_floor

            # Optimized Hysteresis with very low min_vol (10) for debugging
            sharp_strength = apply_hysteresis_3d(sharp_energy, low_t_3d, high_t_3d, min_vol=10)
            
            global_max = np.max(sharp_strength) + 1e-9
            B_crop = B_orig[:, :, n_skip:-n_skip]
            save_video(np.moveaxis(rescale_prctile(B_crop), -1, 0), os.path.join(output_subdir, 'B.mp4'))
            
            estr_norm = sharp_strength / global_max
            estr_log = np.log1p(estr_norm * 3) / np.log1p(3)
            estr_v = vis_edge(estr_log[:, :, n_skip:-n_skip], False, params['ESTR_VIS_PRCTILE_THRESH'])
            save_video(np.moveaxis(estr_v, -1, 0), os.path.join(output_subdir, 'estr_sharp_fast.mp4'))
            
            # Flow Visualization
            flo1 = flo_1d_to_uv(edges['normal_velocity'], edges['orientation'])
            rel_flo1 = (sharp_strength > 0)
            flo1_v = vis_flo_dense(flo1[:, :, :, n_skip:-n_skip], rel_flo1[:, :, n_skip:-n_skip], params['MAX_FLO'], params['FLO1_VIS_THICKEN'])
            save_video(np.moveaxis(flo1_v, -1, 0), os.path.join(output_subdir, 'flo1_fast.mp4'))

            print(f"Finished {subdir_name} in {time.time() - t_var:0.2f}s")

    return {"status": "success", "method": "optimized_steerable_pipeline"}