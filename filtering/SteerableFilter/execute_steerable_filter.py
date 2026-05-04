import time
import os
import numpy as np
import copy
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, gaussian_filter1d, median_filter, label

# Pipeline-specific imports
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
        # We use the provided sigma and tau for initial smoothing
        I_smoothed = gaussian_filter(I, sigma=[self.sigma, self.sigma, self.tau])
        
        # 1st order derivatives (Odd)
        self.basis['Ix'] = np.gradient(I_smoothed, axis=1)
        self.basis['Iy'] = np.gradient(I_smoothed, axis=0)
        self.basis['It'] = np.gradient(I_smoothed, axis=2)
        
        # 2nd order derivatives (Even)
        self.basis['Ixx'] = np.gradient(self.basis['Ix'], axis=1)
        self.basis['Iyy'] = np.gradient(self.basis['Iy'], axis=0)
        self.basis['Ixy'] = np.gradient(self.basis['Ix'], axis=0)
        
    def get_energy_response(self, theta_deg, velocity=0):
        """
        Computes magnitude of complex quadrature response.
        """
        theta = np.radians(theta_deg)
        c = np.cos(theta)
        s = np.sin(theta)
        
        # Odd response (Hilbert transform)
        R_odd = c * self.basis['Ix'] + s * self.basis['Iy']
        
        # Even response (Zero-crossing detector)
        R_even = (c**2 * self.basis['Ixx'] + 
                  s**2 * self.basis['Iyy'] + 
                  2*c*s * self.basis['Ixy'])
        
        # Energy = Magnitude of the complex pair
        R_total = (R_even + 1j * R_odd) + (velocity * self.basis['It'])
        
        return np.abs(R_total).astype(np.float32)

def nms_3d_edges(energy, orientations, normal_velocities):
    """
    Performs 3D Non-Maximum Suppression.
    1. Spatial NMS: Thins the edge in the X-Y plane.
    2. Temporal NMS: Suppresses tau-induced ghosting along the motion trajectory
                     using a 3x3 max-pool to tolerate SPAD jitter.
    """
    H, W, T = energy.shape
    sharp_energy = np.zeros_like(energy)
    angle = (np.degrees(orientations) % 180)
    
    grad_angle = orientations + (np.pi / 2.0) 
    U = normal_velocities * np.cos(grad_angle)
    V = normal_velocities * np.sin(grad_angle)
    
    for t in range(1, T - 1):
        img = energy[:, :, t]
        ang = angle[:, :, t]
        u_map = U[:, :, t]
        v_map = V[:, :, t]
        out = np.zeros_like(img)
        
        for i in range(1, H-1):
            for j in range(1, W-1):
                if img[i,j] == 0: continue
                
                # --- STEP 1: SPATIAL NMS ---
                q, r = 0, 0
                if (0 <= ang[i,j] < 22.5) or (157.5 <= ang[i,j] <= 180):
                    q, r = img[i+1, j], img[i-1, j]
                elif (22.5 <= ang[i,j] < 67.5):
                    q, r = img[i+1, j-1], img[i-1, j+1]
                elif (67.5 <= ang[i,j] < 112.5):
                    q, r = img[i, j+1], img[i, j-1]
                elif (112.5 <= ang[i,j] < 157.5):
                    q, r = img[i-1, j-1], img[i+1, j+1]

                # Relaxed Spatial NMS: Allows pixels that are at least 60% of the local peak to survive.
                # This naturally widens the edge to 2-3 pixels, making it highly robust against breaking.
                if (img[i,j] >= 0.6 * q) and (img[i,j] >= 0.6 * r):
                    
                    # --- STEP 2: TEMPORAL NMS (Motion-Compensated & Jitter Tolerant) ---
                    # Calculate integer pixel displacement
                    dy = int(round(v_map[i, j])) # Y-axis movement
                    dx = int(round(u_map[i, j])) # X-axis movement
                    
                    # Target coordinates in the previous and next frames
                    prev_i, prev_j = i - dy, j - dx
                    next_i, next_j = i + dy, j + dx
                    
                    # Use a 3x3 max-pool window around the expected trajectory to 
                    # account for SPAD photon flickering and noisy velocity estimations
                    val_prev = 0
                    if 1 <= prev_i < H-1 and 1 <= prev_j < W-1:
                        val_prev = np.max(energy[prev_i-1:prev_i+2, prev_j-1:prev_j+2, t - 1])
                    elif 0 <= prev_i < H and 0 <= prev_j < W:
                        val_prev = energy[prev_i, prev_j, t - 1] # Fallback for borders
                        
                    val_next = 0
                    if 1 <= next_i < H-1 and 1 <= next_j < W-1:
                        val_next = np.max(energy[next_i-1:next_i+2, next_j-1:next_j+2, t + 1])
                    elif 0 <= next_i < H and 0 <= next_j < W:
                        val_next = energy[next_i, next_j, t + 1] # Fallback for borders
                        
                    # Only preserve the pixel if it is the peak across time
                    # 0.75 multiplier tolerates intensity flickering (SPAD scintillation)
                    if (img[i,j] >= 0.75 * val_prev) and (img[i,j] >= 0.75 * val_next):
                        out[i,j] = img[i,j]
                        
        sharp_energy[:, :, t] = out
        
    return sharp_energy

def apply_temporal_continuity(energy, alpha=0.3):
    """
    Applies a forward-backward recursive filter to the energy map.
    """
    H, W, T = energy.shape
    smoothed_energy = np.copy(energy)
    
    for t in range(1, T):
        smoothed_energy[:, :, t] = (1 - alpha) * energy[:, :, t] + alpha * smoothed_energy[:, :, t-1]
        
    for t in range(T-2, -1, -1):
        smoothed_energy[:, :, t] = (1 - alpha) * smoothed_energy[:, :, t] + alpha * smoothed_energy[:, :, t+1]
        
    return smoothed_energy

def directional_edge_link_3d(energy, orientations, link_radius=3):
    """
    Connects short edge chunks by looking along the local edge direction.
    """
    H, W, T = energy.shape
    linked = np.copy(energy)
    angle = (np.degrees(orientations) % 180)
    
    for t in range(T):
        img = energy[:, :, t]
        ang = angle[:, :, t]
        out_layer = np.copy(img)
        
        for i in range(link_radius, H - link_radius):
            for j in range(link_radius, W - link_radius):
                a = ang[i, j]
                offsets = []
                # Look ALONG the edge direction
                if (0 <= a < 22.5) or (157.5 <= a <= 180): # Vertical edge, link along Y
                    offsets = [(d, 0) for d in range(-link_radius, link_radius + 1) if d != 0]
                elif (22.5 <= a < 67.5): # 45 deg edge 
                    offsets = [(-d, d) for d in range(-link_radius, link_radius + 1) if d != 0]
                elif (67.5 <= a < 112.5): # Horizontal edge, link along X
                    offsets = [(0, d) for d in range(-link_radius, link_radius + 1) if d != 0]
                elif (112.5 <= a < 157.5): # 135 deg edge
                    offsets = [(d, d) for d in range(-link_radius, link_radius + 1) if d != 0]
                
                side1 = [img[i + dy, j + dx] for dy, dx in offsets if (dy < 0 or (dy == 0 and dx < 0))]
                side2 = [img[i + dy, j + dx] for dy, dx in offsets if (dy > 0 or (dy == 0 and dx > 0))]
                
                if any(n > 0 for n in side1) and any(n > 0 for n in side2):
                    out_layer[i, j] = max(img[i, j], (max(side1) + max(side2)) / 2.0)
                    
        linked[:, :, t] = out_layer
                        
    return linked
                
def apply_hysteresis_3d(energy, low_threshold, high_threshold, min_vol=80):
    """
    Applies 3D hysteresis thresholding.
    Now supports spatially-varying thresholding (arrays) via numpy broadcasting.
    Also includes a Volumetric Size Filter to destroy floating noise clouds.
    """
    strong_edges = energy > high_threshold
    weak_edges = energy > low_threshold
    
    # Reverted to full 26-connectivity np.ones((3,3,3)). 
    # Moving edges create diagonal steps across the X-Y-T volume. 
    # Strict 6-connectivity breaks moving edges completely.
    labels, num_labels = label(weak_edges, structure=np.ones((3,3,3)))
    
    # 1. Find components that hit the high threshold
    strong_labels = np.unique(labels[strong_edges])
    strong_labels = strong_labels[strong_labels > 0]
    
    # 2. Volumetric Noise Filtering: Kill components smaller than min_vol
    if min_vol > 0 and len(strong_labels) > 0:
        component_sizes = np.bincount(labels.ravel())
        valid_labels = [lbl for lbl in strong_labels if component_sizes[lbl] >= min_vol]
    else:
        valid_labels = strong_labels
    
    final_mask = np.isin(labels, valid_labels)
    
    result = np.zeros_like(energy)
    result[final_mask] = energy[final_mask]
    return result

def execute_steerable_filter(data_file, output_dir, filts_config, params, crop_bounds=None):
    print(f"--- Enhanced Steerable Pipeline (Adaptive Thresholds w/ Glare Penalty) ---")
    
    # 1. Data Loading
    data = loadmat(data_file)
    B_orig = data['B'].astype(np.float32)
    M = data['M'].item()
    if M != 1: B_orig = B_orig * (1.0 / M)

    if crop_bounds:
        imin, imax, jmin, jmax = crop_bounds
        B_orig = B_orig[imin:imax, jmin:jmax, :]

    # 2. Noise Estimation (Keep full 3D Map)
    t_noise = time.time()
    local_flux = ippd(gaussian_filter(B_orig, sigma=2)) 
    flux_noise_std = binom_noise_std(local_flux, M)  # This is a 3D volume
    avg_noise_floor = np.median(flux_noise_std)
    print(f"Noise estimation: {time.time() - t_noise:.3f}s (Median Floor: {avg_noise_floor:.4e})")

    sigmas = filts_config.get('sigma_variations', [0.8]) 
    taus = filts_config.get('tau_variations', [3])
    n_skip = params['N_SKIP_OUTPUT']

    for sigma in sigmas:
        for tau in taus:
            t_var = time.time()
            subdir_name = f"sigma_{sigma:0.1f}_tau_{tau:0.1f}_crisp"
            output_subdir = os.path.join(output_dir, subdir_name)
            os.makedirs(output_subdir, exist_ok=True)
            
            # 3. Steerable Basis Computation
            H, W, N = B_orig.shape
            steerable = SteerableFilterBank3D((H, W, N), sigma=sigma, tau=tau)
            steerable.compute_basis(B_orig)
            
            # 4. Energy-Based Edge Strength 
            angles = filts_config['orientations']
            max_energy = np.zeros_like(B_orig)
            for ang in angles:
                e = steerable.get_energy_response(ang, velocity=0)
                max_energy = np.maximum(max_energy, e)
            
            # 5. Structure Tensor for Orientation
            PC_x2 = steerable.basis['Ix']**2 + steerable.basis['Ixx']**2
            PC_y2 = steerable.basis['Iy']**2 + steerable.basis['Iyy']**2
            PC_t2 = steerable.basis['It']**2
            PC_xy = steerable.basis['Ix'] * steerable.basis['Iy']
            PC_yt = steerable.basis['Iy'] * steerable.basis['It']
            PC_xt = steerable.basis['Ix'] * steerable.basis['It']

            edges, _, _ = features_3D_structure_tensor(PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt)
            
            # 6. Apply strict NMS to find the initial ridge
            sharp_energy = nms_3d_edges(max_energy, edges['orientation'], edges['normal_velocity'])
            
            # 7. Directional Linking to restore connectivity
            sharp_energy = directional_edge_link_3d(sharp_energy, edges['orientation'], link_radius=4)
            
            # 8. Adaptive Hysteresis Denoising with Glare Penalty
            # Bright regions (light sources) cause SPAD pile-up, which artificially lowers 
            # the theoretical binomial noise floor. Meanwhile, 2nd-order derivatives 
            # mathematically explode on high-intensity Poisson noise. 
            # FIX: We apply a heavy "glare penalty" based on the absolute photon flux.
            
            base_noise_floor = np.maximum(flux_noise_std, 1e-6)
            
            glare_penalty = 1.0 + (15.0 * local_flux)
            robust_noise_floor = base_noise_floor * glare_penalty
            
            snr_map = max_energy / robust_noise_floor
            valid_snr = snr_map[snr_map > 0]

            if len(valid_snr) > 0:
                # Calculate percentiles based on the heavily penalized SNR
                # Lowered thresholds slightly because the volumetric filter will now handle the noise
                low_snr_t = np.percentile(valid_snr, 93) 
                high_snr_t = np.percentile(valid_snr, 96) 
            else:
                low_snr_t, high_snr_t = 1.0, 3.0 

            # Create 3D threshold arrays
            low_t_3d = low_snr_t * robust_noise_floor
            high_t_3d = high_snr_t * robust_noise_floor

            # Apply the arrays. 
            # min_vol=80 ensures that tiny flickering noise clouds are destroyed
            sharp_strength = apply_hysteresis_3d(sharp_energy, low_t_3d, high_t_3d, min_vol=90)
            
            # 9. Normalization & Visualization
            global_max = np.max(sharp_strength) + 1e-9
            B_crop = B_orig[:, :, n_skip:-n_skip]
            save_video(np.moveaxis(rescale_prctile(B_crop), -1, 0), os.path.join(output_subdir, 'B.mp4'))
            
            estr_norm = sharp_strength / global_max
            estr_log = np.log1p(estr_norm * 3) / np.log1p(3)
            estr_v = vis_edge(estr_log[:, :, n_skip:-n_skip], False, params['ESTR_VIS_PRCTILE_THRESH'])
            save_video(np.moveaxis(estr_v, -1, 0), os.path.join(output_subdir, 'estr_sharp_crisp.mp4'))
            
            # Flow Visualization
            flo1 = flo_1d_to_uv(edges['normal_velocity'], edges['orientation'])
            rel_flo1 = (sharp_strength > low_t_3d)
            
            flo1_v = vis_flo_dense(flo1[:, :, :, n_skip:-n_skip], rel_flo1[:, :, n_skip:-n_skip], params['MAX_FLO'], params['FLO1_VIS_THICKEN'])
            save_video(np.moveaxis(flo1_v, -1, 0), os.path.join(output_subdir, 'flo1_crisp.mp4'))

            print(f"Finished {subdir_name} in {time.time() - t_var:0.2f}s")

    return {"status": "success", "method": "adaptive_steerable_pipeline"}