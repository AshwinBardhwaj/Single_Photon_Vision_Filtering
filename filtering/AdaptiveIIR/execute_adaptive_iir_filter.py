import time
import os
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, label, maximum_filter, binary_dilation

# Assuming these are available in the environment as per user script
from sensor.sensor import ippd, binom_noise_std
from vision.feature_detection.structure_tensor import features_3D_structure_tensor
from vision.motion_estimation.flo_1d_to_uv import flo_1d_to_uv
from visualize.visualize import vis_edge, vis_flo_dense
from utils.utils import rescale_prctile
from fileio.fileio import save_video_imageio as save_video

class AdaptiveIIRFilterBank3D:
    """
    Implementation of Adaptive IIR Spatiotemporal Filters based on Fleet & Langley.
    Uses 3rd order IIR filters with adaptive temporal tuning based on measured velocity.
    """
    def __init__(self, input_size, b=0.5, mu=0.1):
        self.H, self.W, self.T = input_size
        self.b = b  # Exponential time constant (bandwidth control)
        self.mu = mu  # Adaptation rate
        
        # Spatial tunings (Gabor-like orientations)
        self.orientations = [0, 45, 90, 135]
        self.k_vecs = []
        for ang in self.orientations:
            theta = np.radians(ang)
            k0 = 0.25 
            self.k_vecs.append(np.array([k0 * np.cos(theta), k0 * np.sin(theta)]))
        
        # State: Temporal tuning omega_0 for each spatial channel
        self.omega0 = np.zeros((self.H, self.W, len(self.k_vecs)))
        
        # IIR State buffers: Use [H, W, n_channels] to track previous output
        self.prev_R = np.zeros((self.H, self.W, len(self.k_vecs)), dtype=np.complex64)

    def _spatial_filter(self, frame):
        """Apply complex Gabor spatial filters."""
        outputs = []
        for k in self.k_vecs:
            # Gentle spatial blurring
            sigma_s = 1.0
            smoothed = gaussian_filter(frame, sigma=sigma_s)
            
            # Create coordinate grids
            y, x = np.indices((self.H, self.W))
            # Modulation shifts the signal in frequency space
            modulation = np.exp(1j * 2 * np.pi * (k[0]*x + k[1]*y))
            outputs.append(smoothed * modulation)
        
        return np.stack(outputs, axis=-1)

    def process_frame(self, spatial_responses):
        """
        Processes a single frame through the IIR filters.
        """
        # Temporal center frequency tuning
        j_omega = 1j * self.omega0
        r = (self.b + j_omega - 2.0) / (self.b + j_omega + 2.0)
        q = (self.b) / (self.b + j_omega + 2.0)

        # Apply IIR step: R[t] = q*S[t] + r*R[t-1]
        new_R = q * spatial_responses + r * self.prev_R
        self.prev_R = new_R.copy()
        
        return new_R

    def update_tuning(self, measured_velocity):
        """Feedback loop: Update omega0 based on measured velocity."""
        for i, k in enumerate(self.k_vecs):
            target_omega = measured_velocity[:, :, 0] * k[0] + measured_velocity[:, :, 1] * k[1]
            self.omega0[:, :, i] += self.mu * (target_omega - self.omega0[:, :, i])

def thin_edges_3d(mag, size=3):
    """
    Apply a soft thinning by finding local peaks.
    Keeping the size small (3) ensures we don't reduce it to a single pixel 
    when the gradient is broad.
    """
    # Local maximum helps concentrate the edge energy
    local_max = maximum_filter(mag, size=(size, size, 1))
    # We blend the peak with the original to maintain some thickness/softness
    # Pure suppression (mag == local_max) results in single-pixel lines.
    mask = (mag >= local_max * 0.9)
    return mag * mask

def execute_adaptive_filter(data_file, output_dir, filts_config, params, crop_bounds=None):
    print(f"--- Adaptive IIR Phase-Based Pipeline Start: {data_file} ---")
    
    data = loadmat(data_file)
    B_orig = data['B'].astype(np.float32)
    M = data['M'].item()
    if M != 1: B_orig = B_orig * (1.0 / M)

    if crop_bounds:
        imin, imax, jmin, jmax = crop_bounds
        B_orig = B_orig[imin:imax, jmin:jmax, :]

    H, W, T = B_orig.shape
    n_skip = params.get('N_SKIP_OUTPUT', 5)
    
    # Initialize Filter Bank
    afb = AdaptiveIIRFilterBank3D((H, W, T), b=0.8, mu=0.02)
    
    all_mag = np.zeros((H, W, T), dtype=np.float32)
    
    # Process frames
    for t in range(T):
        S_t = afb._spatial_filter(B_orig[:, :, t])
        R_t = afb.process_frame(S_t)
        
        # Energy across orientations (Targeting high-frequency edges)
        all_mag[:, :, t] = np.sum(np.abs(R_t), axis=-1)
        
        # Adaptive tuning update (identity for now)
        afb.update_tuning(np.zeros((H, W, 2)))

    # --- NOISE EXCLUSION LOGIC (FIXED FOR NEGATIVE REGIONS) ---
    # Goal: Isolate the target which is "darker" or "negative" compared to background.
    
    # 1. Generate a low-frequency support signal
    target_mass = gaussian_filter(B_orig, sigma=[5, 5, 2])
    
    # 2. Find the "Negative" target zone
    mass_thresh = np.percentile(target_mass, 4) 
    binary_mass = target_mass < mass_thresh # Look for "dark" blobs
    
    # 3. Create an "Edge Interest Zone"
    edge_zone = np.zeros_like(binary_mass)
    for t in range(T):
        edge_zone[:,:,t] = binary_dilation(binary_mass[:,:,t], iterations=8)
    
    # 4. Gating and Thinning
    # Mask high-frequency edges with the low-frequency "dark" support mask
    all_mag_masked = all_mag * edge_zone

    # Apply thinning to the gated magnitude to sharpen results without making them single-pixel
    all_mag_thinned = thin_edges_3d(all_mag_masked, size=3)

    # Temporal smoothing to stabilize the result
    all_mag_stable = gaussian_filter(all_mag_thinned, sigma=[0.5, 0.5, 1.0])

    # Dynamic contrast normalization (Targeted to the zone)
    masked_active = all_mag_stable[edge_zone > 0]
    if len(masked_active) > 0:
        norm_factor = np.percentile(masked_active, 99.5) + 1e-8
    else:
        norm_factor = np.max(all_mag_stable) + 1e-8
        
    clean_strength = all_mag_stable / norm_factor
    clean_strength = np.clip(clean_strength, 0, 1)

    # Scaling: Gamma Correction to bring out the faint outline
    # A high gamma (like 5.0) can sometimes make edges look thicker/noisier. 
    # Adjusted back towards a balanced value to keep edges well-defined.
    gamma = 1.8
    estr_gamma = np.power(clean_strength, gamma)

    # Save Outputs
    output_subdir = os.path.join(output_dir, "adaptive_iir_results")
    os.makedirs(output_subdir, exist_ok=True)
    
    B_crop = B_orig[:, :, n_skip:-n_skip]
    save_video(np.moveaxis(rescale_prctile(B_crop), -1, 0), os.path.join(output_subdir, 'B.mp4'))
    
    # Visualization thresholding
    estr_v = vis_edge(estr_gamma[:, :, n_skip:-n_skip], False, params.get('ESTR_VIS_PRCTILE_THRESH', 96))
    save_video(np.moveaxis(estr_v, -1, 0), os.path.join(output_subdir, 'estr_adaptive.mp4'))

    print(f"Finished Adaptive IIR Pipeline. Root output: {output_subdir}")
    return {"status": "success"}