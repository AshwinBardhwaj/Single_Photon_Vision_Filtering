import numpy as np
import scipy.io as sio
import os
import sys
import subprocess
from scipy.fft import fft, ifft, fft2, ifft2, ifftn, fftshift

# 1. HANDLE PATHS
# Add the parent directory to sys.path so we can import the Python class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Assuming your converted class is in a file named log_gabor.py in the parent dir
# from log_gabor import LogGaborBank3DSepT 

# --- HELPER FUNCTIONS FOR PYTHON VERSION ---
def fft_freqs(n):
    """Replicates MATLAB fft_freqs (standard FFT frequency ordering)"""
    return np.fft.fftfreq(n)

def stdev_response(flux_noise_std, filter_energy):
    eps_single = np.finfo(np.float32).eps
    return np.maximum(eps_single, flux_noise_std * np.sqrt(0.5 * filter_energy))

def phid_vtuned(v):
    v = np.atleast_1d(v)
    phid = np.degrees(np.arctan2(1, v))
    phid = np.where(phid < 0, phid + 180, phid)
    return phid

# --- INSERT THE CLASS DEFINITION HERE ---
class LogGaborBank3DSepT:
    # ... (Paste the class code from the previous response here) ...
    def __init__(self):
        self.input_size = [256, 512, 50]
        self.num_scales = 3
        self.orientations = np.array([0, 60, 120])
        self.num_orientations = len(self.orientations)
        self.velocities = np.array([1, -1, 1/3, -1/3, 0])
        self.num_velocities = len(self.velocities)
        self.min_wavelength = 4
        self.mult = 2.1
        self.sigma_on_f = 0.55
        self.dtheta_max = 60
        self.spacing_spatial = 1
        self.spacing_temporal = 1
        self.filters_spatial = {}
        self.filters_temporal = {}
        self.filter_energies = None

    def set_up_filters(self):
        diff_ori = self.orientations[1] - self.orientations[0] if len(self.orientations) > 1 else 60
        self.dtheta_max = min(self.dtheta_max, diff_ori)
        rows, cols, frames = self.input_size
        
        # Note: using np.fft.fftfreq and reshapes to match your MATLAB logic
        fx = np.fft.fftfreq(cols).reshape(1, cols, 1).astype(np.float32)
        fy = np.fft.fftfreq(rows).reshape(rows, 1, 1).astype(np.float32)
        ft = np.fft.fftfreq(frames).reshape(1, 1, frames).astype(np.float32)
        
        # Apply ifftshift to match MATLAB ifftshift(fft_freqs)
        fx = np.fft.ifftshift(fx)
        fy = np.fft.ifftshift(fy)
        ft = np.fft.ifftshift(ft)

        f_mag_sp = np.hypot(fx, fy)
        f_theta = np.degrees(np.arctan2(fy, fx))
        sinftheta = np.sin(np.radians(f_theta))
        cosftheta = np.cos(np.radians(f_theta))
        
        lp_cutoff, lp_n = 0.45, 15
        lp = 1.0 / (1.0 + (f_mag_sp / lp_cutoff)**(2 * lp_n))
        
        for i1 in range(self.num_scales, 0, -1):
            ind_s = i1 - 1
            f0 = (1.0 / self.min_wavelength) / (self.mult**(self.num_scales - i1))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                lg_r = np.exp((-(np.log(f_mag_sp / f0))**2) / (2 * np.log(self.sigma_on_f)**2))
            lg_r *= lp
            lg_r[0, 0, 0] = 0
            
            for i2 in range(self.num_orientations):
                angl = self.orientations[i2]
                ds_s = sinftheta * np.cos(np.radians(angl)) - cosftheta * np.sin(np.radians(angl))
                dc_s = cosftheta * np.cos(np.radians(angl)) + sinftheta * np.sin(np.radians(angl))
                dftheta = np.abs(np.degrees(np.arctan2(ds_s, dc_s)))
                dftheta = 180 * np.minimum(1, dftheta / self.dtheta_max)
                spread = (np.cos(np.radians(dftheta)) + 1) / 2
                self.filters_spatial[(i2, ind_s)] = lg_r * spread
                
            ft_c = -f0 * self.velocities
            for i2 in range(self.num_velocities):
                if ft_c[i2] == 0:
                    ft_c_mag = np.abs(ft_c)
                    f1 = np.min(ft_c_mag[ft_c_mag > 0])
                    sigmaf = f1 / 2
                    self.filters_temporal[(i2, ind_s)] = np.exp(-0.5 * (ft / sigmaf)**2)
                elif abs(ft_c[i2]) > (1/3):
                    self.filters_temporal[(i2, ind_s)] = None
                else:
                    lg_ft = np.zeros_like(ft)
                    ft_scaled = ft / ft_c[i2]
                    mask = ft_scaled > 0
                    lg_ft[mask] = np.exp(-(np.log(ft_scaled[mask])**2) / (2 * np.log(self.sigma_on_f)**2))
                    self.filters_temporal[(i2, ind_s)] = lg_ft
        self.set_up_filter_energies()

    def set_up_filter_energies(self):
        self.filter_energies = np.zeros((self.num_velocities * self.num_orientations, self.num_scales))
        for ind_s in range(self.num_scales):
            for ind_filt in range(self.num_orientations * self.num_velocities):
                ind_v, ind_o = self.ind_filt_to_ind_tuning(ind_filt + 1)
                Gs = self.filters_spatial[(ind_o - 1, ind_s)]
                Gt = self.filters_temporal[(ind_v - 1, ind_s)]
                if Gt is not None:
                    g = fftshift(ifftn(Gs * Gt))
                    self.filter_energies[ind_filt, ind_s] = np.sum(np.abs(g)**2)

    def ind_tuning_to_ind_filt(self, ind_v, ind_o):
        return (ind_v - 1) * self.num_orientations + (ind_o - 1)

    def ind_filt_to_ind_tuning(self, ind_filt):
        ind_v = 1 + ((ind_filt - 1) // self.num_orientations)
        ind_o = 1 + ((ind_filt - 1) % self.num_orientations)
        return ind_v, ind_o

    def response(self, I):
        R = {}
        I_F_sp = fft2(I, axes=(0, 1))
        for ind_s in range(self.num_scales):
            for ind_o in range(self.num_orientations):
                Gs = self.filters_spatial[(ind_o, ind_s)]
                R_sp = ifft2(I_F_sp * Gs, axes=(0, 1))
                R_F_t = fft(R_sp, axis=2)
                for ind_v in range(self.num_velocities):
                    Gt = self.filters_temporal[(ind_v, ind_s)]
                    if Gt is not None:
                        ind_filt = self.ind_tuning_to_ind_filt(ind_v + 1, ind_o + 1)
                        res = ifft(R_F_t * Gt, axis=2)
                        R[(ind_filt, ind_s)] = res
        return R

def generate_matlab_validation_script(input_filename, output_filename):
    """Creates a temporary MATLAB script. Adds current dir to path."""
    matlab_code = f"""
    addpath(pwd); % Ensure MATLAB finds classes in current directory
    load('{input_filename}', 'test_input');
    
    obj = logGabor_bank_3D_sep_t();
    obj.input_size = size(test_input);
    obj = obj.set_up_filters();
    
    [R_mat, ~] = obj.response(test_input);
    filter_energies = obj.filter_energies;
    
    save('{output_filename}', 'R_mat', 'filter_energies');
    exit;
    """
    with open("run_matlab_validation.m", "w") as f:
        f.write(matlab_code)

def run_numerical_comparison():
    size_dims = (32, 32, 20)
    test_input = np.random.rand(*size_dims).astype(np.float32)
    sio.savemat('test_input.mat', {'test_input': test_input})
    
    print("--- Running Python Implementation ---")
    bank = LogGaborBank3DSepT()
    bank.input_size = list(size_dims)
    bank.set_up_filters()
    R_py = bank.response(test_input)
    
    print("--- Running MATLAB Implementation ---")
    generate_matlab_validation_script('test_input.mat', 'test_output_matlab.mat')
    
    try:
        subprocess.run(["matlab", "-batch", "run_matlab_validation"], check=True)
    except FileNotFoundError:
        print("Error: 'matlab' not found. Run 'run_matlab_validation.m' manually.")
        return

    if os.path.exists('test_output_matlab.mat'):
        mat_data = sio.loadmat('test_output_matlab.mat')
        R_mat = mat_data['R_mat']
        E_mat = mat_data['filter_energies']
        
        print("\n=== COMPARISON RESULTS ===")
        # Check energy
        energy_diff = np.linalg.norm(bank.filter_energies - E_mat)
        print(f"Filter Energy Difference: {energy_diff:.2e}")
        
        # Check Response (Filt 0, Scale 0)
        # Note: MATLAB cell (1,1) is R_mat[0,0] in Python loadmat
        py_res = R_py[(0, 0)]
        mat_res = R_mat[0, 0]
        
        rel_error = np.linalg.norm(py_res - mat_res) / np.linalg.norm(mat_res)
        print(f"Response Rel. Error (Scale 0, Filt 0): {rel_error:.2e}")

if __name__ == "__main__":
    run_numerical_comparison()