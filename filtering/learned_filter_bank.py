import torch
import torch.nn as nn
import torch.fft


class LearnedLogGaborBank3D(nn.Module):
    def __init__(self, input_size, num_scales=3, num_orientations=3, num_velocities=3):
        super().__init__()
        self.input_size = input_size
        self.num_scales = num_scales
        self.num_orientations = num_orientations
        self.num_velocities = num_velocities

        self.min_wavelength = nn.Parameter(torch.tensor(4.0))
        self.mult = nn.Parameter(torch.tensor(2.1))
        self.sigma_on_f = nn.Parameter(torch.tensor(0.55))
        self.dtheta_max = nn.Parameter(torch.tensor(60.0))

        self.orientations = nn.Parameter(torch.linspace(0, 120, num_orientations))
        self.velocities = nn.Parameter(torch.linspace(-1, 1, num_velocities))

    def _get_freq_grids(self, device):
        rows, cols, frames = self.input_size

        fx = torch.fft.fftfreq(cols, device=device).view(1, cols, 1)
        fy = torch.fft.fftfreq(rows, device=device).view(rows, 1, 1)
        ft = torch.fft.fftfreq(frames, device=device).view(1, 1, frames)

        f_mag_sp = torch.hypot(fx, fy)
        f_theta = torch.rad2deg(torch.atan2(fy, fx))

        return fx, fy, ft, f_mag_sp, f_theta

    def forward(self, x, flux_noise_std=None):
        device = x.device
        fx, fy, ft, f_mag_sp, f_theta = self._get_freq_grids(device)

        # Strictly clamp physical parameters so they cannot diverge and cause log(0)
        safe_sigma_on_f = torch.clamp(self.sigma_on_f, 0.1, 0.9)
        safe_min_wavelength = torch.clamp(self.min_wavelength, 2.0, 20.0)
        safe_mult = torch.clamp(self.mult, 1.1, 5.0)
        safe_dtheta_max = torch.clamp(self.dtheta_max, 10.0, 180.0)

        lp_cutoff, lp_n = 0.45, 15
        lp = 1.0 / (1.0 + (f_mag_sp / lp_cutoff) ** (2 * lp_n))

        log_sigma = torch.log(safe_sigma_on_f)

        x_fft = torch.fft.fftn(x, dim=(0, 1, 2))

        responses = []
        filter_energies = []
        dirs = []

        for i1 in range(self.num_scales, 0, -1):
            f0 = (1.0 / safe_min_wavelength) / (safe_mult ** (self.num_scales - i1))

            lg_r = torch.exp((-(torch.log(f_mag_sp / f0 + 1e-8)) ** 2) / (2 * log_sigma ** 2 + 1e-8))
            lg_r = lg_r * lp
            lg_r[0, 0, 0] = 0.0

            for i2 in range(self.num_orientations):
                angl = self.orientations[i2]
                rad_angl = torch.deg2rad(angl)

                sinftheta = torch.sin(torch.deg2rad(f_theta))
                cosftheta = torch.cos(torch.deg2rad(f_theta))

                ds_s = sinftheta * torch.cos(rad_angl) - cosftheta * torch.sin(rad_angl)
                dc_s = cosftheta * torch.cos(rad_angl) + sinftheta * torch.sin(rad_angl)
                dftheta = torch.abs(torch.rad2deg(torch.atan2(ds_s, dc_s)))

                dftheta = 180.0 * torch.clamp(dftheta / safe_dtheta_max, max=1.0)
                spread = (torch.cos(torch.deg2rad(dftheta)) + 1.0) / 2.0

                Gs = lg_r * spread

                for i3 in range(self.num_velocities):
                    v = self.velocities[i3]
                    ft_c = -f0 * v

                    safe_ft_c = torch.where(torch.abs(ft_c) < 1e-6, torch.sign(ft_c + 1e-9) * 1e-6, ft_c)
                    ft_scaled = ft / safe_ft_c

                    lg_ft_nz = torch.exp(-(torch.log(torch.relu(ft_scaled) + 1e-8) ** 2) / (2 * log_sigma ** 2 + 1e-8))
                    lg_ft_nz = torch.where(ft_scaled > 0, lg_ft_nz, torch.zeros_like(lg_ft_nz))

                    sigmaf = 0.1 / 2.0
                    lg_ft_zero = torch.exp(-0.5 * (ft / sigmaf) ** 2)

                    is_zero_vel = torch.exp(- (v / 1e-3) ** 2)
                    lg_ft = is_zero_vel * lg_ft_zero + (1.0 - is_zero_vel) * lg_ft_nz

                    G = Gs * lg_ft

                    # Safe energy calculation (avoid torch.abs on complex)
                    g_spatial = torch.fft.ifftn(G)
                    energy = torch.sum(g_spatial.real ** 2 + g_spatial.imag ** 2)
                    filter_energies.append(energy)

                    R_f = x_fft * G
                    R = torch.fft.ifftn(R_f)
                    responses.append(R)

                    dirs.append(torch.stack([torch.cos(rad_angl), torch.sin(rad_angl), v]))

        return responses, torch.stack(filter_energies), torch.stack(dirs)

import os
from pathlib import Path
import numpy as np

class LearnedFilterBank3DSepT:
    def __init__(self):
        # Default Properties to match LogGaborBank3DSepT
        self.input_size = [256, 512, 50]
        self.num_scales = 3
        
        # We enforce matching the user's evaluation script config exactly
        self.orientations = np.array([0, 60, 120])
        self.velocities = np.array([1, -1, 0.3, -0.3, 0])
        self.num_orientations = len(self.orientations)
        self.num_velocities = len(self.velocities)
        
        self.spacing_spatial = 1
        self.spacing_temporal = 1

        self.model = None
        self.device = torch.device('cpu') # Eval happens on CPU for memory reasons by default, or MPS/CUDA if available
        self._dirs_cache = None

    def set_up_filters(self):
        self.num_orientations = len(self.orientations)
        self.num_velocities = len(self.velocities)
        
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            
        self.model = LearnedLogGaborBank3D(
            input_size=self.input_size,
            num_scales=self.num_scales,
            num_orientations=self.num_orientations,
            num_velocities=self.num_velocities
        ).to(self.device)
        
        # Try to load best_model.pt if it exists
        root_dir = Path(__file__).resolve().parent.parent
        weights_path = root_dir / "scripts" / "output" / "weights" / "best_model.pt"
        if weights_path.exists():
            print(f"Loading trained weights from {weights_path}")
            self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        else:
            print("No trained weights found. Using initial parameters.")
            
        self.model.eval()
        return self

    def ind_tuning_to_ind_filt(self, ind_v, ind_o):
        return (ind_v - 1) * self.num_orientations + (ind_o - 1)

    def ind_filt_to_ind_tuning(self, ind_filt):
        ind_v = 1 + ((ind_filt - 1) // self.num_orientations)
        ind_o = 1 + ((ind_filt - 1) % self.num_orientations)
        return ind_v, ind_o

    def response(self, I, flux_noise_std=None):
        import gc
        R = {}
        Rz = {}
        
        # We need to run the model forward pass
        I_tensor = torch.tensor(I.astype(np.float32), device=self.device)
        
        with torch.no_grad():
            self.model.input_size = I.shape
            responses, energies, dirs = self.model(I_tensor)
            
            # The energies output by LearnedLogGaborBank3D are in the inner loop (1D list)
            # length = num_scales * num_orientations * num_velocities
            # We must map them back to numpy dicts
            
            num_dirs = self.num_orientations * self.num_velocities
            self.filter_energies = np.zeros((num_dirs, self.num_scales), dtype=np.float32)
            
            # The loop in forward is scale (reverse) -> orientation -> velocity
            # Wait, scale loop goes from num_scales down to 1.
            # So responses[0] is scale=num_scales-1, responses[-1] is scale=0
            
            idx = 0
            for i1 in range(self.num_scales, 0, -1):
                ind_s = i1 - 1
                for i2 in range(self.num_orientations):
                    ind_o = i2 + 1
                    for i3 in range(self.num_velocities):
                        ind_v = i3 + 1
                        
                        ind_filt = self.ind_tuning_to_ind_filt(ind_v, ind_o)
                        
                        res = responses[idx].cpu().numpy().astype(np.complex64)
                        en = energies[idx].item()
                        
                        self.filter_energies[ind_filt, ind_s] = en
                        R[(ind_filt, ind_s)] = res
                        
                        # Rz approximation
                        if flux_noise_std is not None:
                            from filtering.stdev_response import stdev_response
                            from scipy.ndimage import zoom
                            
                            if flux_noise_std.shape != res.shape:
                                zoom_factors = [r / n for r, n in zip(res.shape, flux_noise_std.shape)]
                                noise_std_os = zoom(flux_noise_std, zoom_factors, order=0)
                            else:
                                noise_std_os = flux_noise_std
                                
                            std_R = stdev_response(noise_std_os, en)
                            Rz[(ind_filt, ind_s)] = (np.abs(res) / std_R).astype(np.float32)
                        else:
                            Rz[(ind_filt, ind_s)] = np.full(res.shape, np.inf, dtype=np.float32)
                            
                        idx += 1
                        
            # Cache the directions from the model
            self._dirs_cache = dirs[:num_dirs].cpu().numpy()
            
        del I_tensor
        gc.collect()
        
        return R, Rz

    def tuning_directions(self):
        # Learned tuning directions might have changed from initialization
        # If response() hasn't been called, return initial
        if self._dirs_cache is not None:
            return self._dirs_cache
            
        dirs = np.zeros((self.num_orientations * self.num_velocities, 3), dtype=np.float32)
        with torch.no_grad():
            ori_t = self.model.orientations.cpu().numpy()
            vel_t = self.model.velocities.cpu().numpy()
            
            for ind_filt in range(self.num_orientations * self.num_velocities):
                ind_v, ind_o = self.ind_filt_to_ind_tuning(ind_filt + 1)
                
                theta = np.radians(ori_t[ind_o - 1])
                v = vel_t[ind_v - 1]
                
                dirs[ind_filt, 0] = np.cos(theta)
                dirs[ind_filt, 1] = np.sin(theta)
                dirs[ind_filt, 2] = v
                
        return dirs