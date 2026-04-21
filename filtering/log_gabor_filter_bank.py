import numpy as np
import gc
from scipy.fft import fft, ifft, fft2, ifft2, ifftn, fftshift
from scipy.ndimage import zoom
from filtering.stdev_response import stdev_response

class LogGaborBank3DSepT:
    def __init__(self):
        # Default Properties
        self.input_size = [256, 512, 50]
        self.num_scales = 3
        self.orientations = np.array([0, 60, 120])
        self.num_orientations = len(self.orientations)
        self.velocities = np.array([1, -1, 1 / 3, -1 / 3, 0])
        self.num_velocities = len(self.velocities)
        self.min_wavelength = 4
        self.mult = 2.1
        self.sigma_on_f = 0.55
        self.dtheta_max = 60
        self.spacing_spatial = 1
        self.spacing_temporal = 1

        # Storage for filters
        self.filters_spatial = {}  # Keys: (ind_o, ind_s)
        self.filters_temporal = {}  # Keys: (ind_v, ind_s)
        self.filter_energies = None

    def set_up_filters(self):
        # Update derived properties
        diff_ori = self.orientations[1] - self.orientations[0] if len(self.orientations) > 1 else 60
        self.dtheta_max = min(self.dtheta_max, diff_ori)
        self.num_orientations = len(self.orientations)
        self.num_velocities = len(self.velocities)

        rows, cols, frames = self.input_size

        # Frequency grids (Equivalent to MATLAB's fft_freqs logic)
        def get_freqs(n):
            return np.fft.fftfreq(n)

        fx = get_freqs(cols).reshape(1, cols, 1).astype(np.float32)
        fy = get_freqs(rows).reshape(rows, 1, 1).astype(np.float32)
        ft = get_freqs(frames).reshape(1, 1, frames).astype(np.float32)

        f_mag_sp = np.hypot(fx, fy)
        f_theta = np.degrees(np.arctan2(fy, fx))
        sinftheta = np.sin(np.radians(f_theta))
        cosftheta = np.cos(np.radians(f_theta))

        # Low-pass Butterworth
        lp_cutoff, lp_n = 0.45, 15
        lp = 1.0 / (1.0 + (f_mag_sp / lp_cutoff) ** (2 * lp_n))

        log_sigma = np.log(self.sigma_on_f)

        for i1 in range(self.num_scales, 0, -1):
            ind_s = i1 - 1
            f0 = (1.0 / self.min_wavelength) / (self.mult ** (self.num_scales - i1))

            # 1. Spatial Filters
            with np.errstate(divide='ignore'):
                lg_r = np.exp((-(np.log(f_mag_sp / f0)) ** 2) / (2 * log_sigma ** 2))
            lg_r *= lp
            lg_r[0, 0, 0] = 0  # DC component

            for i2 in range(self.num_orientations):
                angl = self.orientations[i2]
                rad_angl = np.radians(angl)
                ds_s = sinftheta * np.cos(rad_angl) - cosftheta * np.sin(rad_angl)
                dc_s = cosftheta * np.cos(rad_angl) + sinftheta * np.sin(rad_angl)
                dftheta = np.abs(np.degrees(np.arctan2(ds_s, dc_s)))

                dftheta = 180 * np.minimum(1, dftheta / self.dtheta_max)
                spread = (np.cos(np.radians(dftheta)) + 1) / 2
                self.filters_spatial[(i2, ind_s)] = lg_r * spread

            # 2. Temporal Filters
            ft_c = -f0 * self.velocities
            for i2 in range(self.num_velocities):
                if ft_c[i2] == 0:
                    # Lowpass for zero velocity
                    ft_c_mag = np.abs(ft_c)
                    f1 = np.min(ft_c_mag[ft_c_mag > 0]) if any(ft_c_mag > 0) else 0.1
                    sigmaf = f1 / 2
                    self.filters_temporal[(i2, ind_s)] = np.exp(-0.5 * (ft / sigmaf) ** 2)
                elif abs(ft_c[i2]) > (1 / 3):
                    self.filters_temporal[(i2, ind_s)] = None
                else:
                    lg_ft = np.zeros_like(ft)
                    ft_scaled = ft / ft_c[i2]
                    mask = ft_scaled > 0
                    lg_ft[mask] = np.exp(-(np.log(ft_scaled[mask]) ** 2) / (2 * log_sigma ** 2))
                    self.filters_temporal[(i2, ind_s)] = lg_ft

        self.set_up_filter_energies()
        return self

    def set_up_filter_energies(self):
        self.filter_energies = np.zeros((self.num_velocities * self.num_orientations, self.num_scales))
        for ind_s in range(self.num_scales):
            for ind_filt in range(self.num_orientations * self.num_velocities):
                ind_v, ind_o = self.ind_filt_to_ind_tuning(ind_filt + 1)

                # Cast to float32 to halve memory during the massive 3D broadcast
                Gs = self.filters_spatial[(ind_o - 1, ind_s)].astype(np.float32)
                Gt = self.filters_temporal[(ind_v - 1, ind_s)].astype(np.float32)

                if Gt is not None:
                    # Combined spatio-temporal filter energy
                    # SciPy will keep this as complex64 because inputs are float32
                    g = fftshift(ifftn(Gs * Gt))
                    self.filter_energies[ind_filt, ind_s] = np.sum(np.abs(g) ** 2)

                    # Delete massive temporary variables immediately
                    del g, Gs, Gt
                    gc.collect()

    def ind_tuning_to_ind_filt(self, ind_v, ind_o):
        return (ind_v - 1) * self.num_orientations + (ind_o - 1)

    def ind_filt_to_ind_tuning(self, ind_filt):
        # Returns 1-based indexing to match your MATLAB logic if needed
        ind_v = 1 + ((ind_filt - 1) // self.num_orientations)
        ind_o = 1 + ((ind_filt - 1) % self.num_orientations)
        return ind_v, ind_o

    def response(self, I, flux_noise_std=None):
        R = {}
        Rz = {}

        # 1. Spatial FFT of the input video
        I_F_sp = fft2(I.astype(np.float32), axes=(0, 1))

        for ind_s in range(self.num_scales):
            for ind_o in range(self.num_orientations):
                Gs = self.filters_spatial[(ind_o, ind_s)].astype(np.float32)

                # 2. Spatial filtering
                R_sp = ifft2(I_F_sp * Gs, axes=(0, 1))

                # 3. Spatial downsampling (decimation)
                ss = self.spacing_spatial if np.isscalar(self.spacing_spatial) else self.spacing_spatial[ind_s]
                if ss != 1:
                    offset = ss // 2
                    R_sp = R_sp[offset::ss, offset::ss, :]

                # 4. Temporal FFT
                R_F_t = fft(R_sp, axis=2)
                del R_sp # Free spatial response memory

                for ind_v in range(self.num_velocities):
                    Gt = self.filters_temporal[(ind_v, ind_s)]

                    # Condition to match MATLAB: skip zero velocity if orientation >= 180
                    if Gt is not None and not (self.velocities[ind_v] == 0 and self.orientations[ind_o] >= 180):
                        Gt = Gt.astype(np.float32)
                        ind_filt = self.ind_tuning_to_ind_filt(ind_v + 1, ind_o + 1)

                        # 5. Temporal filtering
                        res = ifft(R_F_t * Gt, axis=2).astype(np.complex64)

                        # 6. Temporal downsampling
                        st = self.spacing_temporal if np.isscalar(self.spacing_temporal) else self.spacing_temporal[ind_v, ind_s]
                        if st != 1:
                            offset_t = st // 2
                            res = res[:, :, offset_t::st]

                        # Store filter response
                        R[(ind_filt, ind_s)] = res

                        # --- Z-SCORE CALCULATION ---
                        if flux_noise_std is not None:
                            # 7. Match noise resolution to filtered response resolution
                            # If downsampling occurred, we need to resize the noise estimate
                            if flux_noise_std.shape != res.shape:
                                # Calculate zoom factors for H, W, and T
                                zoom_factors = [r / n for r, n in zip(res.shape, flux_noise_std.shape)]
                                # 'order=0' replicates MATLAB 'nearest' neighbor interpolation
                                noise_std_os = zoom(flux_noise_std, zoom_factors, order=0)
                            else:
                                noise_std_os = flux_noise_std

                            # 8. Calculate standard deviation of response
                            # (Using the helper function converted previously)
                            # std_R = flux_noise_std * sqrt(0.5 * filter_energy)
                            filter_energy = self.filter_energies[ind_filt, ind_s]
                            std_R = stdev_response(noise_std_os, filter_energy)

                            # 9. Compute Z-Score: magnitude of response / expected noise std
                            Rz[(ind_filt, ind_s)] = (np.abs(res) / std_R).astype(np.float32)
                        else:
                            # If no noise estimate provided, Rz is effectively infinite (infinite SNR)
                            Rz[(ind_filt, ind_s)] = np.full(res.shape, np.inf, dtype=np.float32)
                        
                        del res

                del R_F_t
                gc.collect()

        return R, Rz
    
    def tuning_directions(self):
        """
        Returns an (N, 3) array of the 3D tuning directions [nx, ny, nt]
        for each spatiotemporal filter.
        """
        num_filters = self.num_orientations * self.num_velocities
        dirs = np.zeros((num_filters, 3), dtype=np.float32)

        for ind_filt in range(num_filters):
            # Using our 1-based indexing helper to match the MATLAB loop logic
            ind_v, ind_o = self.ind_filt_to_ind_tuning(ind_filt + 1)

            # Convert to 0-based indices for array lookups
            theta = np.radians(self.orientations[ind_o - 1])
            v = self.velocities[ind_v - 1]

            # Create the tuning vector
            # Spatial orientation dictates X and Y. Velocity dictates T.
            dirs[ind_filt, 0] = np.cos(theta)
            dirs[ind_filt, 1] = np.sin(theta)
            dirs[ind_filt, 2] = v

            # Note: If your original MATLAB implementation normalized these
            # to unit length (direction cosines), you would do:
            # norm = np.sqrt(1 + v**2)
            # dirs[ind_filt, :] /= norm

        return dirs