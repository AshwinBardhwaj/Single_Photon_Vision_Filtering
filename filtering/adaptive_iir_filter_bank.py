import numpy as np
import gc
from scipy.fft import fft2, ifft2
from scipy.signal import lfilter
from scipy.ndimage import zoom
from filtering.stdev_response import stdev_response

class AdaptiveIIRFilterBank3DSepT:
    # Number of frames to pre-pad for IIR warmup (eliminates startup transient)
    N_WARMUP = 10

    def __init__(self):
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
        self.b_val = 0.8

        self.filters_spatial = {}
        self.filters_temporal_b = {}
        self.filters_temporal_a = {}
        self.filter_energies = None

    def set_up_filters(self):
        diff_ori = self.orientations[1] - self.orientations[0] if len(self.orientations) > 1 else 60
        self.dtheta_max = min(self.dtheta_max, diff_ori)
        self.num_orientations = len(self.orientations)
        self.num_velocities = len(self.velocities)

        rows, cols, frames = self.input_size

        def get_freqs(n):
            return np.fft.fftfreq(n)

        fx = get_freqs(cols).reshape(1, cols, 1).astype(np.float32)
        fy = get_freqs(rows).reshape(rows, 1, 1).astype(np.float32)
        ft = get_freqs(frames).reshape(1, 1, frames).astype(np.float32)

        f_mag_sp = np.hypot(fx, fy)
        f_theta = np.degrees(np.arctan2(fy, fx))
        sinftheta = np.sin(np.radians(f_theta))
        cosftheta = np.cos(np.radians(f_theta))

        lp_cutoff, lp_n = 0.45, 15
        lp = 1.0 / (1.0 + (f_mag_sp / lp_cutoff) ** (2 * lp_n))

        log_sigma = np.log(self.sigma_on_f)

        for i1 in range(self.num_scales, 0, -1):
            ind_s = i1 - 1
            f0 = (1.0 / self.min_wavelength) / (self.mult ** (self.num_scales - i1))

            with np.errstate(divide='ignore'):
                lg_r = np.exp((-(np.log(f_mag_sp / f0)) ** 2) / (2 * log_sigma ** 2))
            lg_r *= lp
            lg_r[0, 0, 0] = 0

            for i2 in range(self.num_orientations):
                angl = self.orientations[i2]
                rad_angl = np.radians(angl)
                ds_s = sinftheta * np.cos(rad_angl) - cosftheta * np.sin(rad_angl)
                dc_s = cosftheta * np.cos(rad_angl) + sinftheta * np.sin(rad_angl)
                dftheta = np.abs(np.degrees(np.arctan2(ds_s, dc_s)))

                dftheta = 180 * np.minimum(1, dftheta / self.dtheta_max)
                spread = (np.cos(np.radians(dftheta)) + 1) / 2
                self.filters_spatial[(i2, ind_s)] = lg_r * spread

            ft_c = -f0 * self.velocities
            for i2 in range(self.num_velocities):
                if abs(ft_c[i2]) > (1 / 3):
                    self.filters_temporal_b[(i2, ind_s)] = None
                    self.filters_temporal_a[(i2, ind_s)] = None
                else:
                    if ft_c[i2] == 0:
                        omega_0 = 0.0
                    else:
                        omega_0 = 2 * np.tan(np.pi * ft_c[i2])

                    q = self.b_val / (self.b_val - 1j * omega_0 + 2)
                    r = (self.b_val - 1j * omega_0 - 2) / (self.b_val - 1j * omega_0 + 2)

                    b = (q**3) * np.array([1, 3, 3, 1], dtype=np.complex64)
                    a = np.array([1, 3*r, 3*(r**2), r**3], dtype=np.complex64)

                    self.filters_temporal_b[(i2, ind_s)] = b
                    self.filters_temporal_a[(i2, ind_s)] = a

        self.set_up_filter_energies()
        return self

    def set_up_filter_energies(self):
        from scipy.fft import ifftn, fftshift
        self.filter_energies = np.zeros((self.num_velocities * self.num_orientations, self.num_scales))
        frames = self.input_size[2]

        # Get the IIR impulse response and convert to frequency domain once per (vel, scale)
        impulse = np.zeros(frames, dtype=np.complex64)
        impulse[0] = 1.0
        Gt_freq_cache = {}
        for ind_v in range(self.num_velocities):
            for ind_s in range(self.num_scales):
                b = self.filters_temporal_b[(ind_v, ind_s)]
                a = self.filters_temporal_a[(ind_v, ind_s)]
                if b is not None and a is not None:
                    h_t = lfilter(b, a, impulse)
                    Gt_freq_cache[(ind_v, ind_s)] = np.fft.fft(h_t).reshape(1, 1, frames).astype(np.complex64)

        for ind_s in range(self.num_scales):
            for ind_filt in range(self.num_orientations * self.num_velocities):
                ind_v, ind_o = self.ind_filt_to_ind_tuning(ind_filt + 1)

                Gs = self.filters_spatial[(ind_o - 1, ind_s)].astype(np.float32)
                
                if (ind_v - 1, ind_s) in Gt_freq_cache:
                    Gt_freq = Gt_freq_cache[(ind_v - 1, ind_s)]

                    # Construct full 3D filter in frequency domain and compute
                    # energy via Parseval's theorem, matching LogGaborBank3DSepT
                    G_3d = Gs * Gt_freq
                    g = fftshift(ifftn(G_3d))
                    self.filter_energies[ind_filt, ind_s] = np.sum(np.abs(g) ** 2)

                    del G_3d, g
                    gc.collect()

    def ind_tuning_to_ind_filt(self, ind_v, ind_o):
        return (ind_v - 1) * self.num_orientations + (ind_o - 1)

    def ind_filt_to_ind_tuning(self, ind_filt):
        ind_v = 1 + ((ind_filt - 1) // self.num_orientations)
        ind_o = 1 + ((ind_filt - 1) % self.num_orientations)
        return ind_v, ind_o

    @staticmethod
    def _iir_filter_warmup(b, a, x, n_warmup, axis=2):
        """Apply IIR filter with warmup padding to eliminate startup transient.
        Pads the input by repeating the first frame, filters, then trims."""
        if n_warmup > 0:
            pad_frame = np.take(x, [0], axis=axis)
            pad = np.repeat(pad_frame, n_warmup, axis=axis)
            x_padded = np.concatenate([pad, x], axis=axis)
            y = lfilter(b, a, x_padded, axis=axis)
            return np.take(y, range(n_warmup, y.shape[axis]), axis=axis).astype(np.complex64)
        return lfilter(b, a, x, axis=axis).astype(np.complex64)

    def response(self, I, flux_noise_std=None):
        R = {}
        Rz = {}

        I_F_sp = fft2(I.astype(np.float32), axes=(0, 1))

        for ind_s in range(self.num_scales):
            for ind_o in range(self.num_orientations):
                Gs = self.filters_spatial[(ind_o, ind_s)].astype(np.float32)

                R_sp = ifft2(I_F_sp * Gs, axes=(0, 1))

                ss = self.spacing_spatial if np.isscalar(self.spacing_spatial) else self.spacing_spatial[ind_s]
                if ss != 1:
                    offset = ss // 2
                    R_sp = R_sp[offset::ss, offset::ss, :]

                # Instead of temporal FFT, we filter sequentially along axis 2
                for ind_v in range(self.num_velocities):
                    b = self.filters_temporal_b[(ind_v, ind_s)]
                    a = self.filters_temporal_a[(ind_v, ind_s)]

                    if b is not None and not (self.velocities[ind_v] == 0 and self.orientations[ind_o] >= 180):
                        ind_filt = self.ind_tuning_to_ind_filt(ind_v + 1, ind_o + 1)

                        # IIR filtering with warmup to avoid startup transient
                        res = self._iir_filter_warmup(b, a, R_sp, self.N_WARMUP, axis=2)

                        st = self.spacing_temporal if np.isscalar(self.spacing_temporal) else self.spacing_temporal[ind_v, ind_s]
                        if st != 1:
                            offset_t = st // 2
                            res = res[:, :, offset_t::st]

                        R[(ind_filt, ind_s)] = res

                        if flux_noise_std is not None:
                            if flux_noise_std.shape != res.shape:
                                zoom_factors = [r / n for r, n in zip(res.shape, flux_noise_std.shape)]
                                noise_std_os = zoom(flux_noise_std, zoom_factors, order=0)
                            else:
                                noise_std_os = flux_noise_std

                            filter_energy = self.filter_energies[ind_filt, ind_s]
                            std_R = stdev_response(noise_std_os, filter_energy)

                            Rz[(ind_filt, ind_s)] = (np.abs(res) / std_R).astype(np.float32)
                        else:
                            pass
                        
                        del res

                del R_sp
                gc.collect()

        return R, Rz
    
    def tuning_directions(self):
        num_filters = self.num_orientations * self.num_velocities
        dirs = np.zeros((num_filters, 3), dtype=np.float32)

        for ind_filt in range(num_filters):
            ind_v, ind_o = self.ind_filt_to_ind_tuning(ind_filt + 1)

            theta = np.radians(self.orientations[ind_o - 1])
            v = self.velocities[ind_v - 1]

            dirs[ind_filt, 0] = np.cos(theta)
            dirs[ind_filt, 1] = np.sin(theta)
            dirs[ind_filt, 2] = v

        return dirs

    def reconstruct(self, R):
        """
        Reconstructs the signal by summing the real parts of the filter responses.
        Args:
            R: Filter response dictionary returned by response()
        Returns:
            Reconstructed video array of the same shape as the input
        """
        first_key = list(R.keys())[0]
        reconstructed = np.zeros_like(R[first_key], dtype=np.float32)
        for key, res in R.items():
            if res.shape == reconstructed.shape:
                reconstructed += np.real(res).astype(np.float32)
        return reconstructed

    def response_and_edges(self, I, flux_noise_std=None):
        import sys
        from vision.feature_detection.phase_congruency import pc_sumR_flux_noise_std
        from vision.feature_detection.structure_tensor import features_3D_structure_tensor
        from scipy.fft import fft2, ifft2
        
        I_F_sp = fft2(I.astype(np.float32), axes=(0, 1))
        
        dirs = self.tuning_directions()
        num_dirs = len(dirs)
        
        PC_x2 = np.zeros(I.shape, dtype=np.float32)
        PC_y2 = np.zeros(I.shape, dtype=np.float32)
        PC_t2 = np.zeros(I.shape, dtype=np.float32)
        PC_xy = np.zeros(I.shape, dtype=np.float32)
        PC_yt = np.zeros(I.shape, dtype=np.float32)
        PC_xt = np.zeros(I.shape, dtype=np.float32)

        valid_dirs = []

        for ind_filt in range(num_dirs):
            ind_v, ind_o = self.ind_filt_to_ind_tuning(ind_filt + 1)
            dir_n = dirs[ind_filt]
            
            sum_R = np.zeros(I.shape, dtype=np.complex64)
            sum_MagR = np.zeros(I.shape, dtype=np.float32)
            max_MagR = None
            
            valid_scales = 0
            filt_energies = []
            
            for ind_s in range(self.num_scales):
                b = self.filters_temporal_b[(ind_v - 1, ind_s)]
                a = self.filters_temporal_a[(ind_v - 1, ind_s)]
                
                if b is not None and not (self.velocities[ind_v - 1] == 0 and self.orientations[ind_o - 1] >= 180):
                    Gs = self.filters_spatial[(ind_o - 1, ind_s)].astype(np.float32)
                    R_sp = ifft2(I_F_sp * Gs, axes=(0, 1))
                    res = self._iir_filter_warmup(b, a, R_sp, self.N_WARMUP, axis=2)
                    
                    sum_R += res
                    MagR = np.abs(res)
                    sum_MagR += MagR
                    if max_MagR is None:
                        max_MagR = MagR.copy()
                    else:
                        max_MagR = np.maximum(max_MagR, MagR)
                        
                    valid_scales += 1
                    filt_energies.append(self.filter_energies[ind_filt, ind_s])
                    
            if valid_scales < 2:
                print(f'Not enough responses for direction #{ind_filt} = {dir_n}', file=sys.stderr)
                continue
                
            valid_dirs.append(ind_filt)
            
            noise_n = None
            if flux_noise_std is not None:
                noise_n = pc_sumR_flux_noise_std(flux_noise_std, filt_energies)
                
            width = (sum_MagR / np.maximum(np.finfo(np.float32).eps, max_MagR) - 1.0) / (valid_scales - 1)
            weight = 1.0 / (1.0 + np.exp((0.5 - width) * 10.0))
            
            XEnergy = np.abs(sum_R)
            XEnergy = np.minimum(XEnergy, sum_MagR)
            
            if noise_n is not None:
                z_XEnergy = XEnergy / noise_n
                w_noise = 1.0 - np.exp(-np.maximum(0.0, z_XEnergy - 2.0))
                weight = weight * w_noise
                
            ratio = XEnergy / np.maximum(sum_MagR, np.finfo(np.float32).eps)
            ratio = np.minimum(1.0, ratio)
            PC = weight * np.maximum(0.0, 1.0 - 1.5 * np.arccos(ratio))
            
            ex, ey, et = dir_n[0], dir_n[1], dir_n[2]
            PC2 = PC ** 2
            PC_x2 += PC2 * (ex * ex)
            PC_y2 += PC2 * (ey * ey)
            PC_t2 += PC2 * (et * et)
            PC_xy += PC2 * (ex * ey)
            PC_yt += PC2 * (ey * et)
            PC_xt += PC2 * (ex * et)
            
            # Allow garbage collection
            del sum_R, sum_MagR, max_MagR, PC, PC2
            gc.collect()

        dirs_valid = dirs[valid_dirs]
        norm_factor = 3.0 / np.sum(dirs_valid ** 2)
        PC_x2 *= norm_factor
        PC_y2 *= norm_factor
        PC_t2 *= norm_factor
        PC_xy *= norm_factor
        PC_yt *= norm_factor
        PC_xt *= norm_factor
        
        edges, _, _ = features_3D_structure_tensor(PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt)
        return edges


class BidirAdaptiveIIRFilterBank3DSepT(AdaptiveIIRFilterBank3DSepT):
    """Zero-phase variant: applies the IIR temporal filter forward then backward.

    Eliminates the causal phase lag that causes AdaptiveIIR to smear fast-moving
    objects (balls, etc.). The backward pass filters the time-reversed forward
    result, so the combined response is |H(ω)|² with zero phase shift.

    Trade-offs vs AdaptiveIIR:
      - Runtime ~2× (two lfilter passes per filter)
      - Full sequence must be in memory — no true streaming advantage
      - Velocity selectivity is preserved in magnitude but the squared response
        means both +v and −v tunings reinforce each other slightly
    """

    @staticmethod
    def _iir_filter_warmup(b, a, x, n_warmup, axis=2):
        # Forward pass with warmup
        if n_warmup > 0:
            pad = np.repeat(np.take(x, [0], axis=axis), n_warmup, axis=axis)
            y_fwd = lfilter(b, a, np.concatenate([pad, x], axis=axis), axis=axis)
            y_fwd = np.take(y_fwd, range(n_warmup, y_fwd.shape[axis]), axis=axis).astype(np.complex64)
        else:
            y_fwd = lfilter(b, a, x, axis=axis).astype(np.complex64)

        # Backward pass: filter the time-reversed forward result, then reverse back
        y_rev = np.flip(y_fwd, axis=axis)
        if n_warmup > 0:
            pad = np.repeat(np.take(y_rev, [0], axis=axis), n_warmup, axis=axis)
            y_bwd = lfilter(b, a, np.concatenate([pad, y_rev], axis=axis), axis=axis)
            y_bwd = np.take(y_bwd, range(n_warmup, y_bwd.shape[axis]), axis=axis).astype(np.complex64)
        else:
            y_bwd = lfilter(b, a, y_rev, axis=axis).astype(np.complex64)

        return np.flip(y_bwd, axis=axis).copy()
