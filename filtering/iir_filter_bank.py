import numpy as np
import gc
from scipy.fft import fft, ifft, fft2, ifft2, ifftn, fftshift
from scipy.ndimage import zoom
from filtering.stdev_response import stdev_response

class IIRFilterBank3DSepT:
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
        self.filters_temporal = {}
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
        z_inv = np.exp(-1j * 2 * np.pi * ft)

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
                    self.filters_temporal[(i2, ind_s)] = None
                else:
                    if ft_c[i2] == 0:
                        omega_0 = 0.0
                    else:
                        omega_0 = 2 * np.tan(np.pi * ft_c[i2])

                    q = self.b_val / (self.b_val - 1j * omega_0 + 2)
                    r = (self.b_val - 1j * omega_0 - 2) / (self.b_val - 1j * omega_0 + 2)

                    num = (q**3) * (1 + 3*z_inv + 3*(z_inv**2) + (z_inv**3))
                    den = 1 + 3*r*z_inv + 3*(r**2)*(z_inv**2) + (r**3)*(z_inv**3)
                    H = num / den

                    self.filters_temporal[(i2, ind_s)] = H.astype(np.complex64)

        self.set_up_filter_energies()
        return self

    def set_up_filter_energies(self):
        self.filter_energies = np.zeros((self.num_velocities * self.num_orientations, self.num_scales))
        for ind_s in range(self.num_scales):
            for ind_filt in range(self.num_orientations * self.num_velocities):
                ind_v, ind_o = self.ind_filt_to_ind_tuning(ind_filt + 1)

                Gs = self.filters_spatial[(ind_o - 1, ind_s)].astype(np.float32)
                Gt = self.filters_temporal[(ind_v - 1, ind_s)]

                if Gt is not None:
                    Gt = Gt.astype(np.complex64)
                    g = fftshift(ifftn(Gs * Gt))
                    self.filter_energies[ind_filt, ind_s] = np.sum(np.abs(g) ** 2)

                    del g, Gs, Gt
                    gc.collect()

    def ind_tuning_to_ind_filt(self, ind_v, ind_o):
        return (ind_v - 1) * self.num_orientations + (ind_o - 1)

    def ind_filt_to_ind_tuning(self, ind_filt):
        ind_v = 1 + ((ind_filt - 1) // self.num_orientations)
        ind_o = 1 + ((ind_filt - 1) % self.num_orientations)
        return ind_v, ind_o

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

                R_F_t = fft(R_sp, axis=2)
                del R_sp

                for ind_v in range(self.num_velocities):
                    Gt = self.filters_temporal[(ind_v, ind_s)]

                    if Gt is not None and not (self.velocities[ind_v] == 0 and self.orientations[ind_o] >= 180):
                        Gt = Gt.astype(np.complex64)
                        ind_filt = self.ind_tuning_to_ind_filt(ind_v + 1, ind_o + 1)

                        res = ifft(R_F_t * Gt, axis=2).astype(np.complex64)

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
                            Rz[(ind_filt, ind_s)] = np.full(res.shape, np.inf, dtype=np.float32)
                        
                        del res

                del R_F_t
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
