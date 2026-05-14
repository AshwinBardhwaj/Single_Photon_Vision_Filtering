import numpy as np
import gc
from scipy.fft import fft, ifft, fft2, ifft2, ifftn, fftshift
from scipy.ndimage import zoom
from filtering.stdev_response import stdev_response


class SteerableFilterBank3DSepT:
    """
    Regular steerable filter bank based on nth-order derivative-of-Gaussian basis
    (Freeman & Adelson 1991).

    Spatial filters: analytic G_n steerable filters.
      In the frequency domain the filter pointed at angle theta is:

          H(fx, fy, theta) = (f_n / f_c)^n * exp(-0.5 * (f_mag / f_c)^2)

      where f_n = fx*cos(theta) + fy*sin(theta) is the component of spatial
      frequency along the filter's normal direction, and f_mag = sqrt(fx^2+fy^2).
      The one-sided (analytic) spectrum is retained by zeroing the half-plane
      where f_n < 0.

    Temporal filters: Log-Gabor (same as LogGaborBank3DSepT).
    """

    def __init__(self):
        self.input_size = [256, 512, 50]
        self.num_scales = 3
        self.orientations = np.array([0, 60, 120])
        self.num_orientations = len(self.orientations)
        self.velocities = np.array([1, -1, 1 / 3, -1 / 3, 0])
        self.num_velocities = len(self.velocities)

        self.min_wavelength = 3
        self.mult = 2.1   # scale spacing (frequency ratio between adjacent scales)
        self.order = 2    # derivative order: 2 = G2 (optimal for edges)

        self.spacing_spatial = 1
        self.spacing_temporal = 1

        self.filters_spatial = {}
        self.filters_temporal = {}
        self.filter_energies = None

    def set_up_filters(self):
        self.num_orientations = len(self.orientations)
        self.num_velocities = len(self.velocities)

        rows, cols, frames = self.input_size
        n = self.order
        f0 = 1.0 / self.min_wavelength
        log_sigma = np.log(0.55)

        fx = np.fft.fftfreq(cols).reshape(1, cols, 1).astype(np.float32)
        fy = np.fft.fftfreq(rows).reshape(rows, 1, 1).astype(np.float32)
        ft = np.fft.fftfreq(frames).reshape(1, 1, frames).astype(np.float32)

        f_mag_sp = np.hypot(fx, fy)

        for i1 in range(self.num_scales, 0, -1):
            ind_s = i1 - 1
            f_c = f0 / (self.mult ** (self.num_scales - i1))

            # Radial: nth-order derivative-of-Gaussian profile.
            # Peak is at f_mag = f_c * sqrt(n), falls off as a Gaussian either side.
            r = f_mag_sp / f_c
            rad_r = (r ** n) * np.exp(-0.5 * r ** 2)
            rad_r[0, 0, 0] = 0.0
            rad_max = rad_r.max()
            if rad_max > 0:
                rad_r = rad_r / rad_max

            for i2 in range(self.num_orientations):
                angl = np.radians(self.orientations[i2])
                cos_a, sin_a = float(np.cos(angl)), float(np.sin(angl))

                # Normal-direction frequency component
                f_n = fx * cos_a + fy * sin_a  # (rows, cols, 1)

                # cos(angle_diff) = f_n / f_mag, clamped to [-1, 1]
                with np.errstate(invalid='ignore'):
                    cos_ang = np.where(f_mag_sp > 0, f_n / f_mag_sp, 0.0)
                cos_ang = np.clip(cos_ang, -1.0, 1.0)

                # One-sided analytic spectrum: keep only the half-plane where f_n >= 0
                ang_r = np.where(cos_ang >= 0, cos_ang ** n, 0.0).astype(np.float32)

                self.filters_spatial[(i2, ind_s)] = rad_r * ang_r

            # Temporal: Log-Gabor filters (identical to LogGaborBank3DSepT)
            ft_c = -f_c * self.velocities
            for i2 in range(self.num_velocities):
                if ft_c[i2] == 0:
                    ft_c_mag = np.abs(ft_c)
                    f1 = (np.min(ft_c_mag[ft_c_mag > 0])
                          if (ft_c_mag > 0).any() else 0.1)
                    sigmaf = f1 / 2.0
                    self.filters_temporal[(i2, ind_s)] = np.exp(
                        -0.5 * (ft / sigmaf) ** 2).astype(np.float32)
                elif abs(ft_c[i2]) > (1.0 / 3.0):
                    self.filters_temporal[(i2, ind_s)] = None
                else:
                    lg_ft = np.zeros_like(ft)
                    with np.errstate(divide='ignore', invalid='ignore'):
                        ft_scaled = ft / ft_c[i2]
                    mask = ft_scaled > 0
                    lg_ft[mask] = np.exp(
                        -(np.log(ft_scaled[mask]) ** 2) / (2 * log_sigma ** 2))
                    self.filters_temporal[(i2, ind_s)] = lg_ft.astype(np.float32)

        self.set_up_filter_energies()
        return self

    def set_up_filter_energies(self):
        self.filter_energies = np.zeros(
            (self.num_velocities * self.num_orientations, self.num_scales))
        for ind_s in range(self.num_scales):
            for ind_filt in range(self.num_orientations * self.num_velocities):
                ind_v, ind_o = self.ind_filt_to_ind_tuning(ind_filt + 1)
                Gs = self.filters_spatial[(ind_o - 1, ind_s)].astype(np.float32)
                Gt = self.filters_temporal[(ind_v - 1, ind_s)]
                if Gt is not None:
                    Gt = Gt.astype(np.float32)
                    g = fftshift(ifftn(Gs * Gt))
                    self.filter_energies[ind_filt, ind_s] = float(np.sum(np.abs(g) ** 2))
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

                ss = (self.spacing_spatial if np.isscalar(self.spacing_spatial)
                      else self.spacing_spatial[ind_s])
                if ss != 1:
                    offset = ss // 2
                    R_sp = R_sp[offset::ss, offset::ss, :]

                R_F_t = fft(R_sp, axis=2)
                del R_sp

                for ind_v in range(self.num_velocities):
                    Gt = self.filters_temporal[(ind_v, ind_s)]
                    if Gt is not None and not (
                            self.velocities[ind_v] == 0
                            and self.orientations[ind_o] >= 180):
                        Gt = Gt.astype(np.float32)
                        ind_filt = self.ind_tuning_to_ind_filt(ind_v + 1, ind_o + 1)
                        res = ifft(R_F_t * Gt, axis=2).astype(np.complex64)

                        st = (self.spacing_temporal if np.isscalar(self.spacing_temporal)
                              else self.spacing_temporal[ind_v, ind_s])
                        if st != 1:
                            offset_t = st // 2
                            res = res[:, :, offset_t::st]

                        R[(ind_filt, ind_s)] = res

                        if flux_noise_std is not None:
                            if flux_noise_std.shape != res.shape:
                                zoom_factors = [rv / nv for rv, nv in
                                                zip(res.shape, flux_noise_std.shape)]
                                noise_std_os = zoom(flux_noise_std, zoom_factors, order=0)
                            else:
                                noise_std_os = flux_noise_std
                            filter_energy = self.filter_energies[ind_filt, ind_s]
                            std_R = stdev_response(noise_std_os, filter_energy)
                            Rz[(ind_filt, ind_s)] = (
                                np.abs(res) / std_R).astype(np.float32)
                        else:
                            Rz[(ind_filt, ind_s)] = np.full(
                                res.shape, np.inf, dtype=np.float32)
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


# Backward-compatibility alias
SteerablePyramidBank3DSepT = SteerableFilterBank3DSepT
