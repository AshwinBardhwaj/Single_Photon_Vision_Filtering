import numpy as np
import gc
from scipy.fft import fft, ifft, fft2, ifft2, ifftn, fftshift
from scipy.ndimage import zoom
from filtering.stdev_response import stdev_response


class DTCWTFilterBank3DSepT:
    """
    Frequency-domain approximation of the 2D Dual-Tree Complex Wavelet Transform
    (DT-CWT, Kingsbury 1999) with separable temporal Log-Gabor filtering.

    Spatial filters mimic the 6 DT-CWT subbands per level:
      - 6 fixed orientations at 15°, 45°, 75°, 105°, 135°, 165°
        (the canonical DT-CWT directions, spaced 30° apart)
      - One-sided raised-cosine angular window, 30° half-width, giving
        a tight 60°-wide partition that tiles 0–180° with unity sum
      - Log-Gabor radial profile (same as LogGaborBank3DSepT)

    Key advantages over the 3-orientation Log-Gabor / Steerable banks:
      - 2× finer orientation resolution (30° vs 60° spacing)
      - Approximate shift invariance from the analytic (one-sided) subbands
      - Tight angular tiling: adjacent subbands sum to 1 everywhere

    Temporal filters: identical Log-Gabor design as LogGaborBank3DSepT.

    Note: orientations are always locked to the 6 DT-CWT values regardless
    of any external configuration applied before set_up_filters().
    """

    # DT-CWT canonical orientations — never overridden externally
    _DTCWT_ORIENTATIONS = np.array([15, 45, 75, 105, 135, 165], dtype=np.float32)
    _DTCWT_DTHETA_MAX   = 30.0  # degrees — tight 60°-bandwidth window

    def __init__(self):
        self.input_size       = [256, 512, 50]
        self.num_scales       = 3
        self.orientations     = self._DTCWT_ORIENTATIONS.copy()
        self.num_orientations = len(self.orientations)
        self.velocities       = np.array([1, -1, 1 / 3, -1 / 3, 0])
        self.num_velocities   = len(self.velocities)
        self.min_wavelength   = 4
        self.mult             = 2.1
        self.sigma_on_f       = 0.55
        self.spacing_spatial  = 1
        self.spacing_temporal = 1

        self.filters_spatial  = {}
        self.filters_temporal = {}
        self.filter_energies  = None

    def set_up_filters(self):
        # Lock DT-CWT orientations regardless of external config
        self.orientations     = self._DTCWT_ORIENTATIONS.copy()
        self.num_orientations = len(self.orientations)
        self.num_velocities   = len(self.velocities)

        rows, cols, frames = self.input_size

        fx = np.fft.fftfreq(cols).reshape(1, cols, 1).astype(np.float32)
        fy = np.fft.fftfreq(rows).reshape(rows, 1, 1).astype(np.float32)
        ft = np.fft.fftfreq(frames).reshape(1, 1, frames).astype(np.float32)

        f_mag_sp  = np.hypot(fx, fy)
        f_theta   = np.degrees(np.arctan2(fy, fx))       # angle in [-180°, 180°]
        sinftheta = np.sin(np.radians(f_theta))
        cosftheta = np.cos(np.radians(f_theta))

        # Butterworth anti-aliasing lowpass (same as LogGaborBank3DSepT)
        lp_cutoff, lp_n = 0.45, 15
        lp = 1.0 / (1.0 + (f_mag_sp / lp_cutoff) ** (2 * lp_n))

        log_sigma = np.log(self.sigma_on_f)
        dtheta    = self._DTCWT_DTHETA_MAX

        for i1 in range(self.num_scales, 0, -1):
            ind_s = i1 - 1
            f0 = (1.0 / self.min_wavelength) / (self.mult ** (self.num_scales - i1))

            # Log-Gabor radial profile
            with np.errstate(divide='ignore'):
                lg_r = np.exp(-(np.log(f_mag_sp / f0)) ** 2 / (2 * log_sigma ** 2))
            lg_r *= lp
            lg_r[0, 0, 0] = 0.0

            for i2 in range(self.num_orientations):
                angl     = float(self.orientations[i2])
                rad_angl = np.radians(angl)

                # Signed angular difference from filter centre (wraps to [-180°, 180°])
                ds_s     = sinftheta * np.cos(rad_angl) - cosftheta * np.sin(rad_angl)
                dc_s     = cosftheta * np.cos(rad_angl) + sinftheta * np.sin(rad_angl)
                dftheta  = np.abs(np.degrees(np.arctan2(ds_s, dc_s)))

                # One-sided raised-cosine window: full width = 2*dtheta, zero outside
                # (same formula as LogGaborBank3DSepT but with dtheta=30° not 60°)
                dftheta_norm = 180.0 * np.minimum(1.0, dftheta / dtheta)
                spread = (np.cos(np.radians(dftheta_norm)) + 1.0) / 2.0

                self.filters_spatial[(i2, ind_s)] = (lg_r * spread).astype(np.float32)

            # Temporal Log-Gabor filters (identical to LogGaborBank3DSepT)
            ft_c = -f0 * self.velocities
            for i2 in range(self.num_velocities):
                if ft_c[i2] == 0:
                    ft_c_mag = np.abs(ft_c)
                    f1 = np.min(ft_c_mag[ft_c_mag > 0]) if (ft_c_mag > 0).any() else 0.1
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
        R  = {}
        Rz = {}

        I_F_sp = fft2(I.astype(np.float32), axes=(0, 1))

        for ind_s in range(self.num_scales):
            for ind_o in range(self.num_orientations):
                Gs   = self.filters_spatial[(ind_o, ind_s)].astype(np.float32)
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
                        Gt       = Gt.astype(np.float32)
                        ind_filt = self.ind_tuning_to_ind_filt(ind_v + 1, ind_o + 1)
                        res      = ifft(R_F_t * Gt, axis=2).astype(np.complex64)

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
                            std_R  = stdev_response(noise_std_os, filter_energy)
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
            theta = np.radians(float(self.orientations[ind_o - 1]))
            v     = self.velocities[ind_v - 1]
            dirs[ind_filt, 0] = np.cos(theta)
            dirs[ind_filt, 1] = np.sin(theta)
            dirs[ind_filt, 2] = v
        return dirs
