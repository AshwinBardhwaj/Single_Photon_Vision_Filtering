import numpy as np
import gc
from scipy.ndimage import gaussian_filter, maximum_filter, label
from filtering.stdev_response import stdev_response
from vision.feature_detection.structure_tensor import features_3D_structure_tensor


def _nms_2d_slices(energy, rel_threshold=0.7):
    """Spatial NMS: keep pixels within rel_threshold of the local 3x3 peak."""
    local_max = maximum_filter(energy, size=(3, 3, 1))
    return np.where(energy >= rel_threshold * local_max, energy, 0.0)


def _hysteresis_3d(energy, low_t, high_t, min_vol=10):
    """3D hysteresis: connected components of weak edges that contain a strong edge."""
    strong = energy > high_t
    weak   = energy > low_t
    lbl, n = label(weak, structure=np.ones((3, 3, 3)))
    if n == 0:
        return np.zeros_like(energy)
    strong_sum = np.bincount(lbl.ravel(), weights=strong.ravel())
    sizes      = np.bincount(lbl.ravel())
    valid      = (strong_sum > 0) & (sizes >= min_vol)
    out        = np.zeros_like(energy)
    out[np.isin(lbl, np.where(valid)[0])] = energy[np.isin(lbl, np.where(valid)[0])]
    return out


class GradientSteerableFilterBank3DSepT:
    """
    Gradient-based steerable filter bank (Freeman & Adelson 1991 G2/H2).

    Spatial filters: Gaussian pre-smoothing + analytic gradient steering.
      For orientation theta the response is the quadrature pair:
        R_even = cos²θ·Ixx + sin²θ·Iyy + 2cosθsinθ·Ixy   (2nd derivative)
        R_odd  = cosθ·Ix   + sinθ·Iy                        (1st derivative)
        R      = R_even + j·R_odd

    Any orientation is exactly steerable from the Ix/Iy/Ixx/Iyy/Ixy basis —
    unlike Log-Gabor, no new convolution is needed per angle.

    Temporal: Gaussian smoothing with sigma=tau, then finite-difference It.
      Velocity tuning via R += v·It  (linear approximation).

    Sigma per scale is derived from min_wavelength/mult to match the other
    filter banks (sigma_x = 1/(2π·f_c), same Gaussian that underlies G2).

    Edge detection uses NMS + 3-D hysteresis on the max-energy map, giving
    the response_and_edges() path used by the eval pipeline.
    """

    # response_and_edges + response both recompute the gradient basis, so the
    # measured wall-time counts the work twice. run_pipeline divides by this.
    runtime_divisor = 2

    def __init__(self):
        self.input_size       = [256, 512, 50]
        self.num_scales       = 3
        self.orientations     = np.array([0, 60, 120])
        self.num_orientations = 3
        self.velocities       = np.array([1, -1, 1 / 3, -1 / 3, 0])
        self.num_velocities   = 5
        self.min_wavelength   = 3
        self.mult             = 2.1
        self.tau              = 2.0    # temporal Gaussian sigma [frames]
        self.min_vol          = 10     # hysteresis minimum component volume [voxels]

        self.sigma_values    = []
        self.filter_energies = None

    # ── Setup ─────────────────────────────────────────────────────────────────

    def set_up_filters(self):
        self.num_orientations = len(self.orientations)
        self.num_velocities   = len(self.velocities)

        # Derive sigma_x per scale from spatial frequency parameters.
        # G2 impulse response uses sigma_x = 1 / (2π · f_c).
        self.sigma_values = [
            1.0 / (2.0 * np.pi * (1.0 / self.min_wavelength)
                   / (self.mult ** (self.num_scales - 1 - s)))
            for s in range(self.num_scales)
        ]

        self.filter_energies = self._compute_filter_energies()
        return self

    def _compute_filter_energies(self):
        """Numerical filter energies via a small delta-impulse volume."""
        nh, nw, nt = 64, 64, 32
        imp = np.zeros((nh, nw, nt), dtype=np.float32)
        imp[nh // 2, nw // 2, nt // 2] = 1.0

        n_filts  = self.num_velocities * self.num_orientations
        energies = np.ones((n_filts, self.num_scales), dtype=np.float32)

        for ind_s, sigma in enumerate(self.sigma_values):
            basis = self._compute_basis(imp, sigma, self.tau)
            for ind_v, vel in enumerate(self.velocities):
                for ind_o, theta_deg in enumerate(self.orientations):
                    res = self._complex_response(basis, theta_deg, vel)
                    e   = float(np.sum(np.abs(res) ** 2))
                    ind_filt = self.ind_tuning_to_ind_filt(ind_v + 1, ind_o + 1)
                    energies[ind_filt, ind_s] = max(e, 1e-12)
        return energies

    # ── Basis computation ─────────────────────────────────────────────────────

    def _compute_basis(self, I, sigma, tau):
        I_s = gaussian_filter(I.astype(np.float64), sigma=[sigma, sigma, tau])
        b = {}
        b['Ix']  = np.gradient(I_s, axis=1).astype(np.float32)
        b['Iy']  = np.gradient(I_s, axis=0).astype(np.float32)
        b['It']  = np.gradient(I_s, axis=2).astype(np.float32)
        b['Ixx'] = np.gradient(b['Ix'].astype(np.float64), axis=1).astype(np.float32)
        b['Iyy'] = np.gradient(b['Iy'].astype(np.float64), axis=0).astype(np.float32)
        b['Ixy'] = np.gradient(b['Ix'].astype(np.float64), axis=0).astype(np.float32)
        return b

    def _complex_response(self, basis, theta_deg, velocity=0.0):
        theta  = np.radians(theta_deg)
        c, s   = float(np.cos(theta)), float(np.sin(theta))
        R_even = (c ** 2 * basis['Ixx'] + s ** 2 * basis['Iyy']
                  + 2 * c * s * basis['Ixy']).astype(np.float32)
        R_odd  = (c * basis['Ix'] + s * basis['Iy']).astype(np.float32)
        res    = (R_even + 1j * R_odd).astype(np.complex64)
        if velocity != 0.0:
            res += (float(velocity) * basis['It']).astype(np.complex64)
        return res

    # ── Index helpers ─────────────────────────────────────────────────────────

    def ind_tuning_to_ind_filt(self, ind_v, ind_o):
        return (ind_v - 1) * self.num_orientations + (ind_o - 1)

    def ind_filt_to_ind_tuning(self, ind_filt):
        ind_v = 1 + ((ind_filt - 1) // self.num_orientations)
        ind_o = 1 + ((ind_filt - 1) % self.num_orientations)
        return ind_v, ind_o

    def tuning_directions(self):
        dirs = []
        for vel in self.velocities:
            for theta_deg in self.orientations:
                theta = np.radians(theta_deg)
                dirs.append([float(np.cos(theta)), float(np.sin(theta)), float(vel)])
        return np.array(dirs, dtype=np.float32)

    # ── Response ──────────────────────────────────────────────────────────────

    def response(self, I, flux_noise_std=None):
        R  = {}
        Rz = {}

        for ind_s, sigma in enumerate(self.sigma_values):
            basis = self._compute_basis(I, sigma, self.tau)

            for ind_v, vel in enumerate(self.velocities):
                for ind_o, theta_deg in enumerate(self.orientations):
                    res      = self._complex_response(basis, theta_deg, vel)
                    ind_filt = self.ind_tuning_to_ind_filt(ind_v + 1, ind_o + 1)
                    R[(ind_filt, ind_s)] = res

                    if flux_noise_std is not None:
                        fe = self.filter_energies[ind_filt, ind_s]
                        Rz[(ind_filt, ind_s)] = (
                            np.abs(res) / stdev_response(flux_noise_std, fe)
                        ).astype(np.float32)
                    else:
                        Rz[(ind_filt, ind_s)] = np.full(
                            res.shape, np.inf, dtype=np.float32)

            gc.collect()

        return R, Rz

    def response_and_edges(self, I, flux_noise_std=None):
        """
        Edge detection using max-energy NMS + 3-D hysteresis.
        Orientation/velocity geometry comes from the gradient structure tensor.
        """
        H, W, N = I.shape

        max_energy = np.zeros((H, W, N), dtype=np.float32)
        PC_x2 = np.zeros((H, W, N), dtype=np.float32)
        PC_y2 = np.zeros((H, W, N), dtype=np.float32)
        PC_t2 = np.zeros((H, W, N), dtype=np.float32)
        PC_xy = np.zeros((H, W, N), dtype=np.float32)
        PC_yt = np.zeros((H, W, N), dtype=np.float32)
        PC_xt = np.zeros((H, W, N), dtype=np.float32)

        for sigma in self.sigma_values:
            basis = self._compute_basis(I, sigma, self.tau)

            for theta_deg in self.orientations:
                e = np.abs(self._complex_response(basis, theta_deg, 0.0))
                max_energy = np.maximum(max_energy, e)

            PC_x2 += basis['Ix'] ** 2 + basis['Ixx'] ** 2
            PC_y2 += basis['Iy'] ** 2 + basis['Iyy'] ** 2
            PC_t2 += basis['It'] ** 2
            PC_xy += basis['Ix'] * basis['Iy']
            PC_yt += basis['Iy'] * basis['It']
            PC_xt += basis['Ix'] * basis['It']
            gc.collect()

        edges, _, _ = features_3D_structure_tensor(
            PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt)

        sharp = _nms_2d_slices(max_energy)

        if flux_noise_std is not None:
            noise_floor = np.maximum(flux_noise_std, 1e-6)
        else:
            noise_floor = np.maximum(max_energy * 0.05, 1e-6)

        snr       = max_energy / noise_floor
        valid_snr = snr[max_energy > 0]
        lo        = float(np.percentile(valid_snr, 50)) if valid_snr.size else 0.5
        hi        = float(np.percentile(valid_snr, 80)) if valid_snr.size else 1.0

        edges['strength'] = _hysteresis_3d(
            sharp, lo * noise_floor, hi * noise_floor, min_vol=self.min_vol)
        return edges
