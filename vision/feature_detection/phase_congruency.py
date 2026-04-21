import sys
import numpy as np

EPS32 = np.finfo(np.float32).eps


def pc_sumR_flux_noise_std(flux_noise_std, filt_energies):
    filt_energies = np.asarray(filt_energies).ravel()
    factor = np.sqrt(0.5) * np.sum(np.sqrt(filt_energies))
    return np.maximum(EPS32, np.asarray(flux_noise_std) * factor)


def phase_congruency_1D(R, weight_g=10.0, cut_off=0.5,
                       noise_std_est=None, noise_thresh_zmin=2.0,
                       deviation_gain=1.5):
    num_scales = len(R)
    R0 = R[0]

    sum_R = np.zeros(R0.shape, dtype=R0.dtype)
    sum_MagR = np.zeros(R0.shape, dtype=np.real(R0).dtype)
    max_MagR = None

    for s in range(num_scales):
        sum_R = sum_R + R[s]
        MagR = np.abs(R[s])
        sum_MagR = sum_MagR + MagR
        if max_MagR is None:
            max_MagR = MagR.copy()
        else:
            max_MagR = np.maximum(max_MagR, MagR)

    width = (sum_MagR / np.maximum(EPS32, max_MagR) - 1.0) / (num_scales - 1)
    weight = 1.0 / (1.0 + np.exp((cut_off - width) * weight_g))

    XEnergy = np.abs(sum_R)

    tol = 1e-4
    if np.any(XEnergy - sum_MagR > tol):
        print('phase_congruency_1D: XEnergy > sum_MagR!', file=sys.stderr)

    XEnergy = np.minimum(XEnergy, sum_MagR)

    if noise_std_est is None:
        print('phase_congruency_1D: noise_std_est not supplied.', file=sys.stderr)
    else:
        z_XEnergy = XEnergy / noise_std_est
        w_noise = 1.0 - np.exp(-np.maximum(0.0, z_XEnergy - noise_thresh_zmin))
        weight = weight * w_noise

    ratio = XEnergy / np.maximum(sum_MagR, EPS32)
    ratio = np.minimum(1.0, ratio)
    PC = weight * np.maximum(0.0, 1.0 - deviation_gain * np.arccos(ratio))

    return PC


def phase_congruency_3D_structure_tensor(R, dirs,
                                         flux_noise_std=None,
                                         filter_energies=None,
                                         noise_thresh_zmin=2.0,
                                         input_size=None,
                                         return_per_dir=False,
                                         weight_g=10.0,
                                         cut_off=0.5,
                                         deviation_gain=1.5):
    dirs = np.asarray(dirs, dtype=np.float64)
    num_dirs = dirs.shape[0]
    assert len(R) == num_dirs

    noise_std_est = None
    if flux_noise_std is not None and filter_energies is not None:
        noise_std_est = []
        for n in range(num_dirs):
            noise_std_est.append(pc_sumR_flux_noise_std(flux_noise_std, filter_energies[n]))

    ref_shape = None
    ref_real_dtype = None
    for n in range(num_dirs):
        for s in range(len(R[n])):
            if R[n][s] is not None:
                ref_shape = R[n][s].shape
                ref_real_dtype = np.real(R[n][s]).dtype
                break
        if ref_shape is not None:
            break

    PC_x2 = np.zeros(ref_shape, dtype=ref_real_dtype)
    PC_y2 = np.zeros(ref_shape, dtype=ref_real_dtype)
    PC_t2 = np.zeros(ref_shape, dtype=ref_real_dtype)
    PC_xy = np.zeros(ref_shape, dtype=ref_real_dtype)
    PC_yt = np.zeros(ref_shape, dtype=ref_real_dtype)
    PC_xt = np.zeros(ref_shape, dtype=ref_real_dtype)

    PC_per_dir = [] if return_per_dir else None
    valid_dirs = []

    for ind_dir in range(num_dirs):
        R_n = [r for r in R[ind_dir] if r is not None]
        dir_n = dirs[ind_dir]

        if len(R_n) < 2:
            print(f'Not enough responses for direction #{ind_dir} = {dir_n}', file=sys.stderr)
            if PC_per_dir is not None:
                PC_per_dir.append(None)
            continue

        valid_dirs.append(ind_dir)

        if input_size is not None:
            resized = []
            for r in R_n:
                if tuple(r.shape) != tuple(input_size):
                    from scipy.ndimage import zoom
                    factors = [input_size[i] / r.shape[i] for i in range(3)]
                    resized.append(zoom(r, factors, order=3))
                else:
                    resized.append(r)
            R_n = resized

        noise_n = None
        if noise_std_est is not None:
            noise_n = noise_std_est[ind_dir]
            if np.ndim(noise_n) > 0 and noise_n.shape != R_n[0].shape:
                from scipy.ndimage import zoom
                factors = [R_n[0].shape[i] / noise_n.shape[i] for i in range(3)]
                noise_n = zoom(noise_n, factors, order=0)

        PC = phase_congruency_1D(R_n,
                                 weight_g=weight_g,
                                 cut_off=cut_off,
                                 noise_std_est=noise_n,
                                 noise_thresh_zmin=noise_thresh_zmin,
                                 deviation_gain=deviation_gain)
        if PC_per_dir is not None:
            PC_per_dir.append(PC)

        ex, ey, et = dir_n[0], dir_n[1], dir_n[2]
        PC2 = PC ** 2
        PC_x2 = PC_x2 + PC2 * (ex * ex)
        PC_y2 = PC_y2 + PC2 * (ey * ey)
        PC_t2 = PC_t2 + PC2 * (et * et)
        PC_xy = PC_xy + PC2 * (ex * ey)
        PC_yt = PC_yt + PC2 * (ey * et)
        PC_xt = PC_xt + PC2 * (ex * et)

    dirs_valid = dirs[valid_dirs]
    norm_factor = 3.0 / np.sum(dirs_valid ** 2)
    PC_x2 = norm_factor * PC_x2
    PC_y2 = norm_factor * PC_y2
    PC_t2 = norm_factor * PC_t2
    PC_xy = norm_factor * PC_xy
    PC_yt = norm_factor * PC_yt
    PC_xt = norm_factor * PC_xt

    if return_per_dir:
        return PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt, PC_per_dir
    return PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt