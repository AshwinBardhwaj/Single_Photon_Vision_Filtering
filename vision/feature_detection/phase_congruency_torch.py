import torch


def phase_congruency_1D_torch(R_scales, noise_std_est=None, weight_g=10.0, cut_off=0.5, deviation_gain=1.5,
                              noise_thresh_zmin=2.0):
    num_scales = len(R_scales)

    sum_R = torch.zeros_like(R_scales[0])
    sum_MagR = torch.zeros_like(torch.abs(R_scales[0]))
    max_MagR = torch.zeros_like(sum_MagR)

    for s in range(num_scales):
        sum_R = sum_R + R_scales[s]
        MagR = torch.abs(R_scales[s])
        sum_MagR = sum_MagR + MagR
        max_MagR = torch.maximum(max_MagR, MagR)

    width = (sum_MagR / (max_MagR + 1e-8) - 1.0) / (num_scales - 1)
    weight = 1.0 / (1.0 + torch.exp((cut_off - width) * weight_g))

    XEnergy = torch.abs(sum_R)
    XEnergy = torch.minimum(XEnergy, sum_MagR)

    if noise_std_est is None:
        noise_std_est = 1e-8

    z_XEnergy = XEnergy / noise_std_est
    w_noise = 1.0 - torch.exp(-torch.relu(z_XEnergy - noise_thresh_zmin))
    weight = weight * w_noise

    ratio = XEnergy / (sum_MagR + 1e-8)
    ratio = torch.clamp(ratio, min=-1.0 + 1e-7, max=1.0 - 1e-7)
    PC = weight * torch.relu(1.0 - deviation_gain * torch.acos(ratio))

    return PC


def phase_congruency_3D_torch(responses, dirs, num_scales, num_orientations, num_velocities, flux_noise_std=None,
                              filter_energies=None):
    num_dirs = num_orientations * num_velocities

    ref_shape = responses[0].shape
    device = responses[0].device

    PC_x2 = torch.zeros(ref_shape, device=device)
    PC_y2 = torch.zeros(ref_shape, device=device)
    PC_t2 = torch.zeros(ref_shape, device=device)
    PC_xy = torch.zeros(ref_shape, device=device)
    PC_yt = torch.zeros(ref_shape, device=device)
    PC_xt = torch.zeros(ref_shape, device=device)

    for ind_dir in range(num_dirs):
        R_n = [responses[ind_dir + s * num_dirs] for s in range(num_scales)]
        dir_n = dirs[ind_dir]

        noise_n = None
        if flux_noise_std is not None and filter_energies is not None:
            energy_sum = torch.sum(
                torch.sqrt(torch.stack([filter_energies[ind_dir + s * num_dirs] for s in range(num_scales)]) + 1e-8))
            factor = torch.sqrt(torch.tensor(0.5, device=device)) * energy_sum
            noise_n = torch.clamp(flux_noise_std * factor, min=1e-8)

        PC = phase_congruency_1D_torch(R_n, noise_std_est=noise_n)

        ex, ey, et = dir_n[0], dir_n[1], dir_n[2]
        PC2 = PC ** 2
        PC_x2 = PC_x2 + PC2 * (ex * ex)
        PC_y2 = PC_y2 + PC2 * (ey * ey)
        PC_t2 = PC_t2 + PC2 * (et * et)
        PC_xy = PC_xy + PC2 * (ex * ey)
        PC_yt = PC_yt + PC2 * (ey * et)
        PC_xt = PC_xt + PC2 * (ex * et)

    norm_factor = 3.0 / (torch.sum(dirs[:num_dirs] ** 2) + 1e-8)

    return PC_x2 * norm_factor, PC_y2 * norm_factor, PC_t2 * norm_factor, PC_xy * norm_factor, PC_yt * norm_factor, PC_xt * norm_factor