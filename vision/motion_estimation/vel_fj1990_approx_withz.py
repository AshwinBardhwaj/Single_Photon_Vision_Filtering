import numpy as np
from calc_2d_velocity_pxwise import calc_2d_velocity_pxwise, solve_linear_eq_2x2_sym
from vel_scalecomb_max import vel_scalecomb_max

def vel_fj1990_approx_withz(Rz, dirs, zscore_weight_thresh=6, wsum_rel_thresh=1, solve_2d_pxwise_k=None):
    """
    Fleet & Jepson (1990) approximation using z-score weighted reliability.
    
    Args:
        Rz: 2D dictionary or object array of cell responses [num_dirs, num_scales]
        dirs: Array of shape (num_dirs, 3) representing [nx, ny, nt]
        zscore_weight_thresh: Threshold for z-score suppression
        wsum_rel_thresh: Threshold for total weight reliability
        solve_2d_pxwise_k: Regularization parameter for the solver
    """
    num_dirs, num_scales = Rz.shape if isinstance(Rz, np.ndarray) else (len(Rz), len(Rz[0]))
    
    V0 = [None] * num_scales
    Rel0 = [None] * num_scales
    eps_single = np.finfo(np.float32).eps

    for i1 in range(num_scales):
        nx2_list, nxny_list, ny2_list = [], [], []
        vnx_list, vny_list, w_list = [], [], []

        for i2 in range(num_dirs):
            # Access response (handling dict or array indexing)
            res = Rz[i2, i1] if isinstance(Rz, np.ndarray) else Rz.get((i2, i1))
            
            if res is not None:
                nx, ny, nt = dirs[i2, 0], dirs[i2, 1], dirs[i2, 2]
                
                # Normal velocity calculation: vn = -nt / hypot(nx, ny)
                vn = -nt / np.maximum(eps_single, np.hypot(nx, ny))
                
                # Store oriented components
                nx2_list.append(nx**2)
                nxny_list.append(nx * ny)
                ny2_list.append(ny**2)
                vnx_list.append(vn * nx)
                vny_list.append(vn * ny)
                
                # Z-score based weighting: w = 1 - exp(-max(0, Rz - thresh))
                weight = 1.0 - np.exp(-np.maximum(0.0, res - zscore_weight_thresh))
                w_list.append(weight)

        if not w_list:
            continue

        # Calculate total weight sum across all directions for this scale
        # We stack and sum along the last axis to get spatial reliability
        w_stack = np.stack(w_list, axis=-1)
        wsum = np.sum(w_stack, axis=-1)
        
        # Scale reliability based on weight sum
        rel_wsum = 1.0 - np.exp(-np.maximum(0.0, wsum - wsum_rel_thresh))
        
        # Calculate pixel-wise velocity (referencing your previous function)
        v0, rel_i1d = calc_2d_velocity_pxwise(
            nx2_list, nxny_list, ny2_list, vnx_list, vny_list, w_list, 
            k=solve_2d_pxwise_k
        )
        
        Rel0[i1] = rel_wsum * rel_i1d
        V0[i1] = v0

    # Scale combination
    if num_scales > 1:
        # Assumes you have translated vel_scalecomb_max
        V, Rel = vel_scalecomb_max(V0, Rel0)
    else:
        V = V0[0]
        Rel = Rel0[0]

    return V, Rel, V0, Rel0