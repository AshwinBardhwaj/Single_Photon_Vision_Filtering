import numpy as np

def calc_2d_velocity_pxwise(nx2_list, nxny_list, ny2_list, vnx_list, vny_list, w_list=None, k=None):
    """
    Calculates 2D velocity pixel-wise by aggregating oriented components.
    
    Args:
        nx2_list: List of arrays (one per orientation) for N_x^2
        nxny_list: List of arrays for N_x * N_y
        ny2_list: List of arrays for N_y^2
        vnx_list: List of arrays for V_normal_x
        vny_list: List of arrays for V_normal_y
        w_list: List of weight arrays (optional)
        k: Regularization parameter for the linear solver
        
    Returns:
        V: 4D array [Height, Width, 2, Frames] representing (vx, vy)
        rel: Reliability/condition map (if solver supports it)
    """
    num_orient = len(nx2_list)
    
    # Handle default weights
    if w_list is None:
        w_list = [1.0] * num_orient

    # Check for scalar/early return logic (matching MATLAB)
    first_elem = nx2_list[0]
    first_w = w_list[0]
    
    if np.isscalar(first_elem) and np.isscalar(first_w):
        return np.zeros((1, 1, 2, 1)), True

    # Determine output dimensions (H, W, N_frames)
    if not np.isscalar(first_elem):
        h, w, n = first_elem.shape if first_elem.ndim == 3 else (*first_elem.shape, 1)
    else:
        h, w, n = first_w.shape if first_w.ndim == 3 else (*first_w.shape, 1)

    # Stack lists into 4D arrays [H, W, N_frames, Orientations]
    # MATLAB's cat(4, ...) is equivalent to np.stack(..., axis=3)
    nx2 = np.stack(nx2_list, axis=-1)
    nxny = np.stack(nxny_list, axis=-1)
    ny2 = np.stack(ny2_list, axis=-1)
    vnx = np.stack(vnx_list, axis=-1)
    vny = np.stack(vny_list, axis=-1)
    w = np.stack(w_list, axis=-1) if not np.isscalar(first_w) else np.array(w_list)

    # Check if weights are all identity to save computation
    if np.all(w == 1.0):
        ntn_11 = np.sum(nx2, axis=-1)
        ntn_12 = np.sum(nxny, axis=-1)
        ntn_22 = np.sum(ny2, axis=-1)
        ntvn_1 = np.sum(vnx, axis=-1)
        ntvn_2 = np.sum(vny, axis=-1)
    else:
        ntn_11 = np.sum(w * nx2, axis=-1)
        ntn_12 = np.sum(w * nxny, axis=-1)
        ntn_22 = np.sum(w * ny2, axis=-1)
        ntvn_1 = np.sum(w * vnx, axis=-1)
        ntvn_2 = np.sum(w * vny, axis=-1)

    # Solve the 2x2 symmetric system
    # This assumes you have a python version of solve_linear_eq_2x2_sym
    results = solve_linear_eq_2x2_sym(ntn_11, ntn_12, ntn_22, ntvn_1, ntvn_2, k)
    
    if isinstance(results, tuple) and len(results) == 3:
        vx, vy, rel = results
    else:
        vx, vy = results
        rel = None

    # Construct the final V array [H, W, 2, N]
    # In MATLAB: V(:,:,1,:) = vx;
    # We expand dims to allow proper assignment into the "2" channel axis
    # V = np.zeros((h, w, 2, n), dtype=nx2.dtype)
    V = np.zeros((int(h), int(w), 2, int(n)), dtype=nx2.dtype)
    V[:, :, 0, :] = vx.reshape(h, w, n)
    V[:, :, 1, :] = vy.reshape(h, w, n)

    return (V, rel) if rel is not None else V


def solve_linear_eq_2x2_sym(AtA_11, AtA_12, AtA_22, Atb_1, Atb_2, k=1e-2):
    """Replaces solve_linear_eq_2x2_sym.m"""
    det = (AtA_11 * AtA_22) - (AtA_12 * AtA_12)

    inv_AtA_11 = AtA_22
    inv_AtA_12 = -AtA_12
    inv_AtA_22 = AtA_11

    x1 = inv_AtA_11 * Atb_1 + inv_AtA_12 * Atb_2
    x2 = inv_AtA_12 * Atb_1 + inv_AtA_22 * Atb_2

    eps = np.finfo(float).eps
    x1 = x1 / (eps + det)
    x2 = x2 / (eps + det)

    rel = det > k * ((AtA_11 + AtA_22) ** 2)

    return x1, x2, rel