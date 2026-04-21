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

    # Safely extract target spatial dimensions based on MATLAB logic
    def get_shape(item):
        if np.isscalar(item) or np.ndim(item) == 0:
            return None
        s = np.shape(item)
        return (s[0], s[1] if len(s) > 1 else 1, s[2] if len(s) > 2 else 1)

    # MATLAB looks at nx2 first, and if it's scalar, falls back to inspecting w
    dims = get_shape(nx2_list[0])
    if dims is None:
        dims = get_shape(w_list[0])

    # Early exit for pure scalar inputs (all inputs are scalars)
    if dims is None:
        return np.zeros((1, 1, 2, 1)), True

    h, w_dim, n = dims

    # Stack lists into arrays on the last axis
    nx2 = np.stack(nx2_list, axis=-1)
    nxny = np.stack(nxny_list, axis=-1)
    ny2 = np.stack(ny2_list, axis=-1)
    vnx = np.stack(vnx_list, axis=-1)
    vny = np.stack(vny_list, axis=-1)
    w = np.stack(w_list, axis=-1) if get_shape(w_list[0]) is not None else np.array(w_list)

    # Pad missing spatial dimensions with 1s so broadcasting works cleanly
    # Example: if scalars were passed, shape goes from (15,) to (1, 1, 1, 15)
    # This allows it to cleanly broadcast against w which might be (H, W, N, 15)
    while nx2.ndim < 4:
        nx2 = np.expand_dims(nx2, axis=-2)
        nxny = np.expand_dims(nxny, axis=-2)
        ny2 = np.expand_dims(ny2, axis=-2)
        vnx = np.expand_dims(vnx, axis=-2)
        vny = np.expand_dims(vny, axis=-2)

    while getattr(w, 'ndim', 0) > 0 and w.ndim < 4:
        w = np.expand_dims(w, axis=-2)

    # Compute dot products
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
    results = solve_linear_eq_2x2_sym(ntn_11, ntn_12, ntn_22, ntvn_1, ntvn_2, k)

    if isinstance(results, tuple) and len(results) == 3:
        vx, vy, rel = results
    else:
        vx, vy = results
        rel = None

    # Construct the final V array [H, W, 2, N]
    V = np.zeros((h, w_dim, 2, n), dtype=nx2.dtype)
    V[:, :, 0, :] = np.reshape(vx, (h, w_dim, n))
    V[:, :, 1, :] = np.reshape(vy, (h, w_dim, n))

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

    if k is not None:
        rel = det > k * ((AtA_11 + AtA_22) ** 2)
    else:
        rel = det > 1e-5

    return x1, x2, rel