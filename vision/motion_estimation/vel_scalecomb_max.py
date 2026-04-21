import numpy as np

def vel_scalecomb_max(V_list, W_list):
    """
    Combines velocity estimates across multiple scales by picking the scale 
    with the maximum (Magnitude * Reliability) at each pixel.
    
    Args:
        V_list: List of velocity arrays [H, W, 2, N] (one per scale)
        W_list: List of reliability arrays [H, W, N] (one per scale)
        
    Returns:
        Vm: Combined velocity array [H, W, 2, N]
        Wm: Combined reliability array [H, W, N]
    """
    # 1. Stack into 5D/4D matrices for vectorized operations
    # Vmat: (H, W, 2, N, S)
    Vmat = np.stack(V_list, axis=-1)
    # Wmat: (H, W, N, S)
    Wmat = np.stack(W_list, axis=-1)

    # 2. Calculate Magnitude along the '2' axis (Cartesian components)
    # Vmag: (H, W, N, S)
    Vmag = np.linalg.norm(Vmat, ord=2, axis=2)

    # 3. Find the index of the scale that maximizes Mag * Reliability
    # smax: (H, W, N) - contains values from 0 to num_scales-1
    smax = np.argmax(Vmag * Wmat, axis=-1)

    # 4. Extract the winning velocities and reliabilities
    # We create a grid of indices to select the best scale at each pixel
    h, w, _, n, s = Vmat.shape
    idx_h, idx_w, idx_n = np.ogrid[:h, :w, :n]

    # Select reliability
    Wm = Wmat[idx_h, idx_w, idx_n, smax]

    # Select Velocity components
    # Vmat is (H, W, 2, N, S), we want to pick from S using smax
    # We extract x and y components separately or use a specialized transpose
    Vmx = Vmat[idx_h, idx_w, 0, idx_n, smax]
    Vmy = Vmat[idx_h, idx_w, 1, idx_n, smax]

    # 5. Recombine into (H, W, 2, N)
    Vm = np.stack([Vmx, Vmy], axis=2)

    return Vm, Wm