import numpy as np

# note writes eigenvalues/vectors with different combination of signs but is equivalent
# this file is needed for
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


def eig_3x3_sym(A11, A12, A13, A22, A23, A33):
    """Replaces eig_3x3_sym.m"""
    row1 = np.stack([A11, A12, A13], axis=-1)
    row2 = np.stack([A12, A22, A23], axis=-1)
    row3 = np.stack([A13, A23, A33], axis=-1)
    A = np.stack([row1, row2, row3], axis=-2)

    evals, evecs = np.linalg.eigh(A)

    evals = evals[..., ::-1]
    evecs = evecs[..., ::-1]

    evals = np.maximum(0, evals)

    return evals, evecs


def cross_product(a1, a2, a3, b1, b2, b3):
    y1 = a2 * b3 - a3 * b2
    y2 = a3 * b1 - a1 * b3
    y3 = a1 * b2 - a2 * b1
    return y1, y2, y3


def dircosd_3D(phid, thetad):
    """Replaces dircosd_3D.m"""
    phid = np.asarray(phid).reshape(-1, 1)
    thetad = np.asarray(thetad).reshape(-1, 1)

    phi_rad = np.deg2rad(phid)
    theta_rad = np.deg2rad(thetad)

    x = np.sin(phi_rad) * np.cos(theta_rad)
    y = np.sin(phi_rad) * np.sin(theta_rad)
    z = np.cos(phi_rad)

    return np.concatenate([x, y, z], axis=1).astype(np.float32)


if __name__ == '__main__':
    print("--- solve_linear_eq_2x2_sym ---")
    x1, x2, rel = solve_linear_eq_2x2_sym(np.array([2.0]), np.array([0.5]), np.array([3.0]),
                                          np.array([1.0]), np.array([2.0]))
    print(f"x1: {x1[0]:.4f}, x2: {x2[0]:.4f}, rel: {rel[0]}")

    print("\n--- eig_3x3_sym ---")
    evals, evecs = eig_3x3_sym(np.array([2.0]), np.array([-1.0]), np.array([0.0]),
                               np.array([2.0]), np.array([-1.0]),
                               np.array([2.0]))
    print(f"evals (descending):\n{np.round(evals, 4)}")
    print(f"evecs:\n{np.round(evecs, 4)}")

    print("\n--- cross_product ---")
    y1, y2, y3 = cross_product(1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    print(f"Result: [{y1}, {y2}, {y3}]")

    print("\n--- dircosd_3D ---")
    dc = dircosd_3D([45.0, 90.0], [45.0, 0.0])
    print(f"Direction Cosines:\n{np.round(dc, 4)}")