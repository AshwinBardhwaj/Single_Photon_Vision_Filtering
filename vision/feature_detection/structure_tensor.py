import numpy as np

from .feature_utils import edge_props_3D, edge_coherence, corner_coherence

EPS32 = np.finfo(np.float32).eps


def eig_3x3_sym_video(Ix2, Ixy, Ixt, Iy2, Iyt, It2, enforce_evec_z_nonneg=True):
    shape = Ix2.shape
    N = int(np.prod(shape))
    dtype = Ix2.dtype
    M = np.empty((N, 3, 3), dtype=dtype)
    M[:, 0, 0] = Ix2.ravel()
    M[:, 1, 1] = Iy2.ravel()
    M[:, 2, 2] = It2.ravel()
    M[:, 0, 1] = M[:, 1, 0] = Ixy.ravel()
    M[:, 1, 2] = M[:, 2, 1] = Iyt.ravel()
    M[:, 0, 2] = M[:, 2, 0] = Ixt.ravel()

    w, v = np.linalg.eigh(M)
    w = w[:, ::-1]
    v = v[:, :, ::-1]

    if enforce_evec_z_nonneg:
        flip = v[:, 2, :] < 0
        signs = np.where(flip, -1.0, 1.0).astype(dtype)
        v = v * signs[:, np.newaxis, :]

    evals = w.reshape(shape + (3,))
    evecs = v.reshape(shape + (3, 3))
    return evals, evecs


def features_3D_structure_tensor(Ix2, Iy2, It2, Ixy, Iyt, Ixt,
                                 enforce_evec_z_nonneg=True):
    evals, evecs = eig_3x3_sym_video(Ix2, Ixy, Ixt, Iy2, Iyt, It2,
                                     enforce_evec_z_nonneg=enforce_evec_z_nonneg)

    ex1 = evecs[..., 0, 0]
    ey1 = evecs[..., 1, 0]
    et1 = evecs[..., 2, 0]

    _, eori, ephi, evn = edge_props_3D(ex1, ey1, et1)

    edges = {
        'strength': evals[..., 0],
        'orientation': eori,
        'temporal_angle': ephi,
        'normal_velocity': evn,
        'coherence': edge_coherence(evals[..., 0], evals[..., 1]),
    }

    corners = {
        'strength': evals[..., 1],
        'coherence': corner_coherence(evals[..., 1], evals[..., 2]),
    }
    ev3_xy = evecs[..., 0:2, 2]
    ev3_z = evecs[..., 2, 2]
    vel = ev3_xy / np.maximum(EPS32, ev3_z[..., np.newaxis])
    corners['velocity'] = np.moveaxis(vel, 3, 2)

    transients = {
        'strength': evals[..., 2],
    }

    return edges, corners, transients