import sys
import numpy as np
from scipy.ndimage import map_coordinates, maximum_filter


def _interp2(V, xx, yy, method='linear'):
    order = {'nearest': 0, 'linear': 1, 'cubic': 3}[method]
    coords = np.stack([yy.ravel(), xx.ravel()])
    out = map_coordinates(V, coords, order=order, mode='nearest').reshape(xx.shape)
    return out


def _interp3(V, xx, yy, tt, method='linear'):
    order = {'nearest': 0, 'linear': 1, 'cubic': 3}[method]
    coords = np.stack([yy.ravel(), xx.ravel(), tt.ravel()])
    out = map_coordinates(V, coords, order=order, mode='nearest').reshape(xx.shape)
    return out


def nms2(I, ori=None, radius=1.3, min_branch_length=0, interp_method='linear'):
    oriented = ori is not None

    if oriented:
        thetavals = np.deg2rad(np.arange(181))
        xoff = radius * np.cos(thetavals)
        yoff = radius * np.sin(thetavals)

        rows, cols = I.shape[0], I.shape[1]
        frames = I.shape[2] if I.ndim >= 3 else 1
        I3 = I if I.ndim >= 3 else I[..., None]

        xv = np.arange(cols)
        yv = np.arange(rows)
        xx, yy = np.meshgrid(xv, yv, indexing='xy')

        I_thin = np.zeros_like(I3)
        for n in range(frames):
            frame_n = I3[:, :, n]
            ori_n = ori[:, :, n] if ori.ndim >= 3 else ori
            ind_theta = np.clip(np.round(ori_n).astype(int), 0, 180)
            xx1 = np.clip(xx + xoff[ind_theta], 0, cols - 1)
            yy1 = np.clip(yy + yoff[ind_theta], 0, rows - 1)
            xx2 = np.clip(xx - xoff[ind_theta], 0, cols - 1)
            yy2 = np.clip(yy - yoff[ind_theta], 0, rows - 1)
            I1 = _interp2(frame_n, xx1, yy1, interp_method)
            I2 = _interp2(frame_n, xx2, yy2, interp_method)
            isnonmax = (frame_n < I1) | (frame_n < I2)
            if min_branch_length > 0:
                try:
                    from skimage.morphology import skeletonize
                    skel = skeletonize(~isnonmax)
                    from skimage.morphology import remove_small_objects
                    skel = remove_small_objects(skel, min_size=min_branch_length)
                    isnonmax = ~skel
                except ImportError:
                    pass
            frame_out = frame_n.copy()
            frame_out[isnonmax] = 0
            I_thin[:, :, n] = frame_out

        if I.ndim < 3:
            return I_thin[:, :, 0]
        return I_thin
    else:
        radius_int = int(np.ceil(radius))
        I3 = I if I.ndim >= 3 else I[..., None]
        I_thin = I3.copy()
        for n in range(I3.shape[2]):
            In = I3[:, :, n]
            I_localmax = maximum_filter(In, size=radius_int, mode='constant', cval=0)
            lose = In != I_localmax
            In_out = In.copy()
            In_out[lose] = 0
            I_thin[:, :, n] = In_out
        if I.ndim < 3:
            return I_thin[:, :, 0]
        return I_thin


def nms3(I, ori=None, phi=None, radius=1.7, interp_method='linear'):
    oriented = phi is not None
    if oriented:
        assert ori is not None

    if oriented:
        thetavals = np.deg2rad(np.arange(361))
        phivals = np.deg2rad(np.arange(91))

        toff = radius * np.cos(phivals)
        xoff = radius * np.sin(phivals)[:, None] * np.cos(thetavals)[None, :]
        yoff = radius * np.sin(phivals)[:, None] * np.sin(thetavals)[None, :]

        rows, cols, frames = I.shape

        xv = np.arange(cols)
        yv = np.arange(rows)
        tv = np.arange(frames)
        yy, xx, tt = np.meshgrid(yv, xv, tv, indexing='ij')

        ind_toff = np.clip(np.round(phi).astype(int), 0, 90)
        ind_spoff0 = np.clip(np.round(ori).astype(int), 0, 360)

        xoff_sel = xoff[ind_toff, ind_spoff0]
        yoff_sel = yoff[ind_toff, ind_spoff0]
        toff_sel = toff[ind_toff]

        tt1 = np.clip(tt + toff_sel, 0, frames - 1)
        xx1 = np.clip(xx + xoff_sel, 0, cols - 1)
        yy1 = np.clip(yy + yoff_sel, 0, rows - 1)

        tt2 = np.clip(tt - toff_sel, 0, frames - 1)
        xx2 = np.clip(xx - xoff_sel, 0, cols - 1)
        yy2 = np.clip(yy - yoff_sel, 0, rows - 1)

        I1 = _interp3(I, xx1, yy1, tt1, interp_method)
        I2 = _interp3(I, xx2, yy2, tt2, interp_method)

        isnonmax = (I < I1) | (I < I2)
        I_thin = I.copy()
        I_thin[isnonmax] = 0
        return I_thin
    else:
        print('3D NMS not implemented yet without orientation', file=sys.stderr)
        return I.copy()