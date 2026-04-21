import numpy as np


def _area_filter_2d(bw_2d, min_size):
    if min_size <= 0:
        return bw_2d
    try:
        from skimage.morphology import remove_small_objects
        return remove_small_objects(bw_2d, min_size=int(min_size))
    except ImportError:
        from scipy.ndimage import label as _sp_label
        struct8 = np.ones((3, 3), dtype=bool)
        lbl, _ = _sp_label(bw_2d, structure=struct8)
        sizes = np.bincount(lbl.ravel())
        keep = sizes >= int(min_size)
        keep[0] = False
        return keep[lbl]


def hysthresh_2(estr, thresh=None, min_edge_length=0):
    if thresh is None:
        thresh = [0.5, 0.2]
    thresh = np.atleast_1d(thresh).astype(float)

    if thresh.size == 1:
        hi_thresh = float(thresh[0])
    else:
        hi_thresh = float(np.max(thresh))

    is_3d = estr.ndim >= 3
    E3 = estr if is_3d else estr[..., None]

    emap = E3 > hi_thresh
    out = emap.copy()

    if min_edge_length > 0:
        for n in range(out.shape[2]):
            out[:, :, n] = _area_filter_2d(out[:, :, n], min_edge_length)

    if not is_3d:
        return out[:, :, 0]
    return out