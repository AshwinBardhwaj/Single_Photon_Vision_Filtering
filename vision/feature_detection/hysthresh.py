import numpy as np


def _label_and_prune(bw_2d, seed_mask_2d, min_edge_length):
    try:
        from skimage.measure import label as _sk_label
        from skimage.morphology import remove_small_objects as _sk_rso
        lbl = _sk_label(bw_2d, connectivity=2)
        seeds = np.unique(lbl[seed_mask_2d])
        seeds = seeds[seeds > 0]
        out = np.isin(lbl, seeds)
        if min_edge_length > 0:
            out = _sk_rso(out, min_size=int(min_edge_length))
        return out
    except ImportError:
        from scipy.ndimage import label as _sp_label
        struct8 = np.ones((3, 3), dtype=bool)
        lbl, _ = _sp_label(bw_2d, structure=struct8)
        seeds = np.unique(lbl[seed_mask_2d])
        seeds = seeds[seeds > 0]
        out = np.isin(lbl, seeds)
        if min_edge_length > 0:
            lbl2, ncomp = _sp_label(out, structure=struct8)
            sizes = np.bincount(lbl2.ravel())
            keep = sizes >= int(min_edge_length)
            keep[0] = False
            out = keep[lbl2]
        return out


def hysthresh_2(estr, thresh=None, min_edge_length=0):
    if thresh is None:
        thresh = [0.5, 0.2]
    thresh = np.atleast_1d(thresh).astype(float)

    if thresh.size == 1:
        hi_thresh = float(thresh[0])
        lo_thresh = 0.4 * hi_thresh
    else:
        lo_thresh = float(np.min(thresh))
        hi_thresh = float(np.max(thresh))

    is_3d = estr.ndim >= 3
    E3 = estr if is_3d else estr[..., None]

    emap = E3 > hi_thresh
    above_lo = E3 > lo_thresh

    out = np.zeros(E3.shape, dtype=bool)
    for n in range(E3.shape[2]):
        out[:, :, n] = _label_and_prune(above_lo[:, :, n], emap[:, :, n], min_edge_length)

    if not is_3d:
        return out[:, :, 0]
    return out