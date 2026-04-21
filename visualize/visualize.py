import numpy as np
from scipy.ndimage import maximum_filter, distance_transform_edt
from matplotlib.colors import hsv_to_rgb

from .flow_to_color import flow_to_color

def rescale_array(arr, in_min=None, in_max=None):
    """Helper function to replicate MATLAB's rescale."""
    lo = np.min(arr) if in_min is None else in_min
    hi = np.max(arr) if in_max is None else in_max
    if hi == lo:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def optical_flow_colormap_img(L=200):
    """Generates an optical flow color map image."""
    # x increases left to right
    # y up to down
    if L % 2 == 1:
        L += 1

    val = np.arange(-L / 2, L / 2 + 1)
    xx, yy = np.meshgrid(val, val)

    #     # Assuming flow_to_color expects (H, W, 2)
    img = flow_to_color(np.stack((xx, yy), axis=-1))
    return img


def vis_complex_number(z, p=1.4, scale_r=True):
    """Visualizes a complex number field using HSV color mapping."""
    assert z.ndim == 2, "Input must be a 2D array"

    #     th = np.angle(z)
    r = np.abs(z)

    th = rescale_array(th)
    if scale_r:
        r = rescale_array(r)

    if p != 1.0:
        r = r ** p

    o = np.ones_like(r)
    # Stack into an (H, W, 3) array for hsv_to_rgb
    hsv = np.stack((th, r, o), axis=-1)
    zc = hsv_to_rgb(hsv)
    return zc


def vis_edge(estr, thicken=False, prctile_lim=(75, 97)):
    """Visualizes edge structures with optional thresholding and thickening."""
    estr = np.array(estr, copy=True)

    # Handle 3D inputs recursively
    if estr.ndim > 2:
        evis = np.zeros_like(estr)
        for n in range(estr.shape[-1]):
            evis[..., n] = vis_edge(estr[..., n], thicken, prctile_lim)
        return evis

    if np.issubdtype(estr.dtype, np.floating):
        if prctile_lim is not None:
            lo = np.percentile(estr, min(prctile_lim))
            hi = np.percentile(estr, max(prctile_lim))
        else:
            lo, hi = np.min(estr), np.max(estr)

        estr = np.clip(estr, lo, hi)
        estr = rescale_array(estr, in_min=lo, in_max=hi)

    if thicken:
        # Replicates ordfilt2(estr, 9, true([3 3])) -> 3x3 max filter
        estr = maximum_filter(estr, size=3)

    if estr.dtype == np.uint8:
        evis = 255 - estr
    else:
        evis = 1.0 - estr

    # Add black border
    evis[:, 0] = 0
    evis[:, -1] = 0
    evis[0, :] = 0
    evis[-1, :] = 0

    return evis


def vis_flo_dense(flo, rel_bin=None, max_flo=None, thicken=0):
    """Visualizes dense optical flow with optional binary masking and thickening."""
    if thicken > 0:
        assert rel_bin is not None, "rel_bin mask is required if thicken > 0"

    # Assuming input flo is (H, W, 2, N) as per original size(flo, 4)
    N = flo.shape[3]
    flo_v_list = []

    for n in range(N):
        u_n = flo[:, :, 0, n].copy()
        v_n = flo[:, :, 1, n].copy()

        if rel_bin is not None:
            rel_bin_n = rel_bin[:, :, n].astype(bool).copy()

            if thicken > 0:
                # scipy edt calculates distance to 0, so we invert the mask
                D, ind_src = distance_transform_edt(~rel_bin_n, return_indices=True)
                ind_fill = (D > 0) & (D < thicken)

                u_n[ind_fill] = u_n[ind_src[0][ind_fill], ind_src[1][ind_fill]]
                v_n[ind_fill] = v_n[ind_src[0][ind_fill], ind_src[1][ind_fill]]
                rel_bin_n = rel_bin_n | ind_fill

            u_n[~rel_bin_n] = 0
            v_n[~rel_bin_n] = 0

        flo_n = np.stack((u_n, v_n), axis=-1)
        # Assuming flow_to_color accepts max_flo as a parameter
        colored_flo = flow_to_color(flo_n, max_flo)
        flo_v_list.append(colored_flo)

    # Output shape: (H, W, 3, N)
    flo_v = np.stack(flo_v_list, axis=-1)
    return flo_v