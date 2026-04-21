import numpy as np
import matplotlib.colors as mcolors

# Middlebury's flowToColor
def flow_to_color(flow, max_flow=None):
    u = flow[:, :, 0].astype(float)
    v = flow[:, :, 1].astype(float)

    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / np.pi

    H = (a + 1.0) / 2.0

    if max_flow is None or max_flow <= 0:
        maxrad = np.max(rad)
        if maxrad == 0:
            maxrad = 1.0
    else:
        maxrad = float(max_flow)

    S = np.minimum(rad / maxrad, 1.0)

    V = np.ones_like(u)

    hsv = np.stack((H, S, V), axis=-1)

    rgb = mcolors.hsv_to_rgb(hsv)
    img = (rgb * 255).astype(np.uint8)

    return img