import numpy as np

def rescale_prctile(x, p_hi=97, p_lo=0):
    x = np.asarray(x)

    min_val = np.percentile(x, p_lo)
    max_val = np.percentile(x, p_hi)

    if max_val == min_val:
        # If all values are the same, returning an array of zeros is standard fallback
        return np.zeros_like(x, dtype=float)

    xs = (x - min_val) / (max_val - min_val)

    return xs