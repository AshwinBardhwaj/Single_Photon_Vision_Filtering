import numpy as np
from scipy.ndimage import gaussian_filter

EPS32 = np.finfo(np.float32).eps
EPS64 = np.finfo(np.float64).eps


def acotd(x):
    x = np.asarray(x, dtype=np.float64)
    out = 90.0 - np.degrees(np.arctan(x))
    out = np.where(x < 0, out - 180.0, out)
    return out


def edge_props_3D(ex, ey, et):
    eori = np.round(np.degrees(np.arctan2(ey, ex)))

    es = np.hypot(ex, ey)
    estr = np.hypot(es, et)

    evn = -et / np.maximum(EPS32, es)
    ephi = np.round(acotd(-evn))

    neg_phi = ephi < 0
    ephi = np.where(neg_phi, -ephi, ephi)
    evn = np.where(neg_phi, -evn, evn)
    eori = np.where(neg_phi, eori + 180, eori)

    neg_ori = eori < 0
    eori = np.where(neg_ori, eori + 360, eori)

    return estr, eori, ephi, evn


def edge_coherence(lam1, lam2):
    return (lam1 - lam2) / np.maximum(EPS32, lam1 + lam2)


def corner_coherence(lam2, lam3):
    return (lam2 - lam3) / np.maximum(EPS32, lam2 + lam3)


def _imgaussfilt_matlab(x, sigma, ndim):
    return gaussian_filter(x, sigma=[sigma] * ndim, mode='nearest', truncate=2.0)


def smoothorient(orid, sigma):
    cosori = _imgaussfilt_matlab(np.cos(np.radians(orid)), sigma, 2)
    sinori = _imgaussfilt_matlab(np.sin(np.radians(orid)), sigma, 2)
    smorid = np.degrees(np.arctan2(sinori, cosori))
    smorid = np.where(smorid < 0, smorid + 180, smorid)
    return smorid


def smoothorient_3D(orid, phid, sigma):
    rt = _imgaussfilt_matlab(np.cos(np.radians(phid)), sigma, 3)
    rx = _imgaussfilt_matlab(np.sin(np.radians(phid)) * np.cos(np.radians(orid)), sigma, 3)
    ry = _imgaussfilt_matlab(np.sin(np.radians(phid)) * np.sin(np.radians(orid)), sigma, 3)
    rs = np.hypot(rx, ry)

    smphid = np.round(acotd(rt / (EPS64 + rs)))
    smorid = np.round(np.degrees(np.arctan2(ry, rx)))

    neg_phid = smphid < 0
    smphid = np.where(neg_phid, -smphid, smphid)
    smorid = np.where(neg_phid, smorid + 180, smorid)

    neg_orid = smorid < 0
    smorid = np.where(neg_orid, smorid + 360, smorid)

    return smorid, smphid