import numpy as np
from scipy.ndimage import maximum_filter

from .feature_utils import smoothorient_3D
from .nms import nms3


def _ordfilt2_video_max(I, win):
    out = np.zeros_like(I)
    for n in range(I.shape[2]):
        out[:, :, n] = maximum_filter(I[:, :, n], size=win, mode='constant', cval=0)
    return out


def corner_importance(strength, coherence, velocities,
                      coh_min=0.0, v_max=None, nms_win=3):
    if v_max is None:
        v_max = np.inf

    local_max_str = _ordfilt2_video_max(strength, int(nms_win))
    imp = strength * (strength == local_max_str)

    if coh_min > 0:
        imp = imp * (coherence > coh_min)

    if np.isfinite(v_max):
        v_mag = np.linalg.norm(velocities, axis=2)
        imp = imp * (v_mag < v_max)

    return imp


def postprocess_features_3D(edges, corners, transients,
                            edge_coh_min=0.0,
                            edge_ori_sm_sigma=1.0,
                            corner_coh_min=0.0,
                            corner_v_max=None,
                            corner_nms_win=3,
                            nms_interp_method='linear'):
    estr = edges['strength'].copy()
    if edge_coh_min > 0:
        estr = estr * (edges['coherence'] > edge_coh_min)

    eori_sm, ephi_sm = smoothorient_3D(edges['orientation'],
                                       edges['temporal_angle'],
                                       edge_ori_sm_sigma)
    edges['strength_nms'] = nms3(estr, eori_sm, ephi_sm,
                                 interp_method=nms_interp_method)

    corners['importance'] = corner_importance(corners['strength'],
                                              corners['coherence'],
                                              corners['velocity'],
                                              coh_min=corner_coh_min,
                                              v_max=corner_v_max,
                                              nms_win=corner_nms_win)
    return edges, corners, transients


def cat_t_features_3D(e1, c1, t1, e2, c2, t2):
    ec = {
        'strength': np.concatenate([e1['strength'], e2['strength']], axis=2),
        'orientation': np.concatenate([e1['orientation'], e2['orientation']], axis=2),
        'temporal_angle': np.concatenate([e1['temporal_angle'], e2['temporal_angle']], axis=2),
        'normal_velocity': np.concatenate([e1['normal_velocity'], e2['normal_velocity']], axis=2),
        'coherence': np.concatenate([e1['coherence'], e2['coherence']], axis=2),
    }
    if 'strength_nms' in e1 and 'strength_nms' in e2:
        ec['strength_nms'] = np.concatenate([e1['strength_nms'], e2['strength_nms']], axis=2)

    cc = {
        'strength': np.concatenate([c1['strength'], c2['strength']], axis=2),
        'coherence': np.concatenate([c1['coherence'], c2['coherence']], axis=2),
        'velocity': np.concatenate([c1['velocity'], c2['velocity']], axis=3),
    }
    if 'importance' in c1 and 'importance' in c2:
        cc['importance'] = np.concatenate([c1['importance'], c2['importance']], axis=2)

    tc = {
        'strength': np.concatenate([t1['strength'], t2['strength']], axis=2),
    }

    return ec, cc, tc


def crop_t_features_3D(e, c, t, n0=0, n1=None):
    if n1 is None:
        n1 = e['strength'].shape[2]

    er = dict(e)
    cr = dict(c)
    tr = dict(t)

    er['strength'] = e['strength'][:, :, n0:n1]
    er['orientation'] = e['orientation'][:, :, n0:n1]
    er['temporal_angle'] = e['temporal_angle'][:, :, n0:n1]
    er['normal_velocity'] = e['normal_velocity'][:, :, n0:n1]
    er['coherence'] = e['coherence'][:, :, n0:n1]
    if 'strength_nms' in e:
        er['strength_nms'] = e['strength_nms'][:, :, n0:n1]

    cr['strength'] = c['strength'][:, :, n0:n1]
    cr['coherence'] = c['coherence'][:, :, n0:n1]
    cr['velocity'] = c['velocity'][:, :, :, n0:n1]
    if 'importance' in c:
        cr['importance'] = c['importance'][:, :, n0:n1]

    tr['strength'] = t['strength'][:, :, n0:n1]

    return er, cr, tr