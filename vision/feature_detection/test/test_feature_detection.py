import os
import sys
import numpy as np
from scipy.io import loadmat

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PKG_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..', '..'))
if _PKG_ROOT not in sys.path:
    sys.path.insert(0, _PKG_ROOT)

from vision.feature_detection import (
    pc_sumR_flux_noise_std,
    phase_congruency_1D,
    phase_congruency_3D_structure_tensor,
    features_3D_structure_tensor,
    edge_props_3D,
    edge_coherence,
    corner_coherence,
    smoothorient,
    smoothorient_3D,
    nms3,
    corner_importance,
    postprocess_features_3D,
    cat_t_features_3D,
    crop_t_features_3D,
    hysthresh_2,
)


def _max_abs_diff(a, b, mask=None):
    a = np.asarray(a).astype(np.float64)
    b = np.asarray(b).astype(np.float64)
    if a.shape != b.shape:
        return float('inf'), f'shape mismatch {a.shape} vs {b.shape}'
    finite = np.isfinite(a) & np.isfinite(b)
    if np.any(np.isfinite(a) ^ np.isfinite(b)):
        return float('inf'), 'nan/inf mismatch'
    if mask is not None:
        finite = finite & mask
    if not finite.any():
        return 0.0, 'empty mask'
    d = np.abs(a[finite] - b[finite])
    return float(d.max()), ''


def _max_circular_diff(a, b, period=360.0):
    a = np.asarray(a).astype(np.float64)
    b = np.asarray(b).astype(np.float64)
    if a.shape != b.shape:
        return float('inf'), f'shape mismatch {a.shape} vs {b.shape}'
    finite = np.isfinite(a) & np.isfinite(b)
    if np.any(np.isfinite(a) ^ np.isfinite(b)):
        return float('inf'), 'nan/inf mismatch'
    d = np.mod(a[finite] - b[finite] + period / 2, period) - period / 2
    if d.size == 0:
        return 0.0, ''
    return float(np.abs(d).max()), ''


def _report(name, py, mat, tol=1e-5, mask=None):
    err, info = _max_abs_diff(py, mat, mask=mask)
    status = 'OK' if err <= tol else ('WARN' if err <= max(tol * 100, 1e-2) else 'FAIL')
    extra = f' ({info})' if info else ''
    print(f'  [{status:4s}] {name:40s} max|diff| = {err:.3e}{extra}')
    return err <= tol


def _report_circular(name, py, mat, period=360.0, tol=1.0):
    err, info = _max_circular_diff(py, mat, period)
    status = 'OK' if err <= tol else ('WARN' if err <= 10 else 'FAIL')
    extra = f' ({info})' if info else ''
    print(f'  [{status:4s}] {name:40s} circ|diff|(p={int(period)}) = {err:.3e}{extra}')
    return err <= tol


def _report_boolean(name, py, mat, max_diff_pixels=10):
    py = np.asarray(py).astype(bool)
    mat = np.asarray(mat).astype(bool)
    if py.shape != mat.shape:
        print(f'  [FAIL] {name:40s} shape mismatch')
        return False
    n_diff = int((py != mat).sum())
    total = py.size
    status = 'OK' if n_diff <= max_diff_pixels else ('WARN' if n_diff <= 100 else 'FAIL')
    print(f'  [{status:4s}] {name:40s} diff pixels = {n_diff}/{total}')
    return n_diff <= max_diff_pixels


def _struct_to_dict(s):
    if hasattr(s, '_fieldnames'):
        return {f: _struct_to_dict(getattr(s, f)) for f in s._fieldnames}
    return s


def main():
    mat_path = os.path.join(_THIS_DIR, 'test_feature_detection.mat')
    if not os.path.exists(mat_path):
        print(f'ERROR: {mat_path} not found. Run generate_test_data.m in MATLAB first.')
        sys.exit(1)

    d = loadmat(mat_path, squeeze_me=True, struct_as_record=False)

    H = int(d['H']); W = int(d['W']); T = int(d['T'])
    print(f'Loaded test data: H={H} W={W} T={T}')
    print()

    passed = []

    print('--- pc_sumR_flux_noise_std ---')
    flux_noise_std_arr = d['flux_noise_std_arr']
    filt_energies = d['filt_energies']
    py = pc_sumR_flux_noise_std(flux_noise_std_arr, filt_energies[0, :])
    passed.append(_report('PC_noise_std', py, d['PC_noise_std']))

    print('--- phase_congruency_1D ---')
    R_1D_raw = d['R_1D']
    R_1D = [R_1D_raw[i] for i in range(R_1D_raw.shape[0])]
    py = phase_congruency_1D(R_1D,
                             weight_g=10.0,
                             cut_off=0.5,
                             noise_std_est=d['noise_std_est_1D'],
                             noise_thresh_zmin=2.0,
                             deviation_gain=1.5)
    passed.append(_report('PC_1D_out', py, d['PC_1D_out'], tol=1e-5))

    print('--- phase_congruency_3D_structure_tensor ---')
    R_3D_raw = d['R_3D']
    R_3D = [[R_3D_raw[i, j] for j in range(R_3D_raw.shape[1])]
            for i in range(R_3D_raw.shape[0])]
    dirs = d['dirs']
    Px2, Py2, Pt2, Pxy, Pyt, Pxt = phase_congruency_3D_structure_tensor(
        R_3D, dirs,
        flux_noise_std=flux_noise_std_arr,
        filter_energies=d['filter_energies_3D'],
        noise_thresh_zmin=2.0,
    )
    passed.append(_report('PC_x2', Px2, d['PC_x2'], tol=1e-4))
    passed.append(_report('PC_y2', Py2, d['PC_y2'], tol=1e-4))
    passed.append(_report('PC_t2', Pt2, d['PC_t2'], tol=1e-4))
    passed.append(_report('PC_xy', Pxy, d['PC_xy']))
    passed.append(_report('PC_yt', Pyt, d['PC_yt']))
    passed.append(_report('PC_xt', Pxt, d['PC_xt']))

    print('--- features_3D_structure_tensor ---')
    edges_py, corners_py, transients_py = features_3D_structure_tensor(
        d['Ix2'], d['Iy2'], d['It2'], d['Ixy'], d['Iyt'], d['Ixt'])
    edges_mat = _struct_to_dict(d['edges'])
    corners_mat = _struct_to_dict(d['corners'])
    transients_mat = _struct_to_dict(d['transients'])

    passed.append(_report('edges.strength', edges_py['strength'], edges_mat['strength'], tol=1e-3))
    passed.append(_report_circular('edges.orientation', edges_py['orientation'],
                                   edges_mat['orientation'], period=180.0, tol=1.0))
    passed.append(_report('edges.temporal_angle', edges_py['temporal_angle'],
                          edges_mat['temporal_angle'], tol=1.0))
    passed.append(_report('edges.normal_velocity', edges_py['normal_velocity'],
                          edges_mat['normal_velocity'], tol=1e-2))
    passed.append(_report('edges.coherence', edges_py['coherence'], edges_mat['coherence'], tol=1e-3))
    passed.append(_report('corners.strength', corners_py['strength'], corners_mat['strength'], tol=1e-3))
    passed.append(_report('corners.coherence', corners_py['coherence'], corners_mat['coherence'], tol=1e-3))

    v_py = np.asarray(corners_py['velocity'])
    v_mat = np.asarray(corners_mat['velocity'])
    sat_py = np.abs(v_py) > 1e4
    sat_mat = np.abs(v_mat) > 1e4
    both_ok = ~(sat_py | sat_mat)
    err_vel, _ = _max_abs_diff(v_py, v_mat, mask=both_ok)
    n_both_sat = int((sat_py & sat_mat).sum())
    n_only_py = int((sat_py & ~sat_mat).sum())
    n_only_mat = int((~sat_py & sat_mat).sum())
    n_ok = int(both_ok.sum())
    total = v_py.size
    print(f'  [INFO] corners.velocity                    (sign-convention sensitive)')
    print(f'         both saturated : {n_both_sat}/{total} (|v|>1e4 in both)')
    print(f'         only Python sat: {n_only_py}/{total}')
    print(f'         only MATLAB sat: {n_only_mat}/{total}')
    print(f'         both finite    : {n_ok}/{total}, max|diff| = {err_vel:.3e}')
    if n_only_py + n_only_mat < 0.05 * total and err_vel < 1e-2:
        print(f'         -> sign-convention appears to match eig_3x3_sym.m')
    else:
        print(f'         -> sign-convention differs; send eig_3x3_sym.m to match exactly')

    passed.append(_report('transients.strength', transients_py['strength'],
                          transients_mat['strength'], tol=1e-3))

    print('--- edge_props_3D ---')
    estr_ep, eori_ep, ephi_ep, evn_ep = edge_props_3D(d['ex'], d['ey'], d['et'])
    passed.append(_report('estr_ep', estr_ep, d['estr_ep']))
    passed.append(_report_circular('eori_ep', eori_ep, d['eori_ep'], period=360.0, tol=1.0))
    passed.append(_report('ephi_ep', ephi_ep, d['ephi_ep'], tol=1.0))
    passed.append(_report('evn_ep', evn_ep, d['evn_ep']))

    print('--- edge / corner coherence ---')
    py_ec = edge_coherence(d['lam1'], d['lam2'])
    py_cc = corner_coherence(d['lam2'], d['lam3'])
    passed.append(_report('edge_coherence', py_ec, d['ec_out']))
    passed.append(_report('corner_coherence', py_cc, d['cc_out']))

    print('--- smoothorient_3D ---')
    smo_py, smp_py = smoothorient_3D(d['orid_in'], d['phid_in'], float(d['sigma']))
    passed.append(_report_circular('smoothorient_3D.orid', smo_py, d['smorid3'],
                                   period=360.0, tol=1.0))
    passed.append(_report('smoothorient_3D.phid', smp_py, d['smphid3'], tol=1.0))

    print('--- smoothorient (2D) ---')
    smo2_py = smoothorient(d['orid2_in'], 1.5)
    passed.append(_report_circular('smoothorient 2D', smo2_py, d['smorid2'],
                                   period=180.0, tol=1.0))

    print('--- nms3 (linear in Py vs cubic in MATLAB: expect mild diffs) ---')
    I_thin_py = nms3(d['I_nms'], d['ori_nms'], d['phi_nms'], interp_method='linear')
    err, _ = _max_abs_diff(I_thin_py, d['I_thin_cubic'])
    mat_nz = np.count_nonzero(d['I_thin_cubic'])
    py_nz = np.count_nonzero(I_thin_py)
    print(f'  [INFO] nms3 nonzero count MATLAB={mat_nz}, Python(linear)={py_nz}, max|diff|={err:.3e}')

    print('--- corner_importance ---')
    imp_py = corner_importance(d['str_ci'], d['coh_ci'], d['vel_ci'],
                               coh_min=0.1, v_max=5.0, nms_win=3)
    passed.append(_report('corner_importance', imp_py, d['imp_out']))

    print('--- hysthresh_2 ---')
    emap_py = hysthresh_2(d['estr_h'], [0.6, 0.3], 0)
    passed.append(_report_boolean('hysthresh_2', emap_py, d['emap_out'], max_diff_pixels=10))

    print('--- postprocess_features_3D (edges.strength_nms uses linear interp) ---')
    edges_pp_py, corners_pp_py, transients_pp_py = postprocess_features_3D(
        dict(edges_py), dict(corners_py), dict(transients_py),
        edge_coh_min=0.05,
        edge_ori_sm_sigma=1.0,
        corner_coh_min=0.0,
        corner_v_max=None,
        corner_nms_win=3,
        nms_interp_method='linear',
    )
    edges_pp_mat = _struct_to_dict(d['edges_pp'])
    corners_pp_mat = _struct_to_dict(d['corners_pp'])
    passed.append(_report('edges_pp.coherence', edges_pp_py['coherence'],
                          edges_pp_mat['coherence'], tol=1e-3))
    err, _ = _max_abs_diff(edges_pp_py['strength_nms'], edges_pp_mat['strength_nms'])
    print(f'  [INFO] edges_pp.strength_nms max|diff|={err:.3e} (linear vs cubic interp)')
    passed.append(_report('corners_pp.importance', corners_pp_py['importance'],
                          corners_pp_mat['importance'], tol=1e-3))

    print('--- cat_t_features_3D / crop_t_features_3D ---')
    ec_py, cc_py, tc_py = cat_t_features_3D(edges_pp_py, corners_pp_py, transients_pp_py,
                                             dict(edges_pp_py), dict(corners_pp_py),
                                             dict(transients_pp_py))
    ec_mat = _struct_to_dict(d['ec'])
    passed.append(_report('cat.ec.strength', ec_py['strength'], ec_mat['strength'], tol=1e-3))
    passed.append(_report_circular('cat.ec.orientation', ec_py['orientation'],
                                   ec_mat['orientation'], period=180.0, tol=1.0))

    er_py, cr_py, tr_py = crop_t_features_3D(edges_pp_py, corners_pp_py, transients_pp_py,
                                              n0=1, n1=T - 1)
    er_mat = _struct_to_dict(d['er'])
    passed.append(_report('crop.er.strength', er_py['strength'], er_mat['strength'], tol=1e-3))

    print()
    n_pass = sum(passed)
    n_total = len(passed)
    print(f'Summary: {n_pass}/{n_total} tests within tolerance')
    if n_pass < n_total:
        sys.exit(1)


if __name__ == '__main__':
    main()