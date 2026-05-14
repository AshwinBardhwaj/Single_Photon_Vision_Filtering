import time
import os
import json
import numpy as np
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, binary_dilation

from sensor.sensor import ippd, binom_noise_std
from filtering.log_gabor_filter_bank import LogGaborBank3DSepT
from filtering.iir_filter_bank import IIRFilterBank3DSepT
from filtering.steerable_filter_bank import SteerableFilterBank3DSepT
from filtering.learned_filter_bank import LearnedFilterBank3DSepT
from filtering.dtcwt_filter_bank import DTCWTFilterBank3DSepT
from vision.feature_detection.phase_congruency import phase_congruency_3D_structure_tensor
from vision.feature_detection.structure_tensor import features_3D_structure_tensor
from vision.motion_estimation.flo_1d_to_uv import flo_1d_to_uv
from vision.motion_estimation.vel_fj1990_approx_withz import vel_fj1990_approx_withz
from visualize.visualize import vis_edge, vis_flo_dense
from utils.utils import rescale_prctile

from fileio.fileio import save_video_imageio as save_video


def pr_with_tolerance(pred, gt, tolerance=2):
    pred_total = int(pred.sum())
    gt_total = int(gt.sum())
    if pred_total == 0 or gt_total == 0:
        return 0.0, 0.0, 0.0
    if tolerance > 0:
        gt_d = binary_dilation(gt, iterations=tolerance)
        pred_d = binary_dilation(pred, iterations=tolerance)
    else:
        gt_d, pred_d = gt, pred
    tp_p = int(np.logical_and(pred, gt_d).sum())
    tp_r = int(np.logical_and(gt, pred_d).sum())
    precision = tp_p / pred_total
    recall = tp_r / gt_total
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(fscore)


def eval_edges_at_threshold(edge_strength, edges_gt, threshold, tolerance=2):
    pred = (edge_strength > threshold).astype(bool)
    T = pred.shape[-1]
    pf = np.zeros((T, 3), dtype=np.float64)
    for t in range(T):
        pf[t] = pr_with_tolerance(pred[:, :, t], edges_gt[:, :, t].astype(bool), tolerance)
    return {
        'threshold': float(threshold),
        'precision': float(pf[:, 0].mean()),
        'recall': float(pf[:, 1].mean()),
        'fscore': float(pf[:, 2].mean()),
        'pred': pred,
        'per_frame': pf,
    }


def sweep_eval_edges(edge_strength, edges_gt, percentiles=None, tolerance=2):
    if percentiles is None:
        percentiles = list(range(50, 100, 5))
    if (edge_strength > 0).sum() == 0:
        return None
    results = []
    for pct in percentiles:
        thresh = float(np.percentile(edge_strength, pct))
        r = eval_edges_at_threshold(edge_strength, edges_gt, thresh, tolerance)
        r['percentile'] = float(pct)
        results.append(r)
    fscores = [r['fscore'] for r in results]
    best_idx = int(np.argmax(fscores))
    return {'all': results, 'best_idx': best_idx, 'best': results[best_idx]}


def vis_compare_edges(B, pred, gt, n_skip, crop_bounds):
    out_imin, out_imax, out_jmin, out_jmax = crop_bounds
    Bc = B[out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip]
    pc = pred[out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip].astype(bool)
    gc = gt[out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip].astype(bool)

    Bg = (rescale_prctile(Bc) * 255).astype(np.uint8)
    R = Bg.copy()
    G = Bg.copy()
    Bl = Bg.copy()

    only_pred = pc & ~gc
    only_gt = gc & ~pc
    both = pc & gc

    R[only_pred] = 255
    G[only_pred] = 0
    Bl[only_pred] = 0

    R[only_gt] = 0
    G[only_gt] = 255
    Bl[only_gt] = 0

    R[both] = 255
    G[both] = 255
    Bl[both] = 0

    return np.stack([R, G, Bl], axis=2)


def execute_pipeline_sim(data_file, output_dir, filts_config, params,
                         crop_bounds=None, eval_config=None):
    if eval_config is None:
        eval_config = {'tolerance': 2, 'percentiles': list(range(50, 100, 5))}

    print(f"Loading data from {data_file}...")
    data = loadmat(data_file)
    B = data['B'].astype(np.float32)
    M = data['M'].item()
    edges_gt = data['edges_gt'].astype(np.uint8) if 'edges_gt' in data else None
    has_gt = edges_gt is not None

    if crop_bounds is None:
        out_imin, out_imax = 0, B.shape[0]
        out_jmin, out_jmax = 0, B.shape[1]
    else:
        out_imin, out_imax, out_jmin, out_jmax = crop_bounds

    n_skip = params['N_SKIP_OUTPUT']

    if M != 1:
        B = B * (1.0 / M)

    c_est = ippd(np.mean(B[out_imin:out_imax, out_jmin:out_jmax, :]))
    print(f"Flux level: {c_est * M:.4f} ppp.")

    t0 = time.time()
    local_flux = ippd(gaussian_filter(B, sigma=10))
    flux_noise_std = binom_noise_std(local_flux, M)
    print(f"Estimate local mean flux: {time.time() - t0:.3f} seconds.")

    t0 = time.time()
    H, W, N = B.shape
    filter_type = params.get('FILTER_TYPE', 'log_gabor')
    if filter_type == 'iir':
        filts = IIRFilterBank3DSepT()
    elif filter_type == 'steerable':
        filts = SteerableFilterBank3DSepT()
    elif filter_type == 'learned':
        filts = LearnedFilterBank3DSepT()
    elif filter_type == 'dtcwt':
        filts = DTCWTFilterBank3DSepT()
    else:
        filts = LogGaborBank3DSepT()
    for key, value in filts_config.items():
        setattr(filts, key, value)
    filts.input_size = (H, W, N)
    result = filts.set_up_filters()
    if result is not None:
        filts = result
    print(f"Filter initialization: {time.time() - t0:.3f} seconds.")

    t0 = time.time()
    R, Rz = filts.response(B, flux_noise_std)
    print(f"Get filter responses: {time.time() - t0:.3f} seconds.")

    t0 = time.time()
    num_dirs = filts.num_orientations * filts.num_velocities
    R_formatted = [
        [R.get((d, s), None) for s in range(filts.num_scales)]
        for d in range(num_dirs)
    ]
    pc_out = phase_congruency_3D_structure_tensor(
        R_formatted,
        filts.tuning_directions(),
        filter_energies=filts.filter_energies,
        flux_noise_std=flux_noise_std,
        noise_thresh_zmin=params['TPC_NOISE_THRESH_ZMIN']
    )
    PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt = pc_out
    edges, _, _ = features_3D_structure_tensor(PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt)
    print(f"edge_detection_TPC: {time.time() - t0:.3f} seconds.")

    t0 = time.time()
    flo1 = flo_1d_to_uv(edges['normal_velocity'], edges['orientation'])
    strength_thresh = np.percentile(edges['strength'], 75)
    rel_flo1 = (edges['strength'] > strength_thresh) & \
               (np.abs(edges['normal_velocity']) < params['MAX_FLO']) & \
               (edges['coherence'] > 0.5)
    print(f"edge_velocities_TPC: {time.time() - t0:.3f} seconds.")

    t0 = time.time()
    if Rz:
        Rz_dict = Rz
    else:
        print("Note: Rz is empty, falling back to R dictionary for FJ1990 optical flow.")
        Rz_dict = R

    class RzDictWrapper:
        def __init__(self, d, nd, ns):
            self.d = d
            self.nd = nd
            self.ns = ns
        def __len__(self):
            return self.nd
        def __getitem__(self, idx):
            if idx == 0:
                return [None] * self.ns
            raise KeyError(idx)
        def get(self, key, default=None):
            return self.d.get(key, default)

    Rz_wrapped = RzDictWrapper(Rz_dict, num_dirs, filts.num_scales)
    _, _, flo2_FJ_ms, rel2_FJ_ms = vel_fj1990_approx_withz(
        Rz_wrapped,
        filts.tuning_directions(),
        solve_2d_pxwise_k=params['FJ_SOLVE_2D_PXWISE_K']
    )
    print(f"vel_FJ1990_approx_withz_multiscale: {time.time() - t0:.3f} seconds.")

    B_crop = B[out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip]
    estr = edges['strength']
    estr_crop = estr[out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip]
    flo1_crop = flo1[out_imin:out_imax, out_jmin:out_jmax, :, n_skip:-n_skip]
    rel_flo1_crop = rel_flo1[out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip]

    flo2_FJ_ms_crop = []
    rel2_FJ_ms_crop = []
    for s in range(len(flo2_FJ_ms)):
        flo2_FJ_ms_crop.append(flo2_FJ_ms[s][out_imin:out_imax, out_jmin:out_jmax, :, n_skip:-n_skip])
        rel2_FJ_ms_crop.append(rel2_FJ_ms[s][out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip])

    os.makedirs(output_dir, exist_ok=True)

    B_v = rescale_prctile(B_crop)
    save_video(np.moveaxis(B_v, -1, 0), os.path.join(output_dir, 'B.mp4'))

    estr_v = vis_edge(estr_crop, False, params['ESTR_VIS_PRCTILE_THRESH'])
    save_video(np.moveaxis(estr_v, -1, 0), os.path.join(output_dir, 'estr_TPC.mp4'))

    flo1_v = vis_flo_dense(flo1_crop, rel_flo1_crop, params['MAX_FLO'], params['FLO1_VIS_THICKEN'])
    save_video(np.moveaxis(flo1_v, -1, 0), os.path.join(output_dir, 'flo1_TPC.mp4'))

    for s in range(len(flo2_FJ_ms_crop)):
        rel_mask = rel2_FJ_ms_crop[s] > params['FLO2_REL_THRESH']
        flo2_FJ_ms_c = vis_flo_dense(flo2_FJ_ms_crop[s], rel_mask, params['MAX_FLO'])
        save_video(np.moveaxis(flo2_FJ_ms_c, -1, 0),
                   os.path.join(output_dir, f'flo2_FJ_ms_{s + 1}.mp4'))

    if has_gt:
        t0 = time.time()
        sweep = sweep_eval_edges(estr, edges_gt,
                                 percentiles=eval_config['percentiles'],
                                 tolerance=eval_config['tolerance'])
        print(f"Edge evaluation sweep: {time.time() - t0:.3f} seconds.")

        if sweep is not None:
            best = sweep['best']
            print(f"Best F={best['fscore']:.4f}  P={best['precision']:.4f}  "
                  f"R={best['recall']:.4f}  at percentile={best['percentile']:.0f}  "
                  f"(threshold={best['threshold']:.4g}, tolerance={eval_config['tolerance']}px)")

            metrics_summary = {
                'best_fscore': best['fscore'],
                'best_precision': best['precision'],
                'best_recall': best['recall'],
                'best_percentile': best['percentile'],
                'best_threshold': best['threshold'],
                'tolerance_px': eval_config['tolerance'],
                'sweep': [
                    {'percentile': r['percentile'], 'threshold': r['threshold'],
                     'precision': r['precision'], 'recall': r['recall'],
                     'fscore': r['fscore']}
                    for r in sweep['all']
                ],
            }
            with open(os.path.join(output_dir, 'edge_metrics.json'), 'w') as f:
                json.dump(metrics_summary, f, indent=2)

            gt_crop = edges_gt[out_imin:out_imax, out_jmin:out_jmax, n_skip:-n_skip]
            save_video(np.moveaxis(gt_crop.astype(np.float32), -1, 0),
                       os.path.join(output_dir, 'edges_gt.mp4'))

            cmp_v = vis_compare_edges(B, best['pred'], edges_gt, n_skip,
                                      (out_imin, out_imax, out_jmin, out_jmax))
            save_video(np.moveaxis(cmp_v, -1, 0),
                       os.path.join(output_dir, 'edges_compare.mp4'))

            print("Wrote edges_gt.mp4, edges_compare.mp4 (red=TPC only, "
                  "green=GT only, yellow=both), edge_metrics.json")
    else:
        print("No edges_gt found in data file, skipping evaluation.")

    print(f"Finished processing and saving to {output_dir}")