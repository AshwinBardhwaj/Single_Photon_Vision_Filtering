import os
import sys
import time
import json
import tracemalloc
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, binary_dilation

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from filtering.log_gabor_filter_bank import LogGaborBank3DSepT
from filtering.adaptive_iir_filter_bank import AdaptiveIIRFilterBank3DSepT
from fileio.fileio import save_video_imageio
from vision.feature_detection.phase_congruency import phase_congruency_3D_structure_tensor
from vision.feature_detection.structure_tensor import features_3D_structure_tensor
from visualize.visualize import vis_edge
from sensor.sensor import binom_noise_std, ippd
from skimage.metrics import structural_similarity as ssim


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


def sweep_eval_edges(edge_strength, edges_gt, percentiles=None, tolerance=2):
    if percentiles is None:
        percentiles = list(range(50, 100, 5))
    if (edge_strength > 0).sum() == 0:
        return None
    results = []
    for pct in percentiles:
        thresh = float(np.percentile(edge_strength, pct))
        pred = (edge_strength > thresh).astype(bool)
        T = pred.shape[-1]
        pf = np.zeros((T, 3), dtype=np.float64)
        for t in range(T):
            pf[t] = pr_with_tolerance(pred[:, :, t], edges_gt[:, :, t].astype(bool), tolerance)
        r = {
            'threshold': float(thresh),
            'percentile': float(pct),
            'precision': float(pf[:, 0].mean()),
            'recall': float(pf[:, 1].mean()),
            'fscore': float(pf[:, 2].mean()),
        }
        results.append(r)
    fscores = [r['fscore'] for r in results]
    best_idx = int(np.argmax(fscores))
    return {'all': results, 'best_idx': best_idx, 'best': results[best_idx]}


FILTS_CONFIG = {
    'velocities': np.array([1, -1, 0.3, -0.3, 0]),
    'orientations': np.array([0, 60, 120]),
    'min_wavelength': 3,
    'num_scales': 3,
}

def run_evaluation(filter_class, B_data, name, flux_noise_std=None):
    print(f"\n--- Evaluating {name} ---")

    H, W, N = B_data.shape
    filts = filter_class()
    for key, value in FILTS_CONFIG.items():
        setattr(filts, key, value)
    filts.input_size = (H, W, N)

    tracemalloc.start()
    start_time = time.time()

    result = filts.set_up_filters()
    if result is not None:
        filts = result

    if hasattr(filts, 'response_and_edges'):
        edges = filts.response_and_edges(B_data, flux_noise_std=flux_noise_std)
        edge_strength = edges['strength']
        runtime = time.time() - start_time
    else:
        R, Rz = filts.response(B_data, flux_noise_std=flux_noise_std)
        runtime = time.time() - start_time

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
            noise_thresh_zmin=2
        )
        PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt = pc_out
        edges, _, _ = features_3D_structure_tensor(PC_x2, PC_y2, PC_t2, PC_xy, PC_yt, PC_xt)
        edge_strength = edges['strength']

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mem_mb = peak_mem / (1024 * 1024)
    print(f"[{name}] Runtime: {runtime:.2f} s, Peak Memory: {peak_mem_mb:.2f} MB")

    r_min, r_max = edge_strength.min(), edge_strength.max()
    edge_strength_norm = (edge_strength - r_min) / (r_max - r_min + 1e-8)

    return {
        'runtime': runtime,
        'peak_mem_mb': peak_mem_mb,
        'edge_strength': edge_strength,
        'edge_strength_norm': edge_strength_norm,
    }


if __name__ == "__main__":
    SIM_DATA_DIR = root_dir / "scripts" / "data" / "sim_xvfi_1bit"
    OUTPUT_BASE = root_dir / "scripts" / "output" / "reconstruction_sim"

    PPP_LEVELS = [0.50, 1.25, 2.00]
    MAX_CLIPS_PER_FOLDER = 3
    TOLERANCE = 2
    PERCENTILES = list(range(50, 100, 5))

    parent_dirs = sorted([
        d for d in SIM_DATA_DIR.iterdir()
        if d.is_dir() and d.name != '.DS_Store'
    ])

    all_results = []

    for parent_dir in parent_dirs:
        clip_dirs = sorted([
            d for d in parent_dir.iterdir()
            if d.is_dir() and d.name != '.DS_Store'
        ])[:MAX_CLIPS_PER_FOLDER]

        for clip_dir in clip_dirs:
            for ppp in PPP_LEVELS:
                data_file = clip_dir / f"ppp_{ppp:.2f}_data.mat"
                if not data_file.exists():
                    continue

                clip_id = f"{parent_dir.name}/{clip_dir.name}/ppp_{ppp:.2f}"
                print(f"\n{'='*60}")
                print(f"Processing {clip_id}")
                print(f"{'='*60}")

                try:
                    data = loadmat(str(data_file))
                    B = data['B'].astype(np.float32)
                    M = data['M'].item()
                    edges_gt = data['edges_gt'].astype(np.uint8)

                    if M != 1:
                        B = B * (1.0 / M)

                    print(f"Data shape: {B.shape}, M={M}, ppp={ppp}")

                    flux_est = ippd(gaussian_filter(B, sigma=10))
                    flux_noise_std = binom_noise_std(flux_est, M)

                    out_dir = OUTPUT_BASE / parent_dir.name / clip_dir.name / f"ppp_{ppp:.2f}"
                    os.makedirs(out_dir, exist_ok=True)

                    lg_res = run_evaluation(LogGaborBank3DSepT, B, "Log Gabor", flux_noise_std)
                    iir_res = run_evaluation(AdaptiveIIRFilterBank3DSepT, B, "Adaptive IIR", flux_noise_std)

                    N_SKIP = 15
                    H, W, N = B.shape

                    if N > 2 * N_SKIP:
                        lg_estr_crop = lg_res['edge_strength'][:, :, N_SKIP:-N_SKIP]
                        lg_norm_crop = lg_res['edge_strength_norm'][:, :, N_SKIP:-N_SKIP]
                        iir_estr_crop = iir_res['edge_strength'][:, :, N_SKIP:-N_SKIP]
                        iir_norm_crop = iir_res['edge_strength_norm'][:, :, N_SKIP:-N_SKIP]
                        edges_gt_crop = edges_gt[:, :, N_SKIP:-N_SKIP]
                    else:
                        lg_estr_crop = lg_res['edge_strength']
                        lg_norm_crop = lg_res['edge_strength_norm']
                        iir_estr_crop = iir_res['edge_strength']
                        iir_norm_crop = iir_res['edge_strength_norm']
                        edges_gt_crop = edges_gt

                    ssim_between = ssim(lg_norm_crop, iir_norm_crop, data_range=1.0)
                    print(f"SSIM (LG vs IIR): {ssim_between:.4f}")

                    lg_sweep = sweep_eval_edges(lg_estr_crop, edges_gt_crop, PERCENTILES, TOLERANCE)
                    iir_sweep = sweep_eval_edges(iir_estr_crop, edges_gt_crop, PERCENTILES, TOLERANCE)

                    lg_best = lg_sweep['best'] if lg_sweep else {'fscore': 0, 'precision': 0, 'recall': 0, 'percentile': 0}
                    iir_best = iir_sweep['best'] if iir_sweep else {'fscore': 0, 'precision': 0, 'recall': 0, 'percentile': 0}

                    print(f"[Log Gabor]     Best F={lg_best['fscore']:.4f}  P={lg_best['precision']:.4f}  R={lg_best['recall']:.4f}  @pct={lg_best['percentile']:.0f}")
                    print(f"[Adaptive IIR]  Best F={iir_best['fscore']:.4f}  P={iir_best['precision']:.4f}  R={iir_best['recall']:.4f}  @pct={iir_best['percentile']:.0f}")

                    for label, estr_crop in [("log_gabor", lg_estr_crop), ("adaptive_iir", iir_estr_crop)]:
                        d = out_dir / label
                        os.makedirs(d, exist_ok=True)
                        ev = vis_edge(estr_crop, thicken=False, prctile_lim=[75, 97])
                        ev = (ev * 255).astype(np.uint8) if ev.dtype != np.uint8 else ev
                        save_video_imageio(np.moveaxis(ev, -1, 0), str(d / "edge_detection.mp4"))

                    metrics = {
                        'clip': clip_id,
                        'data_shape': list(B.shape),
                        'ppp': ppp,
                        'ssim_between': ssim_between,
                        'log_gabor': {
                            'runtime': lg_res['runtime'],
                            'peak_mem_mb': lg_res['peak_mem_mb'],
                            'best_fscore': lg_best['fscore'],
                            'best_precision': lg_best['precision'],
                            'best_recall': lg_best['recall'],
                            'best_percentile': lg_best.get('percentile', 0),
                        },
                        'adaptive_iir': {
                            'runtime': iir_res['runtime'],
                            'peak_mem_mb': iir_res['peak_mem_mb'],
                            'best_fscore': iir_best['fscore'],
                            'best_precision': iir_best['precision'],
                            'best_recall': iir_best['recall'],
                            'best_percentile': iir_best.get('percentile', 0),
                        },
                    }
                    with open(out_dir / "metrics.json", "w") as f:
                        json.dump(metrics, f, indent=2)

                    all_results.append(metrics)

                except Exception as e:
                    print(f"Error processing {clip_id}: {e}")
                    import traceback
                    traceback.print_exc()

    if all_results:
        print(f"\n\n{'='*70}")
        print("FINAL SUMMARY ACROSS SIMULATED DATA")
        print(f"{'='*70}")
        print(f"Total Clips Processed: {len(all_results)}")

        for ppp in PPP_LEVELS:
            ppp_results = [r for r in all_results if r['ppp'] == ppp]
            if not ppp_results:
                continue

            avg_lg_f = np.mean([r['log_gabor']['best_fscore'] for r in ppp_results])
            avg_iir_f = np.mean([r['adaptive_iir']['best_fscore'] for r in ppp_results])
            avg_lg_p = np.mean([r['log_gabor']['best_precision'] for r in ppp_results])
            avg_iir_p = np.mean([r['adaptive_iir']['best_precision'] for r in ppp_results])
            avg_lg_r = np.mean([r['log_gabor']['best_recall'] for r in ppp_results])
            avg_iir_r = np.mean([r['adaptive_iir']['best_recall'] for r in ppp_results])
            avg_lg_rt = np.mean([r['log_gabor']['runtime'] for r in ppp_results])
            avg_iir_rt = np.mean([r['adaptive_iir']['runtime'] for r in ppp_results])
            avg_lg_mem = np.mean([r['log_gabor']['peak_mem_mb'] for r in ppp_results])
            avg_iir_mem = np.mean([r['adaptive_iir']['peak_mem_mb'] for r in ppp_results])
            avg_ssim = np.mean([r['ssim_between'] for r in ppp_results])

            print(f"\n--- PPP = {ppp:.2f} ({len(ppp_results)} clips) ---")
            print(f"{'Metric':<25} | {'Log Gabor':<15} | {'Adaptive IIR':<15}")
            print("-" * 61)
            print(f"{'Avg Best F-score':<25} | {avg_lg_f:<15.4f} | {avg_iir_f:<15.4f}")
            print(f"{'Avg Precision':<25} | {avg_lg_p:<15.4f} | {avg_iir_p:<15.4f}")
            print(f"{'Avg Recall':<25} | {avg_lg_r:<15.4f} | {avg_iir_r:<15.4f}")
            print(f"{'Avg Runtime (s)':<25} | {avg_lg_rt:<15.2f} | {avg_iir_rt:<15.2f}")
            print(f"{'Avg Peak Memory (MB)':<25} | {avg_lg_mem:<15.2f} | {avg_iir_mem:<15.2f}")
            print(f"{'Avg SSIM (LG vs IIR)':<25} | {avg_ssim:<15.4f}")

        print(f"\n--- OVERALL ({len(all_results)} clips) ---")
        print(f"{'Metric':<25} | {'Log Gabor':<15} | {'Adaptive IIR':<15}")
        print("-" * 61)
        avg_lg_f = np.mean([r['log_gabor']['best_fscore'] for r in all_results])
        avg_iir_f = np.mean([r['adaptive_iir']['best_fscore'] for r in all_results])
        avg_lg_rt = np.mean([r['log_gabor']['runtime'] for r in all_results])
        avg_iir_rt = np.mean([r['adaptive_iir']['runtime'] for r in all_results])
        avg_lg_mem = np.mean([r['log_gabor']['peak_mem_mb'] for r in all_results])
        avg_iir_mem = np.mean([r['adaptive_iir']['peak_mem_mb'] for r in all_results])
        avg_ssim = np.mean([r['ssim_between'] for r in all_results])
        print(f"{'Avg Best F-score':<25} | {avg_lg_f:<15.4f} | {avg_iir_f:<15.4f}")
        print(f"{'Avg Runtime (s)':<25} | {avg_lg_rt:<15.2f} | {avg_iir_rt:<15.2f}")
        print(f"{'Avg Peak Memory (MB)':<25} | {avg_lg_mem:<15.2f} | {avg_iir_mem:<15.2f}")
        print(f"{'Avg SSIM (LG vs IIR)':<25} | {avg_ssim:<15.4f}")
        print("=" * 70)

        summary_path = OUTPUT_BASE / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nFull results saved to {summary_path}")
