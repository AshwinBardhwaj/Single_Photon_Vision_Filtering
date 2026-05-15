import os
import sys
import time
import tracemalloc
import numpy as np
from pathlib import Path
from scipy.io import loadmat

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from filtering.log_gabor_filter_bank import LogGaborBank3DSepT
from filtering.adaptive_iir_filter_bank import AdaptiveIIRFilterBank3DSepT
from fileio.fileio import save_video_imageio
from vision.feature_detection.phase_congruency import phase_congruency_3D_structure_tensor
from vision.feature_detection.structure_tensor import features_3D_structure_tensor
from visualize.visualize import vis_edge
from sensor.sensor import ippd, binom_noise_std
from scipy.ndimage import gaussian_filter
from skimage.metrics import structural_similarity as ssim

def calculate_ssim(img1, img2, data_range=1.0):
    return ssim(img1, img2, data_range=data_range)

from scipy.ndimage import binary_dilation

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
    precision = tp_p / pred_total if pred_total > 0 else 0.0
    recall = tp_r / gt_total if gt_total > 0 else 0.0
    fscore = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return float(precision), float(recall), float(fscore)

def sweep_eval_edges_relative(pred_strength, gt_strength, percentiles=None, tolerance=2):
    if percentiles is None:
        percentiles = list(range(50, 100, 5))
    if (pred_strength > 0).sum() == 0 or (gt_strength > 0).sum() == 0:
        return None
    results = []
    for pct in percentiles:
        gt_thresh = float(np.percentile(gt_strength, pct))
        pred_thresh = float(np.percentile(pred_strength, pct))
        gt_mask = (gt_strength > gt_thresh).astype(bool)
        pred_mask = (pred_strength > pred_thresh).astype(bool)
        T = pred_mask.shape[-1]
        pf = np.zeros((T, 3), dtype=np.float64)
        for t in range(T):
            pf[t] = pr_with_tolerance(pred_mask[:, :, t], gt_mask[:, :, t], tolerance)
        r = {
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
        edge_time = 0.0
    else:
        R, Rz = filts.response(B_data, flux_noise_std=flux_noise_std)
        runtime = time.time() - start_time

        rec_time_start = time.time()
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
        edge_time = time.time() - rec_time_start

    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    peak_mem_mb = peak_mem / (1024 * 1024)
    print(f"[{name}] Runtime (Filter): {runtime:.2f} seconds")
    print(f"[{name}] Edge Detection Time: {edge_time:.2f} seconds")
    print(f"[{name}] Peak Memory: {peak_mem_mb:.2f} MB")

    r_min, r_max = edge_strength.min(), edge_strength.max()
    edge_strength_norm = (edge_strength - r_min) / (r_max - r_min + 1e-8)

    return {
        'runtime': runtime,
        'peak_mem_mb': peak_mem_mb,
        'edge_strength': edge_strength,
        'edge_strength_norm': edge_strength_norm
    }

if __name__ == "__main__":
    DATA_BASEDIR = root_dir / "scripts" / "data"
    data_files = list(DATA_BASEDIR.glob("*.mat"))

    if not data_files:
        print(f"No .mat files found in {DATA_BASEDIR}")
        sys.exit(0)

    print(f"Found {len(data_files)} real data files for evaluation.")
    results = []

    output_base_dir = root_dir / "scripts" / "output" / "reconstruction"
    os.makedirs(output_base_dir, exist_ok=True)

    for data_file in data_files:
        print(f"\n==================================================")
        print(f"Processing {data_file.name}")
        print(f"==================================================")

        try:
            data = loadmat(data_file)
            B = data['B'].astype(np.float32)
            M = float(data.get('M', 1.0))

            if M != 1:
                B = B * (1.0 / M)

            print(f"Data shape: {B.shape}, M={M}")

            local_flux = ippd(gaussian_filter(B, sigma=10))
            flux_noise_std = binom_noise_std(local_flux, M)

            data_out_dir = output_base_dir / data_file.stem
            lg_out_dir = data_out_dir / "log_gabor"
            iir_out_dir = data_out_dir / "adaptive_iir"

            os.makedirs(lg_out_dir, exist_ok=True)
            os.makedirs(iir_out_dir, exist_ok=True)

            lg_results = run_evaluation(LogGaborBank3DSepT, B, "Log Gabor", flux_noise_std)
            iir_results = run_evaluation(AdaptiveIIRFilterBank3DSepT, B, "Adaptive IIR", flux_noise_std)

            N_SKIP = 15
            H, W, N = B.shape

            if N > 2 * N_SKIP:
                lg_estr_crop = lg_results['edge_strength'][:, :, N_SKIP:-N_SKIP]
                iir_estr_crop = iir_results['edge_strength'][:, :, N_SKIP:-N_SKIP]
                B_crop = B[:, :, N_SKIP:-N_SKIP]
            else:
                lg_estr_crop = lg_results['edge_strength']
                iir_estr_crop = iir_results['edge_strength']
                B_crop = B

            lg_vid = vis_edge(lg_estr_crop, thicken=False, prctile_lim=[75, 97])
            lg_vid = (lg_vid * 255).astype(np.uint8) if lg_vid.dtype != np.uint8 else lg_vid
            save_video_imageio(np.moveaxis(lg_vid, -1, 0), str(lg_out_dir / "edge_detection.mp4"))

            iir_vid = vis_edge(iir_estr_crop, thicken=False, prctile_lim=[75, 97])
            iir_vid = (iir_vid * 255).astype(np.uint8) if iir_vid.dtype != np.uint8 else iir_vid
            save_video_imageio(np.moveaxis(iir_vid, -1, 0), str(iir_out_dir / "edge_detection.mp4"))

            b_min, b_max = B_crop.min(), B_crop.max()
            B_norm = (B_crop - b_min) / (b_max - b_min + 1e-8)
            b_vid = (B_norm * 255).astype(np.uint8)
            save_video_imageio(np.moveaxis(b_vid, -1, 0), str(data_out_dir / "original.mp4"))

            lg_norm_crop = lg_results['edge_strength_norm'][:, :, N_SKIP:-N_SKIP] if N > 2 * N_SKIP else lg_results['edge_strength_norm']
            iir_norm_crop = iir_results['edge_strength_norm'][:, :, N_SKIP:-N_SKIP] if N > 2 * N_SKIP else iir_results['edge_strength_norm']
            ssim_between = calculate_ssim(lg_norm_crop, iir_norm_crop, data_range=1.0)
            print(f"SSIM (Log Gabor vs Adaptive IIR): {ssim_between:.4f}")

            PERCENTILES = list(range(50, 100, 5))
            sweep_res = sweep_eval_edges_relative(iir_estr_crop, lg_estr_crop, PERCENTILES, tolerance=2)
            if sweep_res is not None:
                best_f = sweep_res['best']['fscore']
                best_p = sweep_res['best']['precision']
                best_r = sweep_res['best']['recall']
                best_pct = sweep_res['best']['percentile']
                print(f"Relative F-score (IIR vs LG pseudo-GT): {best_f:.4f} (P: {best_p:.4f}, R: {best_r:.4f} @ pct {best_pct:.0f})")
            else:
                best_f = 0.0

            metrics_txt = data_out_dir / "metrics.txt"
            with open(metrics_txt, "w") as f:
                f.write(f"Metrics for {data_file.name}\n")
                f.write("========================================\n")
                f.write(f"Log Gabor Runtime: {lg_results['runtime']:.2f} s\n")
                f.write(f"Log Gabor Peak Memory: {lg_results['peak_mem_mb']:.2f} MB\n")
                f.write(f"Adaptive IIR Runtime: {iir_results['runtime']:.2f} s\n")
                f.write(f"Adaptive IIR Peak Memory: {iir_results['peak_mem_mb']:.2f} MB\n")
                f.write(f"SSIM (Log Gabor vs Adaptive IIR): {ssim_between:.4f}\n")
                f.write(f"Relative F-score (tolerance=2px): {best_f:.4f}\n")

            results.append({
                'file': data_file.name,
                'lg_runtime': lg_results['runtime'],
                'lg_mem': lg_results['peak_mem_mb'],
                'iir_runtime': iir_results['runtime'],
                'iir_mem': iir_results['peak_mem_mb'],
                'ssim_between': ssim_between,
                'fscore': best_f
            })

        except Exception as e:
            print(f"Error processing {data_file.name}: {e}")

    if results:
        print("\n\n==================================================")
        print("FINAL AVERAGE SUMMARY ACROSS REAL DATA")
        print("==================================================")
        print(f"Total Files Processed: {len(results)}")

        avg_lg_rt = np.mean([r['lg_runtime'] for r in results])
        avg_iir_rt = np.mean([r['iir_runtime'] for r in results])
        avg_lg_mem = np.mean([r['lg_mem'] for r in results])
        avg_iir_mem = np.mean([r['iir_mem'] for r in results])
        avg_between_ssim = np.mean([r['ssim_between'] for r in results])
        avg_fscore = np.mean([r['fscore'] for r in results])

        print(f"\n{'Metric':<25} | {'Log Gabor':<15} | {'Adaptive IIR':<15}")
        print("-" * 61)
        print(f"{'Avg Runtime (s)':<25} | {avg_lg_rt:<15.2f} | {avg_iir_rt:<15.2f}")
        print(f"{'Avg Peak Memory (MB)':<25} | {avg_lg_mem:<15.2f} | {avg_iir_mem:<15.2f}")
        print("==================================================")
        print(f"Avg SSIM (Log Gabor vs Adaptive IIR): {avg_between_ssim:.4f}")
        print(f"Avg Relative F-score (tolerance=2px): {avg_fscore:.4f}")
        print("==================================================")
