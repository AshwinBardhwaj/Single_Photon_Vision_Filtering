import os
import sys
import time
import json
import argparse
import tracemalloc
import numpy as np
from pathlib import Path
from scipy.io import loadmat
from scipy.ndimage import gaussian_filter, binary_dilation

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))

from filtering.log_gabor_filter_bank import LogGaborBank3DSepT
from filtering.adaptive_iir_filter_bank import AdaptiveIIRFilterBank3DSepT, BidirAdaptiveIIRFilterBank3DSepT
from filtering.steerable_filter_bank import SteerableFilterBank3DSepT
from filtering.gradient_steerable_filter_bank import GradientSteerableFilterBank3DSepT
from filtering.dtcwt_filter_bank import DTCWTFilterBank3DSepT
from fileio.fileio import save_video_imageio
from vision.feature_detection.phase_congruency import phase_congruency_3D_structure_tensor
from vision.feature_detection.structure_tensor import features_3D_structure_tensor
from vision.motion_estimation.flo_1d_to_uv import flo_1d_to_uv
from vision.motion_estimation.vel_fj1990_approx_withz import vel_fj1990_approx_withz
from visualize.visualize import vis_edge, vis_flo_dense
from sensor.sensor import binom_noise_std, ippd

FILTS_CONFIG = {
    'velocities':     np.array([1, -1, 0.3, -0.3, 0]),
    'orientations':   np.array([0, 60, 120]),
    'min_wavelength': 3,
    'num_scales':     3,
}

FLOW_PARAMS = {
    'MAX_FLO':         1.0,
    'FLO1_STR_PCTILE': 75,
    'FLO1_COH_THRESH': 0.5,
    'FLO1_THICKEN':    3,
    'FJ_K':            0.1,
    'FLO2_REL_THRESH': 0.5,
}

N_SKIP      = 15
TOLERANCE   = 2
PERCENTILES = list(range(50, 100, 5))

REAL_DATA_DIR = root_dir / "scripts" / "data"
SIM_DATA_DIR  = root_dir / "scripts" / "data" / "sim_xvfi_1bit"
OUTPUT_BASE   = root_dir / "scripts" / "output" / "full_comparison"

PPP_LEVELS          = [0.50, 1.25, 2.00]
MAX_CLIPS_PER_SCENE = 3

METHODS = [
    ("Log Gabor",          LogGaborBank3DSepT),
    ("Adaptive IIR",       AdaptiveIIRFilterBank3DSepT),
    ("Adaptive IIR BD",    BidirAdaptiveIIRFilterBank3DSepT),
    ("Steerable",          SteerableFilterBank3DSepT),
    ("Grad Steerable",     GradientSteerableFilterBank3DSepT),
    ("DT-CWT",             DTCWTFilterBank3DSepT),
]
METHOD_NAMES = [m[0] for m in METHODS]
REF_METHOD   = "Log Gabor"


class _RzWrapper:
    def __init__(self, d, num_dirs, num_scales):
        self._d = d
        self._nd = num_dirs
        self._ns = num_scales

    def __len__(self):
        return self._nd

    def __getitem__(self, idx):
        if idx == 0:
            return [None] * self._ns
        raise KeyError(idx)

    def get(self, key, default=None):
        return self._d.get(key, default)


def pr_with_tolerance(pred, gt, tolerance=2):
    pred_total = int(pred.sum())
    gt_total   = int(gt.sum())
    if pred_total == 0 or gt_total == 0:
        return 0.0, 0.0, 0.0
    gt_d   = binary_dilation(gt,   iterations=tolerance) if tolerance > 0 else gt
    pred_d = binary_dilation(pred, iterations=tolerance) if tolerance > 0 else pred
    tp_p = int(np.logical_and(pred, gt_d).sum())
    tp_r = int(np.logical_and(gt,   pred_d).sum())
    p = tp_p / pred_total
    r = tp_r / gt_total
    f = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return float(p), float(r), float(f)


def sweep_fscore(pred_strength, gt_mask_3d, percentiles=PERCENTILES, tolerance=TOLERANCE):
    if (pred_strength > 0).sum() == 0:
        return {'fscore': 0.0, 'precision': 0.0, 'recall': 0.0, 'percentile': 0.0}
    best = {'fscore': -1}
    for pct in percentiles:
        thresh = float(np.percentile(pred_strength, pct))
        pred   = (pred_strength > thresh).astype(bool)
        T      = pred.shape[-1]
        pf     = np.array([pr_with_tolerance(pred[:, :, t],
                                              gt_mask_3d[:, :, t].astype(bool),
                                              tolerance)
                            for t in range(T)])
        f = float(pf[:, 2].mean())
        if f > best['fscore']:
            best = {'fscore': f,
                    'precision': float(pf[:, 0].mean()),
                    'recall':    float(pf[:, 1].mean()),
                    'percentile': float(pct)}
    return best


def sweep_fscore_relative(pred_strength, ref_strength,
                           percentiles=PERCENTILES, tolerance=TOLERANCE):
    if (pred_strength > 0).sum() == 0 or (ref_strength > 0).sum() == 0:
        return {'fscore': 0.0, 'precision': 0.0, 'recall': 0.0, 'percentile': 0.0}
    best = {'fscore': -1}
    for pct in percentiles:
        gt_thresh   = float(np.percentile(ref_strength,  pct))
        pred_thresh = float(np.percentile(pred_strength, pct))
        gt_mask   = (ref_strength  > gt_thresh).astype(bool)
        pred_mask = (pred_strength > pred_thresh).astype(bool)
        T  = pred_mask.shape[-1]
        pf = np.array([pr_with_tolerance(pred_mask[:, :, t],
                                          gt_mask[:, :, t],
                                          tolerance)
                        for t in range(T)])
        f = float(pf[:, 2].mean())
        if f > best['fscore']:
            best = {'fscore': f,
                    'precision': float(pf[:, 0].mean()),
                    'recall':    float(pf[:, 1].mean()),
                    'percentile': float(pct)}
    return best


def flow_coverage(rel_mask):
    return float(rel_mask.mean())


def flow_epe(flo_pred, flo_ref, rel_pred, rel_ref):
    mask = rel_pred & rel_ref
    if mask.sum() == 0:
        return float('nan')
    du  = flo_pred[:, :, 0, :] - flo_ref[:, :, 0, :]
    dv  = flo_pred[:, :, 1, :] - flo_ref[:, :, 1, :]
    epe = np.sqrt(du ** 2 + dv ** 2)
    return float(epe[mask].mean())


def run_pipeline(filter_class, B, name, flux_noise_std):
    print(f"    [{name}] filtering...", end=" ", flush=True)
    H, W, N = B.shape
    filts = filter_class()
    for k, v in FILTS_CONFIG.items():
        setattr(filts, k, v)
    filts.input_size = (H, W, N)

    result = filts.set_up_filters()
    if result is not None:
        filts = result

    num_dirs = filts.num_orientations * filts.num_velocities
    t_start = time.time()

    if hasattr(filts, 'response_and_edges'):
        tracemalloc.start()
        edges = filts.response_and_edges(B, flux_noise_std=flux_noise_std)
        _, peak_mem_edges = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        tracemalloc.start()
        R, Rz = filts.response(B, flux_noise_std=flux_noise_std)
        Rz_w = _RzWrapper(Rz, num_dirs, filts.num_scales)
        _, _, flo2_ms, rel2_ms = vel_fj1990_approx_withz(
            Rz_w, filts.tuning_directions(),
            solve_2d_pxwise_k=FLOW_PARAMS['FJ_K'])
        del R, Rz
        _, peak_mem_flow = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    else:
        tracemalloc.start()
        R, Rz = filts.response(B, flux_noise_std=flux_noise_std)
        R_fmt = [[R.get((d, s), None) for s in range(filts.num_scales)]
                 for d in range(num_dirs)]
        pc = phase_congruency_3D_structure_tensor(
            R_fmt, filts.tuning_directions(),
            filter_energies=filts.filter_energies,
            flux_noise_std=flux_noise_std,
            noise_thresh_zmin=2)
        edges, _, _ = features_3D_structure_tensor(*pc)
        _, peak_mem_edges = tracemalloc.get_traced_memory()

        Rz_w = _RzWrapper(Rz, num_dirs, filts.num_scales)
        _, _, flo2_ms, rel2_ms = vel_fj1990_approx_withz(
            Rz_w, filts.tuning_directions(),
            solve_2d_pxwise_k=FLOW_PARAMS['FJ_K'])
        del R, Rz
        _, peak_mem_flow = tracemalloc.get_traced_memory()
        tracemalloc.stop()

    runtime = (time.time() - t_start) / getattr(filts, 'runtime_divisor', 1)
    print(f"{runtime:.1f}s  "
          f"edge_mem={peak_mem_edges / (1024**2):.0f}MB  "
          f"flow_mem={peak_mem_flow / (1024**2):.0f}MB")

    estr = edges['strength']
    flo1 = flo_1d_to_uv(edges['normal_velocity'], edges['orientation'])
    str_thr  = float(np.percentile(estr, FLOW_PARAMS['FLO1_STR_PCTILE']))
    rel_flo1 = ((estr > str_thr) &
                (np.abs(edges['normal_velocity']) < FLOW_PARAMS['MAX_FLO']) &
                (edges['coherence'] > FLOW_PARAMS['FLO1_COH_THRESH']))

    return {
        'strength':           estr,
        'flo1':               flo1,
        'rel_flo1':           rel_flo1,
        'flo2_ms':            flo2_ms,
        'rel2_ms':            rel2_ms,
        'runtime':            runtime,
        'peak_mem_edges_mb':  peak_mem_edges / (1024**2),
        'peak_mem_flow_mb':   peak_mem_flow  / (1024**2),
    }


def temporal_crop(arr, n_skip):
    N = arr.shape[-1]
    return arr[..., n_skip:-n_skip] if N > 2 * n_skip else arr


def _crop_result(r, n_skip):
    r['strength_crop']  = temporal_crop(r['strength'],  n_skip)
    r['flo1_crop']      = temporal_crop(r['flo1'],      n_skip)
    r['rel_flo1_crop']  = temporal_crop(r['rel_flo1'],  n_skip)
    r['flo2_ms_crop']   = [temporal_crop(f, n_skip) for f in r['flo2_ms']]
    r['rel2_ms_crop']   = [temporal_crop(f, n_skip) for f in r['rel2_ms']]
    return r


def _flow_metrics(r, ref):
    flo1_cov = flow_coverage(r['rel_flo1_crop'])
    flo1_epe = flow_epe(r['flo1_crop'], ref['flo1_crop'],
                        r['rel_flo1_crop'], ref['rel_flo1_crop'])
    flo2_cov = float('nan')
    flo2_epe = float('nan')
    if r['flo2_ms_crop']:
        rel2_0     = r['rel2_ms_crop'][0] > FLOW_PARAMS['FLO2_REL_THRESH']
        rel2_ref_0 = ref['rel2_ms_crop'][0] > FLOW_PARAMS['FLO2_REL_THRESH']
        flo2_cov   = flow_coverage(rel2_0)
        flo2_epe   = flow_epe(r['flo2_ms_crop'][0], ref['flo2_ms_crop'][0],
                               rel2_0, rel2_ref_0)
    return flo1_cov, flo1_epe, flo2_cov, flo2_epe


def _save_method_videos(r, out_dir):
    ev = vis_edge(r['strength_crop'], thicken=False, prctile_lim=[75, 97])
    save_video_imageio(np.moveaxis((ev * 255).astype(np.uint8), -1, 0),
                       str(out_dir / "edges.mp4"))
    flo1_v = vis_flo_dense(r['flo1_crop'], r['rel_flo1_crop'],
                            FLOW_PARAMS['MAX_FLO'], FLOW_PARAMS['FLO1_THICKEN'])
    save_video_imageio(np.moveaxis(flo1_v, -1, 0), str(out_dir / "flo1.mp4"))
    for s, (f2, rel2) in enumerate(zip(r['flo2_ms_crop'], r['rel2_ms_crop'])):
        rel_mask = rel2 > FLOW_PARAMS['FLO2_REL_THRESH']
        flo2_v = vis_flo_dense(f2, rel_mask, FLOW_PARAMS['MAX_FLO'])
        save_video_imageio(np.moveaxis(flo2_v, -1, 0),
                           str(out_dir / f"flo2_s{s + 1}.mp4"))


def run_real_data(output_dir):
    mat_files = sorted(REAL_DATA_DIR.glob("*.mat"))
    if not mat_files:
        print("[real] No .mat files found in", REAL_DATA_DIR)
        return []

    print(f"\n{'='*70}")
    print(f"REAL DATA  ({len(mat_files)} files)")
    print(f"{'='*70}")

    records = []
    for mat_file in mat_files:
        print(f"\n  {mat_file.name}")
        data = loadmat(str(mat_file))
        B    = data['B'].astype(np.float32)
        M    = float(data.get('M', np.array([1.0])))
        if M != 1:
            B = B * (1.0 / M)

        flux_std = binom_noise_std(ippd(gaussian_filter(B, sigma=10)), M)

        out_dir = output_dir / "real" / mat_file.stem
        out_dir.mkdir(parents=True, exist_ok=True)

        b_crop = temporal_crop(B, N_SKIP)
        save_video_imageio(
            np.moveaxis((np.clip(b_crop, 0.0, 1.0) * 255).astype(np.uint8), -1, 0),
            str(out_dir / "B.mp4")
        )
        b_avg = gaussian_filter(b_crop.astype(np.float32), sigma=(0, 0, 5))
        save_video_imageio(
            np.moveaxis((np.clip(b_avg, 0.0, 1.0) * 255).astype(np.uint8), -1, 0),
            str(out_dir / "B_avg.mp4")
        )

        res = {}
        for name, cls in METHODS:
            r = run_pipeline(cls, B, name, flux_std)
            res[name] = _crop_result(r, N_SKIP)

        ref = res[REF_METHOD]
        method_metrics = {}
        for name in METHOD_NAMES:
            r = res[name]
            rel_edge = sweep_fscore_relative(r['strength_crop'], ref['strength_crop'])
            flo1_cov, flo1_epe, flo2_cov, flo2_epe = _flow_metrics(r, ref)
            if name == REF_METHOD:
                flo1_epe = flo2_epe = 0.0
            method_metrics[name] = {
                'rel_f':     rel_edge['fscore'],
                'rel_p':     rel_edge['precision'],
                'rel_r':     rel_edge['recall'],
                'rt':        r['runtime'],
                'mem_edges': r['peak_mem_edges_mb'],
                'mem_flow':  r['peak_mem_flow_mb'],
                'flo1_cov':  flo1_cov,
                'flo1_epe':  flo1_epe,
                'flo2_cov':  flo2_cov,
                'flo2_epe':  flo2_epe,
            }
            epe_str = f"{flo1_epe:.4f}" if not np.isnan(flo1_epe) else "nan"
            print(f"    [{name}]  rel_F={rel_edge['fscore']:.4f}  "
                  f"flo1_cov={flo1_cov:.3f}  flo1_epe={epe_str}  flo2_cov={flo2_cov:.3f}")

            d = out_dir / name.lower().replace(" ", "_")
            d.mkdir(exist_ok=True)
            _save_method_videos(r, d)

        records.append({'file': mat_file.name, 'M': M, 'methods': method_metrics})

    return records


def run_sim_data(output_dir):
    scene_dirs = sorted([d for d in SIM_DATA_DIR.iterdir()
                          if d.is_dir() and d.name != '.DS_Store'])
    if not scene_dirs:
        print("[sim] No scene directories found in", SIM_DATA_DIR)
        return []

    total_clips = sum(
        len(sorted([c for c in s.iterdir() if c.is_dir()])[:MAX_CLIPS_PER_SCENE])
        for s in scene_dirs) * len(PPP_LEVELS)
    print(f"\n{'='*70}")
    print(f"SIMULATED DATA  (~{total_clips} clips, {len(PPP_LEVELS)} PPP levels)")
    print(f"{'='*70}")

    records = []
    for scene_dir in scene_dirs:
        clip_dirs = sorted([d for d in scene_dir.iterdir()
                             if d.is_dir() and d.name != '.DS_Store'])[:MAX_CLIPS_PER_SCENE]
        for clip_dir in clip_dirs:
            for ppp in PPP_LEVELS:
                mat_file = clip_dir / f"ppp_{ppp:.2f}_data.mat"
                if not mat_file.exists():
                    continue
                clip_id = f"{scene_dir.name}/{clip_dir.name}/ppp_{ppp:.2f}"
                print(f"\n  {clip_id}")

                try:
                    data     = loadmat(str(mat_file))
                    B        = data['B'].astype(np.float32)
                    M        = int(data['M'].item())
                    edges_gt = data['edges_gt'].astype(np.uint8)
                    if M != 1:
                        B = B * (1.0 / M)

                    flux_std = binom_noise_std(ippd(gaussian_filter(B, sigma=10)), M)

                    out_dir = (output_dir / "sim" / scene_dir.name
                               / clip_dir.name / f"ppp_{ppp:.2f}")
                    out_dir.mkdir(parents=True, exist_ok=True)

                    res = {}
                    for name, cls in METHODS:
                        r = run_pipeline(cls, B, name, flux_std)
                        res[name] = _crop_result(r, N_SKIP)

                    gt_crop = temporal_crop(edges_gt, N_SKIP)
                    ref     = res[REF_METHOD]

                    method_metrics = {}
                    for name in METHOD_NAMES:
                        r    = res[name]
                        best = sweep_fscore(r['strength_crop'], gt_crop)
                        flo1_cov, flo1_epe, flo2_cov, flo2_epe = _flow_metrics(r, ref)
                        if name == REF_METHOD:
                            flo1_epe = flo2_epe = 0.0
                        method_metrics[name] = {
                            'f':         best['fscore'],
                            'p':         best['precision'],
                            'r':         best['recall'],
                            'rt':        r['runtime'],
                            'mem_edges': r['peak_mem_edges_mb'],
                            'mem_flow':  r['peak_mem_flow_mb'],
                            'flo1_cov':  flo1_cov,
                            'flo1_epe':  flo1_epe,
                            'flo2_cov':  flo2_cov,
                            'flo2_epe':  flo2_epe,
                        }
                        epe_str = f"{flo1_epe:.4f}" if not np.isnan(flo1_epe) else "nan"
                        print(f"    [{name}]  F={best['fscore']:.4f}  "
                              f"P={best['precision']:.4f}  R={best['recall']:.4f}  "
                              f"flo1_cov={flo1_cov:.3f}  flo1_epe={epe_str}")

                        d = out_dir / name.lower().replace(" ", "_")
                        d.mkdir(exist_ok=True)
                        _save_method_videos(r, d)

                    records.append({'clip': clip_id, 'ppp': ppp, 'methods': method_metrics})

                except Exception as e:
                    import traceback
                    print(f"    ERROR: {e}")
                    traceback.print_exc()

    return records


def _nanmean(vals):
    v = [x for x in vals if not (isinstance(x, float) and np.isnan(x))]
    return float(np.mean(v)) if v else float('nan')


def _format_real_summary(records):
    lines = []
    if not records:
        return lines
    lines.append(f"\n{'='*70}")
    lines.append("REAL DATA SUMMARY")
    lines.append(f"{'='*70}")

    col = 9
    hdr = f"{'File':<38}"
    for n in METHOD_NAMES:
        hdr += f" | {(n[:7]+' rF'):<{col}} {(n[:7]+' rt'):<{col}} {(n[:7]+' f1c'):<{col}} {(n[:7]+' f1e'):<{col}}"
    lines.append(hdr)
    lines.append("-" * len(hdr))

    for rec in records:
        row = f"{rec['file']:<38}"
        for n in METHOD_NAMES:
            m = rec['methods'][n]
            epe = f"{m['flo1_epe']:.3f}" if not np.isnan(m['flo1_epe']) else " nan"
            row += (f" | {m['rel_f']:{col}.3f} {m['rt']:{col}.1f} "
                    f"{m['flo1_cov']:{col}.3f} {epe:{col}}")
        lines.append(row)

    lines.append("-" * len(hdr))
    lines.append("Averages:")
    for n in METHOD_NAMES:
        vals = [rec['methods'][n] for rec in records]
        lines.append(f"  {n:<16}  relF={_nanmean([v['rel_f'] for v in vals]):.4f}  "
                     f"rt={_nanmean([v['rt'] for v in vals]):.1f}s  "
                     f"edge_mem={_nanmean([v['mem_edges'] for v in vals]):.0f}MB  "
                     f"flow_mem={_nanmean([v['mem_flow'] for v in vals]):.0f}MB  "
                     f"flo1_cov={_nanmean([v['flo1_cov'] for v in vals]):.3f}  "
                     f"flo1_epe={_nanmean([v['flo1_epe'] for v in vals]):.4f}  "
                     f"flo2_cov={_nanmean([v['flo2_cov'] for v in vals]):.3f}")
    return lines


def _format_sim_summary(records):
    lines = []
    if not records:
        return lines
    lines.append(f"\n{'='*70}")
    lines.append("SIMULATED DATA SUMMARY  (F-score vs ground truth + flow metrics)")
    lines.append(f"{'='*70}")

    col = 14
    for ppp in PPP_LEVELS:
        sub = [r for r in records if r['ppp'] == ppp]
        if not sub:
            continue
        lines.append(f"\n  PPP = {ppp:.2f}  ({len(sub)} clips)")
        hdr = f"  {'Metric':<18} " + " | ".join(f"{n:>{col}}" for n in METHOD_NAMES)
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for metric, key in [
            ('F-score',          'f'),
            ('Precision',        'p'),
            ('Recall',           'r'),
            ('flo1 coverage',    'flo1_cov'),
            ('flo1 EPE (vs LG)', 'flo1_epe'),
            ('flo2 coverage',    'flo2_cov'),
            ('flo2 EPE (vs LG)', 'flo2_epe'),
        ]:
            vals = {n: _nanmean([r['methods'][n][key] for r in sub]) for n in METHOD_NAMES}
            cells = []
            for n in METHOD_NAMES:
                v = vals[n]
                cells.append(f"{v:>{col}.4f}" if not np.isnan(v) else f"{'nan':>{col}}")
            lines.append(f"  {metric:<18} " + " | ".join(cells))

    lines.append(f"\n  OVERALL:")
    hdr = f"  {'Metric':<18} " + " | ".join(f"{n:>{col}}" for n in METHOD_NAMES)
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for metric, key in [
        ('Avg F-score',        'f'),
        ('Avg Runtime (s)',    'rt'),
        ('Avg EdgeMem (MB)',   'mem_edges'),
        ('Avg FlowMem (MB)',   'mem_flow'),
        ('flo1 coverage',      'flo1_cov'),
        ('flo2 coverage',      'flo2_cov'),
    ]:
        vals = {n: _nanmean([r['methods'][n][key] for r in records]) for n in METHOD_NAMES}
        cells = []
        for n in METHOD_NAMES:
            v = vals[n]
            cells.append(f"{v:>{col}.4f}" if not np.isnan(v) else f"{'nan':>{col}}")
        lines.append(f"  {metric:<18} " + " | ".join(cells))

    ref_rt       = _nanmean([r['methods'][REF_METHOD]['rt']        for r in records])
    ref_mem_edge = _nanmean([r['methods'][REF_METHOD]['mem_edges'] for r in records])
    lines.append(f"\n  vs {REF_METHOD}  (edge-mem saving reflects IIR streaming path):")
    for n in METHOD_NAMES:
        rt       = _nanmean([r['methods'][n]['rt']        for r in records])
        mem_edge = _nanmean([r['methods'][n]['mem_edges'] for r in records])
        lines.append(f"    {n:<16}  rt_ratio={rt/ref_rt:.2f}x  "
                     f"edge_mem_saving={(1 - mem_edge/ref_mem_edge)*100:.1f}%")
    lines.append(f"{'='*70}")
    return lines


def print_real_summary(records):
    for line in _format_real_summary(records):
        print(line)


def print_sim_summary(records):
    for line in _format_sim_summary(records):
        print(line)


def save_summary_txt(real_records, sim_records, path):
    import datetime
    lines = []
    lines.append(f"Single-Photon Filter Comparison")
    lines.append(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Methods: {', '.join(METHOD_NAMES)}")
    lines.append(f"Reference: {REF_METHOD}")
    lines.append("")
    lines += _format_real_summary(real_records)
    lines += _format_sim_summary(sim_records)
    with open(path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"\nSummary tables saved to {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Log Gabor vs Adaptive IIR vs Steerable — edges + optical flow")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--real-only", action="store_true",
                       help="Run only real-data evaluation")
    group.add_argument("--sim-only",  action="store_true",
                       help="Run only simulated-data evaluation")
    args = parser.parse_args()

    OUTPUT_BASE.mkdir(parents=True, exist_ok=True)

    real_records = []
    sim_records  = []

    if not args.sim_only:
        real_records = run_real_data(OUTPUT_BASE)

    if not args.real_only:
        sim_records = run_sim_data(OUTPUT_BASE)

    print(f"\n\n{'#'*70}")
    print("FINAL COMPARISON SUMMARY")
    print(f"{'#'*70}")

    print_real_summary(real_records)
    print_sim_summary(sim_records)

    out = {'real': real_records, 'sim': sim_records}
    with open(OUTPUT_BASE / "results.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nResults saved to {OUTPUT_BASE / 'results.json'}")

    save_summary_txt(real_records, sim_records, OUTPUT_BASE / "summary.txt")
