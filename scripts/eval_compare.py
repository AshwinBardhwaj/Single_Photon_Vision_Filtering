import os
import sys
import time
import json
import random
import numpy as np
from pathlib import Path

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
sys.path.append(str(root_dir / "scripts"))

# Mock save_video to avoid disk I/O and speed up execution
import fileio.fileio
def dummy_save_video(*args, **kwargs):
    pass
fileio.fileio.save_video_imageio = dummy_save_video

from run_base_sim import execute_pipeline_sim

DATA_BASEDIR = root_dir / "scripts" / "data"
SIM_DATA_DIR = DATA_BASEDIR / "sim_xvfi_1bit"

def get_all_mat_files(base_dir, test_folders=None):
    mat_files = []
    for root, _, files in os.walk(base_dir):
        # Only include specific test folders if provided
        if test_folders is not None:
            folder_name = Path(root).parts[-2] if len(Path(root).parts) >= 2 else ""
            # Some paths might have the folder deeper, so we check if any part is in test_folders
            if not any(f in Path(root).parts for f in test_folders):
                continue
                
        for file in files:
            if file.endswith('.mat') and 'ppp' in file:
                mat_files.append(Path(root) / file)
    return mat_files

def main():
    # Use folders 014 and 016 strictly as the hold-out test set
    TEST_FOLDERS = ['014', '016']
    mat_files = get_all_mat_files(SIM_DATA_DIR, test_folders=TEST_FOLDERS)
    if not mat_files:
        print("No .mat files found in", SIM_DATA_DIR)
        return

    # Select a random subset to evaluate quickly (or increase to evaluate all)
    num_samples = 20
    random.seed(42)
    selected_files = random.sample(mat_files, min(num_samples, len(mat_files)))

    print(f"Evaluating on {len(selected_files)} random simulated videos...")

    methods = ['log_gabor', 'steerable', 'learned']
    results = {m: {'fscore': [], 'precision': [], 'recall': []} for m in methods}

    filts_config = {
        'velocities': np.array([1, -1, 0.3, -0.3, 0]),
        'orientations': np.array([0, 60, 120]),
        'min_wavelength': 3,
        'num_scales': 3
    }

    params = {
        'TPC_NOISE_THRESH_ZMIN': 2,
        'FJ_SOLVE_2D_PXWISE_K': 0.1,
        'N_SKIP_OUTPUT': 5,
        'ESTR_VIS_PRCTILE_THRESH': [75, 97],
        'MAX_FLO': 1,
        'FLO1_VIS_THICKEN': 3,
        'FLO2_REL_THRESH': 0.5,
    }

    eval_config = {
        'tolerance': 2,
        'percentiles': list(range(50, 100, 5)),
    }

    output_dir = root_dir / "scripts" / "output" / "eval_temp"
    os.makedirs(output_dir, exist_ok=True)

    for i, data_file in enumerate(selected_files):
        print(f"\n--- Video {i+1}/{len(selected_files)}: {data_file.name} ---")
        
        for method in methods:
            print(f"> Running {method}...")
            params['FILTER_TYPE'] = method
            
            try:
                execute_pipeline_sim(
                    data_file=data_file,
                    output_dir=output_dir,
                    filts_config=filts_config,
                    params=params,
                    crop_bounds=None,
                    eval_config=eval_config,
                )
                
                # Read metrics from the saved json
                metrics_path = output_dir / "edge_metrics.json"
                if metrics_path.exists():
                    with open(metrics_path, 'r') as f:
                        metrics = json.load(f)
                    results[method]['fscore'].append(metrics['best_fscore'])
                    results[method]['precision'].append(metrics['best_precision'])
                    results[method]['recall'].append(metrics['best_recall'])
                else:
                    print(f"WARNING: No metrics found for {method} on {data_file.name}")
            except Exception as e:
                print(f"Error evaluating {method} on {data_file.name}: {e}")

    results_str = "\n" + "="*50 + "\nEVALUATION RESULTS\n" + "="*50 + "\n"
    for method in methods:
        f = np.array(results[method]['fscore'])
        p = np.array(results[method]['precision'])
        r = np.array(results[method]['recall'])
        
        if len(f) > 0:
            results_str += f"--- {method.upper()} ---\n"
            results_str += f"F-Score:   {np.mean(f):.4f} ± {np.std(f):.4f}\n"
            results_str += f"Precision: {np.mean(p):.4f} ± {np.std(p):.4f}\n"
            results_str += f"Recall:    {np.mean(r):.4f} ± {np.std(r):.4f}\n"
        else:
            results_str += f"--- {method.upper()} ---\n"
            results_str += "No valid results.\n"
    results_str += "="*50 + "\n"
    print(results_str)
    with open('eval_results.txt', 'w') as f_out:
        f_out.write(results_str)

if __name__ == "__main__":
    main()
