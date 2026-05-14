import os
from pathlib import Path
import sys
import numpy as np

root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
DATA_BASEDIR = root_dir / "scripts" / "data"
OUTPUT_BASEDIR = root_dir / "scripts" / "output"

sys.path.append(str(root_dir / "scripts"))
from run_base_sim import execute_pipeline_sim

SIM_DATA_DIR = DATA_BASEDIR / "sim_xvfi_1bit"

# ---------------------------------------------------------
# Set these variables to match the clip you want to process
# ---------------------------------------------------------
PARENT_DIR = "005"  # The folder number, e.g., '004', '005'
VID_NAME = "occ008.269_f1617"  # The name of the video file without .mp4
PPP = 1.25  # The flux level to evaluate

# Construct the correct path based on the new folder structure
data_file = SIM_DATA_DIR / PARENT_DIR / VID_NAME / f"ppp_{PPP:.2f}_data.mat"

# Set up an organized output directory
output_name = f"{VID_NAME}_ppp{PPP:.2f}"
output_dir = OUTPUT_BASEDIR / PARENT_DIR / output_name

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
    'FILTER_TYPE': 'learned'
}

eval_config = {
    'tolerance': 2,
    'percentiles': list(range(50, 100, 5)),
}

crop_bounds = None

if __name__ == "__main__":
    if not data_file.exists():
        raise FileNotFoundError(f"Could not find the data file: {data_file}")

    print(f"Loading data from: {data_file}")
    print(f"Saving output to: {output_dir}")

    execute_pipeline_sim(
        data_file=data_file,
        output_dir=output_dir,
        filts_config=filts_config,
        params=params,
        crop_bounds=crop_bounds,
        eval_config=eval_config,
    )