import os
from pathlib import Path
import sys
import numpy as np
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
DATA_BASEDIR = root_dir / "scripts" / "data"
OUTPUT_BASEDIR = root_dir / "scripts" / "output"
from run_base import execute_pipeline

data_file = DATA_BASEDIR / "fig08_vision_0604-ball-3.mat"
output_name = "Fig08_21"
output_dir = OUTPUT_BASEDIR / output_name

filts_config = {
    'velocities': np.array([1, -1, 0.3, -0.3, 0]),
    'orientations': np.array([0, 60, 120]),
    'min_wavelength': 3,
    'num_scales': 3
}

params = {
    'TPC_NOISE_THRESH_ZMIN': 2,
    'FJ_SOLVE_2D_PXWISE_K': 0.1,
    'N_SKIP_OUTPUT': 15,
    'ESTR_VIS_PRCTILE_THRESH': [75, 97],
    'MAX_FLO': 1,
    'FLO1_VIS_THICKEN': 3,
    'FLO2_REL_THRESH': 0.5,
}

OUT_IMIN = 0
OUT_IMAX = 256
OUT_JMIN = 93
OUT_JMAX = 349
crop_bounds = (OUT_IMIN, OUT_IMAX, OUT_JMIN, OUT_JMAX)

if __name__ == "__main__":
    execute_pipeline(
        data_file=data_file,
        output_dir=output_dir,
        filts_config=filts_config,
        params=params,
        crop_bounds=crop_bounds
    )
