import os
from pathlib import Path
import sys
import numpy as np
# Add the repository root to Python's search path
root_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(root_dir))
# 1. Base Paths
# DATA_BASEDIR = Path("data")
# OUTPUT_BASEDIR = Path("output")
#
# 1. Base Paths
DATA_BASEDIR = root_dir / "scripts" / "data"
OUTPUT_BASEDIR = root_dir / "scripts" / "output"
from run_base import execute_pipeline # Import the runner we just made
# 2. Parameters from init.m and fig05_02.m
data_file = DATA_BASEDIR / "fig05_vision_0604-jump-1.mat"
output_name = "Fig05_02"
output_dir = OUTPUT_BASEDIR / output_name

# Filter configuration
filts_config = {
    'velocities': np.array([1, -1, 0.3, -0.3, 0]),
    'orientations': np.array([0, 60, 120]),
    'min_wavelength': 4,
    'num_scales': 3
}

# General parameters mapped from MATLAB globals
params = {
    'TPC_NOISE_THRESH_ZMIN': 2,        # Overridden in Fig05 file
    'FJ_SOLVE_2D_PXWISE_K': 1e-1,      # From init
    'N_SKIP_OUTPUT': 15,               # From init
    'ESTR_VIS_PRCTILE_THRESH': [75, 97], # From init (comments in fig05 suggest it was tweaked, keeping init)
    'MAX_FLO': 1,                      # From init
    'FLO1_VIS_THICKEN': 3,             # From init
    'FLO2_REL_THRESH': 0.5             # From init
}

# 3. Crop bounds for the paper (converted to 0-based indexing)
OUT_IMIN = 0
OUT_IMAX = 256
OUT_JMIN = 127
OUT_JMAX = OUT_JMIN + 256
crop_bounds = (OUT_IMIN, OUT_IMAX, OUT_JMIN, OUT_JMAX)

# 4. Execute
if __name__ == "__main__":
    execute_pipeline(
        data_file=data_file,
        output_dir=output_dir,
        filts_config=filts_config,
        params=params,
        crop_bounds=crop_bounds
    )