import os
import sys
import numpy as np

# 1. HANDLE PATHS (Equivalent to addpath and scriptdir)
# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Replicate addpath(genpath('matlab')) - adds the 'matlab' subfolder to Python path
sys.path.append(os.path.join(script_dir, 'matlab'))

# 2. DEFINE BASE DIRECTORIES
DATA_BASEDIR = os.path.join(script_dir, 'data')
OUTPUT_BASEDIR = os.path.join(script_dir, 'output')
OUTPUT_NAME = 'temp'

# 3. INITIALIZE FILTER BANK
# Assuming you have the LogGaborBank3DSepT class imported
from log_gabor_filter_bank import LogGaborBank3DSepT
filts = LogGaborBank3DSepT()

# 4. TUNING PARAMETERS
TPC_NOISE_THRESH_ZMIN = 4
FJ_SOLVE_2D_PXWISE_K = 1e-1

# 5. OUTPUT AND CROPPING CONSTRAINTS
# avoid periodicity assumption artifacts
N_SKIP_OUTPUT = 15

# Python uses 0-based indexing. 
# Subtracting 1 from MATLAB indices to maintain spatial alignment.
OUT_IMIN, OUT_IMAX = 0, 256
OUT_JMIN, OUT_JMAX = 0, 512

# 6. VISUALIZATION AND RELIABILITY THRESHOLDS
ESTR_VIS_PRCTILE_THRESH = [75, 97]
MAX_FLO = 1  # for both visualization and for reliability estimation
FLO1_VIS_THICKEN = 3
FLO2_REL_THRESH = 0.5

# Print confirmation (Optional)
if __name__ == "__main__":
    print(f"Project initialized. Output will be saved to: {OUTPUT_BASEDIR}")