import numpy as np
import scipy.io as sio
import os
import sys
import subprocess
from scipy.fft import fft, ifft, fft2, ifft2, ifftn, fftshift

# 1. HANDLE PATHS
# Get the directory where this test script lives
test_dir = os.path.abspath(os.path.dirname(__file__))
# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(test_dir, '..')))

# 2. IMPORT THE CONVERTED CLASS
# Assuming the file is named log_gabor_filter_bank.py and the class is LogGaborBank3DSepT
try:
    from log_gabor_filter_bank import LogGaborBank3DSepT
except ImportError as e:
    print(f"Error: Could not import LogGaborBank3DSepT from log_gabor_filter_bank.py")
    print(f"Check if the filename and class name match. {e}")
    sys.exit(1)

def generate_matlab_validation_script(input_filename, output_filename):
    """Explicitly targets the 'filtering' subfolder and adds it to MATLAB path."""
    matlab_dir = os.path.join(test_dir, 'filtering')
    
    matlab_code = f"""
    try
        % 1. Move to the directory containing the class and helpers
        if exist('{matlab_dir}', 'dir')
            cd('{matlab_dir}');
            addpath(genpath('{matlab_dir}')); 
        else
            error('Could not find filtering directory at: {matlab_dir}');
        end
        
        % 2. Move back to the test directory to save output/load input
        cd('{test_dir}');
        addpath('{test_dir}');

        % Verify class visibility
        if exist('logGabor_bank_3D_sep_t', 'class') ~= 8
            error('Class logGabor_bank_3D_sep_t still not found in path.');
        end

        load('{input_filename}', 'test_input');
        
        obj = logGabor_bank_3D_sep_t();
        obj.input_size = size(test_input);
        obj = obj.set_up_filters();
        
        [R_mat, ~] = obj.response(test_input);
        filter_energies = obj.filter_energies;
        
        save('{output_filename}', 'R_mat', 'filter_energies');
        fprintf('Success: Results saved.\\n');
    catch ME
        fprintf('MATLAB Error: %s\\n', ME.message);
        for k = 1:length(ME.stack)
            fprintf('  In %s (line %d)\\n', ME.stack(k).name, ME.stack(k).line);
        end
        exit(1);
    end
    exit(0);
    """
    with open(os.path.join(test_dir, "run_matlab_validation.m"), "w") as f:
        f.write(matlab_code)

def run_numerical_comparison():
    # Use small dimensions for testing
    size_dims = (32, 32, 20)
    test_input = np.random.rand(*size_dims).astype(np.float32)
    
    # Save input.mat inside the test directory
    input_path = os.path.join(test_dir, 'test_input.mat')
    output_path = os.path.join(test_dir, 'test_output_matlab.mat')
    sio.savemat(input_path, {'test_input': test_input})
    
    print("--- Running Python Implementation ---")
    bank = LogGaborBank3DSepT()
    bank.input_size = list(size_dims)
    bank.set_up_filters()
    R_py = bank.response(test_input)
    
    print("--- Running MATLAB Implementation ---")
    # Pass only filenames to MATLAB since we handle directory switching inside the .m script
    generate_matlab_validation_script('test_input.mat', 'test_output_matlab.mat')
    
    try:
        # Command ensures MATLAB finds the generated script
        command = f"addpath('{test_dir}'); run_matlab_validation"
        print(f"Executing: matlab -batch \"{command}\"")
        subprocess.run(["matlab", "-batch", command], check=True)
        
    except FileNotFoundError:
        print("Error: 'matlab' executable not found in PATH.")
        return
    except subprocess.CalledProcessError:
        # Error details are printed by MATLAB catch block
        return

    # 5. Numerical Verification
    if os.path.exists(output_path):
        mat_data = sio.loadmat(output_path)
        R_mat = mat_data['R_mat']
        E_mat = mat_data['filter_energies']
        
        print("\n=== COMPARISON RESULTS ===")
        # Check energy
        energy_diff = np.linalg.norm(bank.filter_energies - E_mat)
        print(f"Filter Energy Difference: {energy_diff:.2e}")
        
        # Check Response (Filt 0, Scale 0)
        # Python Key (filt_idx, scale_idx) -> (0, 0)
        # MATLAB cell (1, 1) loaded by sio.loadmat as R_mat[0, 0]
        py_res = R_py[(0, 0)]
        mat_res = R_mat[0, 0]
        
        rel_error = np.linalg.norm(py_res - mat_res) / np.linalg.norm(mat_res)
        print(f"Response Rel. Error (Scale 0, Filt 0): {rel_error:.2e}")
        
        if rel_error < 1e-4:
            print("\n✅ Success! Python matches MATLAB within single-precision tolerance.")
        else:
            print("\n❌ Mismatch detected. Investigate FFT axes or filter normalization.")

if __name__ == "__main__":
    run_numerical_comparison()