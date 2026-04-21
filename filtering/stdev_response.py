import numpy as np

def stdev_response(flux_noise_std, filter_energy):
    
    eps_single = np.finfo(np.float32).eps
    
    std_R = np.maximum(eps_single, flux_noise_std * np.sqrt(0.5 * filter_energy))
    
    return std_R