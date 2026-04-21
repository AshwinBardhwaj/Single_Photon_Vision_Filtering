import numpy as np

def flo_1d_to_uv(vn, vori):
    # 1. Convert orientation to radians for NumPy
    rad = np.radians(vori)
    
    # 2. Project components
    u = vn * np.cos(rad)
    v = vn * np.sin(rad)
    
    # 3. Handle dimension insertion and concatenation
    # MATLAB's reshape to [sz(1:2) 1 sz(3:end)] adds a singleton at dim 3
    u_expanded = np.expand_dims(u, axis=2)
    v_expanded = np.expand_dims(v, axis=2)
    
    # 4. Concatenate along axis 2 (the 3rd dimension)
    V = np.concatenate([u_expanded, v_expanded], axis=2)
    
    return V