import numpy as np

class TunedFrequency:
    @staticmethod
    def freq_2d_ori(f0, thd):
        """
        Calculates 2D frequency orientation.
        thd is expected in degrees.
        """
        # np.deg2rad converts degrees to radians for np.cos and np.sin
        rad = np.radians(thd)
        return f0 * np.array([np.cos(rad), np.sin(rad)])

    @staticmethod
    def freq_t_vtuned(fs, vn):
        """
        Calculates the temporal frequency component.
        Matches MATLAB's dot product behavior.
        """
        # Ensure inputs are numpy arrays for dot product
        fs = np.atleast_1d(fs)
        vn = np.atleast_1d(vn)
        
        if fs.shape != vn.shape:
            raise ValueError("Dimensions of fs and vn must match.")
            
        return -np.dot(fs, vn)

    @staticmethod
    def freq_3d_vtuned(f0, thd, v):
        """
        Combines 2D orientation and temporal frequency into a 3D vector.
        """
        f_2d = TunedFrequency.freq_2d_ori(f0, thd)
        f_t = TunedFrequency.freq_t_vtuned(f0, v)
        
        # Concatenate into a single 1x3 vector
        return np.append(f_2d, f_t)

    @staticmethod
    def phid_vtuned(v):
        """
        Calculates the frequency-domain angle phid.
        Result is in degrees, shifted to the [0, 180] range.
        """
        v = np.atleast_1d(v)
        
        # acot(x) in degrees is atan(1/x)
        # We use np.arctan2(1, v) for better numerical stability
        phid = np.degrees(np.arctan2(1, v))
        
        # Ensure the angle is within [0, 180]
        phid = np.where(phid < 0, phid + 180, phid)
        
        return phid if phid.size > 1 else phid.item()

# --- Example Usage ---
# f = TunedFrequency.freq_3d_vtuned(10, 45, 0.5)
# p = TunedFrequency.phid_vtuned(0.5)