import imageio
import numpy as np


def save_video_imageio(burst, filepath, fps=20):
    """
    Saves a burst of frames to an MP4 video using imageio.
    burst: shape (N, H, W) for grayscale or (N, H, W, 3) for RGB.
    """
    # Uncomment to convert from MATLAB shape to Python shape:
    # burst = np.moveaxis(burst, -1, 0)

    assert burst.ndim in [3, 4], "Burst must be 3D or 4D"

    writer = imageio.get_writer(filepath, fps=fps)

    for frame in burst:
        if frame.dtype != np.uint8:
            if frame.dtype in [np.float32, np.float64]:
                frame = np.clip(frame * 255.0, 0, 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        writer.append_data(frame)

    writer.close()