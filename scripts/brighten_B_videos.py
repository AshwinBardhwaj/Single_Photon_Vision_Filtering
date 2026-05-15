import sys
from pathlib import Path
import numpy as np
import imageio

GAMMA = 0.6
OUTPUT_BASE = Path(__file__).resolve().parent / "output"


def brighten_video(src: Path):
    dst = src.parent / "B_bright.mp4"
    reader = imageio.get_reader(str(src))
    fps = reader.get_meta_data().get("fps", 20)
    writer = imageio.get_writer(str(dst), fps=fps)
    for frame in reader:
        f = frame.astype(np.float32) / 255.0
        f = np.clip(f, 0, 1) ** GAMMA
        writer.append_data((f * 255).astype(np.uint8))
    reader.close()
    writer.close()
    print(f"Saved {dst}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
