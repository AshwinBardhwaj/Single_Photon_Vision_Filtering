import os
import sys
import gc
import shutil
from pathlib import Path
import numpy as np
import imageio.v2 as imageio
from scipy.io import savemat
from skimage.feature import canny
from scipy.ndimage import uniform_filter1d

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sensor.sensor import quanta_sample_direct

XVFI_TEST_ROOT = Path("../encoded_train")
OUTPUT_ROOT = PROJECT_ROOT / "scripts" / "data" / "sim_xvfi_1bit"

DOWNSAMPLE = 8
FLUX_LEVELS_PPP = [0.1, 0.5, 1.25, 2.0]
SEED = 0

CANNY_SIGMA = 1.0
CANNY_LOW = 0.1
CANNY_HIGH = 0.2

MP4_FPS = 10
SMOOTH_WINDOW = 15


def block_mean_downsample_single(img, ds):
    H, W = img.shape
    H2, W2 = (H // ds) * ds, (W // ds) * ds
    return img[:H2, :W2].reshape(H2 // ds, ds, W2 // ds, ds).mean(axis=(1, 3))


def load_clip_as_radiance(video_path, ds=DOWNSAMPLE):
    reader = imageio.get_reader(video_path)
    try:
        n_frames = reader.count_frames()
    except Exception:
        n_frames = sum(1 for _ in reader)
        reader.close()
        reader = imageio.get_reader(video_path)

    first_frame = reader.get_data(0)
    H, W = first_frame.shape[:2]
    out_H, out_W = H // ds, W // ds

    scene = np.zeros((out_H, out_W, n_frames), dtype=np.float32)

    for t, img in enumerate(reader):
        img = img.astype(np.float32) / 255.0
        if img.ndim == 3:
            img = 0.299 * img[..., 0] + 0.587 * img[..., 1] + 0.114 * img[..., 2]
        if ds > 1:
            img = block_mean_downsample_single(img, ds)
        scene[:, :, t] = img

    reader.close()
    return scene


def simulate_1bit(scene, mean_flux_ppp, rng):
    mu = float(scene.mean())
    if mu <= 1e-12:
        raise ValueError("zero-mean scene")
    flux = scene * (np.float32(mean_flux_ppp) / np.float32(mu))
    B = quanta_sample_direct(flux, M=1, eta=1.0, dcr=0.0, tau=1.0, rng=rng)
    return B.astype(np.uint8), flux


def canny_gt_per_frame(scene, sigma=CANNY_SIGMA, low=CANNY_LOW, high=CANNY_HIGH):
    H, W, T = scene.shape
    edges = np.zeros((H, W, T), dtype=np.uint8)
    for t in range(T):
        edges[:, :, t] = canny(scene[:, :, t], sigma=sigma,
                               low_threshold=low, high_threshold=high)
    return edges


def write_binary_mp4(B, path, fps=MP4_FPS):
    H, W, T = B.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p",
                                macro_block_size=2)
    for t in range(T):
        frame = (B[:H2, :W2, t] * 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()


def write_scene_mp4(scene, path, fps=MP4_FPS):
    H, W, T = scene.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p",
                                macro_block_size=2)
    for t in range(T):
        frame = (np.clip(scene[:H2, :W2, t], 0.0, 1.0) * 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()


def write_smoothed_mp4(B, path, window=SMOOTH_WINDOW, fps=MP4_FPS):
    H, W, T = B.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    Bf = uniform_filter1d(B.astype(np.float32), size=window, axis=2, mode="nearest")
    max_val = Bf.max()
    if max_val > 0:
        Bf = Bf / max_val
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p",
                                macro_block_size=2)
    for t in range(T):
        frame = (Bf[:H2, :W2, t] * 255).astype(np.uint8)
        writer.append_data(frame)
    writer.close()


def write_overlay_mp4(B, edges_gt, path, fps=MP4_FPS):
    H, W, T = B.shape
    H2, W2 = (H // 2) * 2, (W // 2) * 2
    writer = imageio.get_writer(str(path), fps=fps, codec="libx264",
                                quality=8, pixelformat="yuv420p",
                                macro_block_size=2)
    for t in range(T):
        gray = (B[:H2, :W2, t] * 255).astype(np.uint8)
        rgb = np.stack([gray, gray, gray], axis=-1)
        e = edges_gt[:H2, :W2, t].astype(bool)
        rgb[e] = [255, 0, 0]
        writer.append_data(rgb)
    writer.close()


def find_test_videos(root):
    videos = []
    for sub_dir in sorted(Path(root).iterdir()):
        if not sub_dir.is_dir():
            continue
        for vid_file in sorted(sub_dir.glob("*.mp4")):
            videos.append(vid_file)
    return videos


def main():
    rng = np.random.default_rng(SEED)

    videos = find_test_videos(XVFI_TEST_ROOT)
    print(f"found {len(videos)} videos under {XVFI_TEST_ROOT}")

    for vid_path in videos:
        vid_out_dir = OUTPUT_ROOT / vid_path.parent.name / vid_path.stem
        vid_out_dir.mkdir(parents=True, exist_ok=True)

        scene = load_clip_as_radiance(vid_path)
        edges_gt = canny_gt_per_frame(scene)

        actual_n_frames = scene.shape[2]

        shutil.copy2(vid_path, vid_out_dir / "ground_truth_original.mp4")

        scene_mp4 = vid_out_dir / "ground_truth_downsampled.mp4"
        write_scene_mp4(scene, scene_mp4)
        print(f"Processed: {vid_path.name} | Output directory: {vid_out_dir}")

        for ppp in FLUX_LEVELS_PPP:
            B, flux = simulate_1bit(scene, ppp, rng)

            mat_path = vid_out_dir / f"ppp_{ppp:.2f}_data.mat"
            mp4_path = vid_out_dir / f"ppp_{ppp:.2f}_binary.mp4"
            overlay_path = vid_out_dir / f"ppp_{ppp:.2f}_overlay_gt.mp4"
            smoothed_path = vid_out_dir / f"ppp_{ppp:.2f}_smoothed.mp4"

            savemat(mat_path, {
                "B": B,
                "flux": flux,
                "scene": scene,
                "edges_gt": edges_gt,
                "mean_ppp": float(ppp),
                "M": 1,
                "eta": 1.0,
                "dcr": 0.0,
                "tau": 1.0,
                "downsample": DOWNSAMPLE,
                "n_frames": actual_n_frames,
                "canny_sigma": CANNY_SIGMA,
                "canny_low": CANNY_LOW,
                "canny_high": CANNY_HIGH,
            }, do_compression=True)

            write_binary_mp4(B, mp4_path)
            write_overlay_mp4(B, edges_gt, overlay_path)
            write_smoothed_mp4(B, smoothed_path)

            p_th = 1.0 - np.exp(-ppp)
            print(f"  ppp={ppp:.2f}: shape={B.shape}  mean(B)={B.mean():.4f}  "
                  f"theory={p_th:.4f}  edges_gt density={edges_gt.mean():.4f}")

            del B, flux
            gc.collect()

        del scene, edges_gt
        gc.collect()


if __name__ == "__main__":
    main()