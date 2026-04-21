import numpy as np


def _range_zero_centered(L):
    if L % 2 == 0:
        return np.arange(-L // 2, L // 2)
    return np.arange(-(L - 1) // 2, (L - 1) // 2 + 1)


def synth_img_step_edge(L, theta, c, alpha, center=(0.0, 0.0)):
    xv = _range_zero_centered(L)
    xx, yy = np.meshgrid(xv, xv)
    cx, cy = center[0], center[1]
    ct = np.cos(np.deg2rad(theta))
    st = np.sin(np.deg2rad(theta))
    proj = (xx - cx) * ct + (yy - cy) * st
    S = np.where(proj >= 0, c * (1 + alpha), c * (1 - alpha))
    emap = np.abs(proj) <= 0.5
    return S, emap


def synth_img_circ_edge(L, r, c, alpha, center=(0.0, 0.0)):
    xv = _range_zero_centered(L)
    xx, yy = np.meshgrid(xv, xv)
    cx, cy = center[0], center[1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    inside = d2 <= r ** 2
    S = np.where(inside, c * (1 + alpha), c * (1 - alpha))
    emap = np.abs(np.hypot(xx - cx, yy - cy) - r) <= 0.5
    return S, emap


def synth_img_cross_edge(L, theta, c, alpha, center=(0.0, 0.0)):
    xv = _range_zero_centered(L)
    xx, yy = np.meshgrid(xv, xv)
    cx, cy = center[0], center[1]
    angl = np.rad2deg(np.arctan2(yy - cy, xx - cx))
    theta1 = theta
    theta2 = theta1 - 90 if theta1 > 0 else theta1 + 90

    if theta1 == np.pi / 2:
        z1 = (angl < 90 - theta1) & (angl >= -90 - theta1)
    else:
        z1 = (angl <= 90 - theta1) & (angl > -90 - theta1)
    if theta2 == np.pi / 2:
        z2 = (angl < 90 - theta2) & (angl >= -90 - theta2)
    else:
        z2 = (angl <= 90 - theta2) & (angl > -90 - theta2)

    bright = (z1 & z2) | (~(z1 | z2))
    S = np.where(bright, c * (1 + alpha), c * (1 - alpha))

    ct1 = np.cos(np.deg2rad(theta1))
    st1 = np.sin(np.deg2rad(theta1))
    ct2 = np.cos(np.deg2rad(theta2))
    st2 = np.sin(np.deg2rad(theta2))
    emap1 = np.abs((xx - cx) * ct1 - (yy - cy) * st1) <= 0.5
    emap2 = np.abs((xx - cx) * ct2 - (yy - cy) * st2) <= 0.5
    emap = emap1 | emap2
    return S, emap


def synth_img_line_edge(L, thickness, theta, c, alpha, center=(0.0, 0.0)):
    xv = _range_zero_centered(L)
    xx, yy = np.meshgrid(xv, xv)
    cx, cy = center[0], center[1]
    ct = np.cos(np.deg2rad(theta))
    st = np.sin(np.deg2rad(theta))
    d = np.abs((xx - cx) * ct + (yy - cy) * st)
    z = d <= thickness / 2
    S = np.where(z, c * (1 + alpha), c * (1 - alpha))
    emap = d <= 0.5
    return S, emap


def synth_img_gaussian_blob(L, sigma, b, beta, center=(0.0, 0.0)):
    xv = _range_zero_centered(L)
    xx, yy = np.meshgrid(xv, xv)
    cx, cy = center[0], center[1]
    d = np.hypot(xx - cx, yy - cy)
    return b * (1 + beta * np.exp(-(d ** 2) / (2 * sigma ** 2)))


def _build_video(frame_fn, N, v):
    v = np.asarray(v, dtype=np.float64)
    loc0 = -(N / 2) * v + np.array([0.5, 0.5])
    frames, emaps = [], []
    for n in range(1, N + 1):
        img, em = frame_fn(loc0 + n * v)
        frames.append(img)
        emaps.append(em)
    S = np.stack(frames, axis=-1).astype(np.float32)
    emap = np.stack(emaps, axis=-1)
    return S, emap


def synth_vid_step_edge(L, theta, c, alpha, N, v):
    return _build_video(
        lambda ctr: synth_img_step_edge(L, theta, c, alpha, ctr), N, v
    )


def synth_vid_circ_edge(L, radius, c, alpha, N, v):
    return _build_video(
        lambda ctr: synth_img_circ_edge(L, radius, c, alpha, ctr), N, v
    )


def synth_vid_cross_edge(L, theta, c, alpha, N, v):
    return _build_video(
        lambda ctr: synth_img_cross_edge(L, theta, c, alpha, ctr), N, v
    )


def synth_vid_line_edge(L, thickness, theta, c, alpha, N, v):
    return _build_video(
        lambda ctr: synth_img_line_edge(L, thickness, theta, c, alpha, ctr), N, v
    )


def synth_vid_moving_1D_box(L, w, N, v, loc0=None):
    if loc0 is None:
        loc0 = L / 2 - (N / 2) * v
    Y = np.zeros((N, L))
    half_w = int(np.floor(w / 2))
    for n in range(1, N + 1):
        loc = loc0 + v * n
        j0 = max(1, min(L, int(round(loc - half_w))))
        j1 = max(1, min(L, int(round(loc + half_w))))
        Y[n - 1, j0 - 1:j1] = 1
    return Y


def synth_vid_linear_phase(sz, freqs):
    H, W, N = int(sz[0]), int(sz[1]), int(sz[2])
    x_I = np.arange(W, dtype=np.float32).reshape(1, W, 1)
    y_I = np.arange(H, dtype=np.float32).reshape(H, 1, 1)
    t_I = np.arange(N, dtype=np.float32).reshape(1, 1, N)

    freqs = np.asarray(freqs)
    nfilt = freqs.shape[1]
    P = np.zeros((H, W, N, nfilt), dtype=np.float32)
    for i in range(nfilt):
        kx = freqs[i, 0]
        ky = freqs[i, 1]
        omega = freqs[i, 2]
        P[:, :, :, i] = np.angle(
            np.exp(1j * (kx * x_I + ky * y_I + omega * t_I))
        )
    return P


if __name__ == "__main__":
    import os
    import imageio.v3 as iio

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vids")
    os.makedirs(out_dir, exist_ok=True)

    def _to_uint8(S):
        lo, hi = float(S.min()), float(S.max())
        if hi <= lo:
            return np.zeros(S.shape, dtype=np.uint8)
        return ((S - lo) / (hi - lo) * 255.0).astype(np.uint8)

    def save_video_gif(name, S, fps=15):
        path = os.path.join(out_dir, name + ".gif")
        frames_u8 = _to_uint8(S)
        frames = [frames_u8[:, :, k] for k in range(frames_u8.shape[2])]
        iio.imwrite(path, frames, duration=1000 / fps, loop=0)
        print(f"wrote {path}  shape={S.shape}  dtype={S.dtype}  "
              f"range=[{S.min():.3f}, {S.max():.3f}]")

    def save_image_png(name, Y):
        path = os.path.join(out_dir, name + ".png")
        iio.imwrite(path, _to_uint8(Y))
        print(f"wrote {path}  shape={Y.shape}")

    L = 128
    N = 30
    c = 1.0
    alpha = 0.5

    print("=== generating sample videos ===")

    S, emap = synth_vid_step_edge(L, 30, c, alpha, N, [1.5, 0.5])
    save_video_gif("step_edge", S)
    save_video_gif("step_edge_emap", emap.astype(np.float32))

    S, emap = synth_vid_circ_edge(L, 20, c, alpha, N, [1.0, 1.0])
    save_video_gif("circ_edge", S)

    S, emap = synth_vid_cross_edge(L, 30, c, alpha, N, [0.8, 0.4])
    save_video_gif("cross_edge", S)

    S, emap = synth_vid_line_edge(L, 4, 60, c, 0.8, N, [1.2, 0.6])
    save_video_gif("line_edge", S)

    Y = synth_vid_moving_1D_box(L, 10, N, 3.0)
    save_image_png("moving_1D_box_spacetime", Y)

    freqs = np.array([[0.30, 0.10, 0.20],
                      [0.10, 0.30, 0.10],
                      [0.05, 0.05, 0.40]])
    P = synth_vid_linear_phase([L, L, N], freqs)
    for i in range(freqs.shape[1]):
        save_video_gif(f"linear_phase_{i}", P[:, :, :, i])

    print(f"\nall outputs in: {out_dir}")