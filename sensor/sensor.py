import numpy as np


def ppd(x):
    return 1.0 - np.exp(-x)


def ippd(p):
    return -np.log(1.0 - p)


def calc_bitdepth(M):
    return max(1, int(np.ceil(np.log2(M))))


def binom_noise_std(c, M):
    p = ppd(c)
    return np.sqrt(p * (1.0 - p) / M)


def quanta_sample_direct(x, sz=None, M=1, sigma_read=0.0,
                         eta=1.0, dcr=0.0, tau=1.0, rng=None):
    x = np.asarray(x, dtype=np.float64)
    if sz is None:
        sz = x.shape
    if rng is None:
        rng = np.random.default_rng()

    xin = (eta * x + dcr) * tau
    xin = np.broadcast_to(xin, sz).astype(np.float64, copy=False)

    y = rng.poisson(xin, size=sz).astype(np.float64)

    if sigma_read > 0:
        read_noise = sigma_read * rng.standard_normal(sz)
        y = np.maximum(0.0, np.round(y + read_noise))

    if np.isfinite(M):
        y = np.minimum(y, M)

    return y


def quanta_sample_burst(x, sz=None, M=1, sigma_read=0.0, rng=None):
    x = np.asarray(x, dtype=np.float64)
    if sz is None:
        sz = x.shape
    if rng is None:
        rng = np.random.default_rng()

    x1 = x if M == 1 else x / M
    y = np.zeros(sz, dtype=np.float64)
    for _ in range(M):
        y = y + quanta_sample_direct(x1, sz=sz, M=1,
                                     sigma_read=sigma_read, rng=rng)
    return y


if __name__ == "__main__":
    np.set_printoptions(precision=6, suppress=True)
    rng = np.random.default_rng(42)

    print("=== ppd ===")
    print(f"ppd(0)    = {ppd(0.0):.6f}")
    print(f"ppd(1)    = {ppd(1.0):.6f}")
    print(f"ppd(5)    = {ppd(5.0):.6f}")
    print(f"ppd(inf)  = {ppd(np.inf):.6f}")
    print(f"ppd array = {ppd(np.array([0.0, 0.5, 1.0, 2.0, 5.0]))}")

    print("\n=== ippd ===")
    print(f"ippd(0)    = {ippd(0.0):.6f}")
    print(f"ippd(0.5)  = {ippd(0.5):.6f}")
    print(f"ippd(0.99) = {ippd(0.99):.6f}")
    print(f"ippd array = {ippd(np.array([0.0, 0.25, 0.5, 0.75]))}")

    print("\n=== roundtrip ippd(ppd(x)) ===")
    xs = np.array([0.1, 1.0, 2.5, 5.0, 10.0])
    print(f"x            = {xs}")
    print(f"ippd(ppd(x)) = {ippd(ppd(xs))}")

    print("\n=== calc_bitdepth ===")
    for M in [1, 2, 3, 4, 15, 16, 17, 255, 256, 1024]:
        print(f"calc_bitdepth({M}) = {calc_bitdepth(M)}")

    print("\n=== binom_noise_std ===")
    cs = np.array([0.1, 0.5, 1.0, 2.0])
    print(f"c = {cs}")
    for M in [1, 4, 16]:
        print(f"M={M:2d}, std = {binom_noise_std(cs, M)}")

    print("\n=== quanta_sample_direct (statistical) ===")
    N = 1000000
    flux = 2.0

    y = quanta_sample_direct(flux * np.ones(N), M=np.inf, sigma_read=0.0, rng=rng)
    print(f"Poisson (M=inf), flux={flux}: mean={y.mean():.4f} (theory {flux:.4f}), "
          f"var={y.var():.4f} (theory {flux:.4f})")

    y = quanta_sample_direct(flux * np.ones(N), M=1, sigma_read=0.0, rng=rng)
    p = ppd(flux)
    print(f"Binary  (M=1),   flux={flux}: mean={y.mean():.4f} (theory {p:.4f}), "
          f"var={y.var():.4f} (theory {p*(1-p):.4f})")

    y = quanta_sample_direct(flux * np.ones(N), M=4, sigma_read=0.5, rng=rng)
    print(f"M=4, read=0.5,   flux={flux}: mean={y.mean():.4f}, var={y.var():.4f}")

    print("\n=== quanta_sample_burst (statistical) ===")
    flux = 4.0
    M = 8
    y = quanta_sample_burst(flux * np.ones(N), M=M, sigma_read=0.0, rng=rng)
    p = ppd(flux / M)
    mean_th = M * p
    var_th = M * p * (1 - p)
    print(f"burst  M={M}, flux={flux}: mean={y.mean():.4f} (theory {mean_th:.4f}), "
          f"var={y.var():.4f} (theory {var_th:.4f})")

    y2 = quanta_sample_direct(flux * np.ones(N), M=M, sigma_read=0.0, rng=rng)
    print(f"direct M={M}, flux={flux}: mean={y2.mean():.4f} (theory {flux:.4f}), "
          f"var={y2.var():.4f} (theory {flux:.4f})")