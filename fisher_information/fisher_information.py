import numpy as np
from scipy.stats import poisson
def ppd(f):
    return 1 - np.exp(-f)


def fi_px_gaussian(sigma):
    return sigma ** (-2)


def fi_px_poisson(f):
    return 1 / f

def fi_px_quanta_1bit(f):
    p = ppd(f)
    return (1 - p) / p

def fi_px_quanta_multibit_burst(f, K):
    return (1 / K) * fi_px_quanta_1bit(f / K)

def fi_px_quanta_multibit_direct(f, M):
    f = np.asarray(f, dtype=float)
    eps = np.finfo(float).eps
    f_safe = np.maximum(eps, f)
    u1 = poisson.cdf(M - 2, f_safe)
    u2 = poisson.pmf(M - 1, f_safe)
    u3 = poisson.pmf(M, f_safe)
    u4 = poisson.sf(M - 1, f_safe)
    u4_safe = np.maximum(eps, u4)
    In = (1 / f_safe) * (u1 - (M - 1) * u2 + M * (u3 * (1 + (u2 / u4_safe))))
    return In


if __name__ == "__main__":
    f_test = np.array([0.1, 0.5, 1.0, 5.0, 10.0])
    sigma_test = np.array([0.5, 1.0, 2.0, 3.0, 5.0])
    K = 4
    M = 7
    print(f"Gaussian (sigma={sigma_test}):\n{fi_px_gaussian(sigma_test)}\n")
    print(f"Poisson (f={f_test}):\n{fi_px_poisson(f_test)}\n")
    print(f"Quanta 1-bit (f={f_test}):\n{fi_px_quanta_1bit(f_test)}\n")
    print(f"Quanta Burst (f={f_test}, K={K}):\n{fi_px_quanta_multibit_burst(f_test, K)}\n")
    print(f"Quanta Direct (f={f_test}, M={M}):\n{fi_px_quanta_multibit_direct(f_test, M)}\n")