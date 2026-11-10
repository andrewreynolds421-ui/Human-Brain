"""A V1-like front end for printed letters (Stage 38).

Simple cells: Gabor receptive fields (MODEL: Gabor functions fit cat simple-cell receptive fields,
Jones & Palmer 1987), 8 orientations, quadrature pairs.  Complex cells: the energy of each pair
(MODEL: Adelson & Bergen 1985), i.e. orientation-tuned and insensitive to the exact phase.  Pooling: the
maximum over 4 x 4 pixel blocks (MODEL: position tolerance by MAX pooling, Riesenhuber & Poggio 1999),
giving 8 orientations x 7 x 7 positions = 392 cells.  Wavelength 6 px, envelope 2.5 px, kernel 9 px:
ASSUMED (a stroke of a 20-px letter is ~2-4 px wide).

The cells' outputs are firing rates; the spiking letter layer receives them as Poisson input spike
trains, exactly as it received pixels before.
"""
import numpy as np
from scipy.ndimage import convolve

N_OR, LAMBDA_PX, SIGMA_PX, KSIZE, POOL = 8, 6.0, 2.5, 9, 4


def gabor(theta, phase, lam=LAMBDA_PX, sigma=SIGMA_PX, size=KSIZE):
    r = np.arange(size) - size // 2
    x, y = np.meshgrid(r, r)
    xr = x * np.cos(theta) + y * np.sin(theta)
    yr = -x * np.sin(theta) + y * np.cos(theta)
    g = np.exp(-(xr ** 2 + yr ** 2) / (2 * sigma ** 2)) * np.cos(2 * np.pi * xr / lam + phase)
    return g - g.mean()


KERNELS = [(gabor(th, 0.0), gabor(th, np.pi / 2)) for th in np.arange(N_OR) * np.pi / N_OR]


def complex_cells(X):
    """X: (n, 28, 28) images in [0, 1] -> (n, 392) complex-cell responses (orientation-major)"""
    X = np.asarray(X, float)
    if X.ndim == 2:
        X = X.reshape(len(X), 28, 28)
    n, h = len(X), X.shape[1] // POOL
    out = np.zeros((n, N_OR, h, h))
    for o, (ke, ko) in enumerate(KERNELS):
        for i in range(n):
            e = np.sqrt(convolve(X[i], ke) ** 2 + convolve(X[i], ko) ** 2)
            out[i, o] = e[:h * POOL, :h * POOL].reshape(h, POOL, h, POOL).max((1, 3))
    return out.reshape(n, -1)


def as_rates(V, total=50.0):
    """input rates (0..1, x the layer's maximum rate) with the same total drive per image - as the
    pixel version scaled each image to the same total ink (Stage 21)"""
    V = np.asarray(V, float)
    return np.clip(V * (total / np.maximum(V.sum(1, keepdims=True), 1e-9)), 0, 1)


SPARSE_Q = 0.75      # ASSUMED: each image keeps the responses above its own 75th percentile (minus that level);
                     # ~23 % of cells clearly active, like the pixel input, and a linear readout keeps 95.9 %
                     # (vs 97.4 % dense) on never-seen families (Stage 38).  Stand-in for V1's output
                     # nonlinearity / normalisation (threshold-linear rates; Heeger 1992 for normalisation).


def sparsify(V, q=SPARSE_Q):
    V = np.asarray(V, float)
    t = np.quantile(V, q, axis=1, keepdims=True)
    return np.maximum(V - t, 0.0)
