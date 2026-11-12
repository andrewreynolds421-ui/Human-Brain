"""Pictures for the primer (Stage 45): 34 things the McGuffey First Reader's pictures show (girl, boy, dog,
hen, tree, kite ...), each drawn in four open emoji styles - OpenMoji (CC BY-SA 4.0), Twemoji (CC BY 4.0),
Noto Emoji (Apache 2.0), Fluent Emoji 3D (MIT) - rendered with random size, position, rotation, mirror
image and brightness into 48 x 48 colour images (data/pictures.npz; sources and licences in
data/pictures/README.md).  Styles play the role font families played for letters: a style held out is a
way of drawing the model has never seen.

Seeing a picture: the V1-like complex cells of synth/v1.py on its brightness (8 orientations x 12 x 12),
plus coarse colour-opponent cells (red-green and blue-yellow, each sign, 6 x 6 positions; MODEL stand-in
for double-opponent V1 cells, Conway 2001).
"""
import json, os
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
DATA = os.path.join(ROOT, "data", "pictures.npz")
CONCEPTS = os.path.join(ROOT, "data", "pictures", "concepts.json")
STYLES = ("openmoji", "twemoji", "noto", "fluent3d")


def concepts():
    c = json.load(open(CONCEPTS))
    return list(c), {k: v[1] for k, v in c.items()}


def render(path, rng, size=48):
    from PIL import Image
    im = Image.open(path).convert("RGBA").resize((96, 96), Image.LANCZOS)
    if rng.random() < 0.5:
        im = im.transpose(Image.FLIP_LEFT_RIGHT)
    im = im.rotate(float(rng.uniform(-15, 15)), resample=Image.BILINEAR)
    s = int(round(96 * rng.uniform(0.65, 0.95)))
    im = im.resize((s, s), Image.BILINEAR)
    bg = Image.new("RGBA", (96, 96), (255, 255, 255, 255))
    off = (96 - s) // 2
    bg.alpha_composite(im, (off + int(rng.integers(-6, 7)), off + int(rng.integers(-6, 7))))
    a = np.asarray(bg.convert("RGB").resize((size, size), Image.BILINEAR), float) * rng.uniform(0.85, 1.0)
    return np.clip(a, 0, 255).astype(np.uint8)


def features(X):
    """X: (n, 48, 48, 3) uint8 -> (n, 1152 + 144) cells"""
    from synth.v1 import complex_cells
    X = np.asarray(X, float) / 255.0
    lum = 1.0 - (0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2])      # dark lines on white -> ink
    edges = complex_cells(lum)
    rg = X[..., 0] - X[..., 1]; by = X[..., 2] - 0.5 * (X[..., 0] + X[..., 1])
    n = len(X); col = []
    for ch in (rg, by):
        g = ch.reshape(n, 6, 8, 6, 8).mean((2, 4)).reshape(n, -1)
        col += [np.maximum(g, 0), np.maximum(-g, 0)]
    return np.hstack([edges, 4.0 * np.hstack(col)]).astype(np.float32)


def object_cells(X):
    """position-tolerant picture cells for recognising a picture again (Stage 45): orientation energy at two
    scales (wavelength 6 and 12 px) MAX-pooled over 4 x 4 and 2 x 2 regions, plus colour-opponent cells over
    the same regions (MODEL: larger receptive fields and tolerance to position in V4 / inferotemporal cortex,
    Riesenhuber & Poggio 1999).  Same-style matching from 3 sightings: 71-85 % vs 64-72 % for the V1 cells."""
    from scipy.ndimage import convolve
    from synth.v1 import gabor
    X = np.asarray(X, float) / 255.0
    lum = 1.0 - (0.299 * X[..., 0] + 0.587 * X[..., 1] + 0.114 * X[..., 2])
    n = len(X); out = []
    for lam, sig, size in ((6.0, 2.5, 9), (12.0, 5.0, 17)):
        E = np.zeros((n, 8, 48, 48))
        for o, th in enumerate(np.arange(8) * np.pi / 8):
            ke, ko = gabor(th, 0.0, lam, sig, size), gabor(th, np.pi / 2, lam, sig, size)
            for i in range(n):
                E[i, o] = np.sqrt(convolve(lum[i], ke) ** 2 + convolve(lum[i], ko) ** 2)
        for g in (4, 2):
            b = 48 // g
            out.append(E.reshape(n, 8, g, b, g, b).max((3, 5)).reshape(n, -1))
    rg = X[..., 0] - X[..., 1]; by = X[..., 2] - 0.5 * (X[..., 0] + X[..., 1])
    for g in (4, 2):
        b = 48 // g
        for ch in (rg, by):
            m = ch.reshape(n, g, b, g, b).mean((2, 4)).reshape(n, -1)
            out += [4.0 * np.maximum(m, 0), 4.0 * np.maximum(-m, 0)]
    return np.hstack(out).astype(np.float32)
