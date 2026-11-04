#!/usr/bin/env python3
"""Render lowercase letters a-z from the installed fonts into 28x28 images (MNIST format),
with random size, position, slant and stroke thickness.  Fonts are split so that test
letters come from typefaces never seen in training.

    python scripts/make_letters.py --per-font 6 --out data/letters.npz
"""
import argparse, glob, os, string
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

LETTERS = string.ascii_lowercase


def usable(path):
    try:
        f = ImageFont.truetype(path, 40)
    except Exception:
        return False
    imgs = []
    for ch in "aegx":
        im = Image.new("L", (64, 64)); ImageDraw.Draw(im).text((10, 0), ch, fill=255, font=f)
        a = np.asarray(im)
        if a.sum() < 2000:
            return False
        imgs.append(a)
    return not np.array_equal(imgs[0], imgs[1])          # glyphs differ: real Latin letters


def render(path, ch, rng):
    size = int(rng.integers(34, 52))
    f = ImageFont.truetype(path, size)
    im = Image.new("L", (96, 96)); d = ImageDraw.Draw(im)
    d.text((20, 10), ch, fill=255, font=f)
    if rng.random() < 0.4:
        im = im.filter(ImageFilter.MaxFilter(3))          # thicker strokes
    shear = rng.uniform(-0.25, 0.25)
    im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * 48, 0, 1, 0), resample=Image.BILINEAR)
    a = np.asarray(im)
    ys, xs = np.nonzero(a > 30)
    if ys.size == 0:
        return None
    crop = im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1))
    w, h = crop.size; s = 20.0 / max(w, h)                 # MNIST: fit in 20x20, place in 28x28
    crop = crop.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    out = Image.new("L", (28, 28))
    out.paste(crop, ((28 - crop.size[0]) // 2 + int(rng.integers(-2, 3)), (28 - crop.size[1]) // 2 + int(rng.integers(-2, 3))))
    return np.asarray(out, np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-font", type=int, default=6)
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0, help="render seed (font split always uses seed 0)")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "..", "data", "letters.npz"))
    a = ap.parse_args()
    paths = sorted(set(glob.glob("/usr/share/fonts/**/*.[ot]tf", recursive=True)))
    fonts = [p for p in paths if usable(p)]
    np.random.default_rng(0).shuffle(fonts)                 # same font split whatever the render seed
    rng = np.random.default_rng(a.seed)
    n_test = int(len(fonts) * a.test_frac)
    split = {"test": fonts[:n_test], "train": fonts[n_test:]}
    out = {}
    for name, fl in split.items():
        X, y = [], []
        for p in fl:
            for k, ch in enumerate(LETTERS):
                for _ in range(a.per_font):
                    img = render(p, ch, rng)
                    if img is not None:
                        X.append(img); y.append(k)
        out[f"X_{name}"] = np.array(X); out[f"y_{name}"] = np.array(y)
        out[f"fonts_{name}"] = np.array([os.path.basename(p) for p in fl])
    np.savez_compressed(a.out, **out)
    print(f"{len(fonts)} usable fonts ({len(split['train'])} train / {n_test} test); "
          f"{len(out['y_train'])} training and {len(out['y_test'])} test images")


if __name__ == "__main__":
    main()
