#!/usr/bin/env python3
"""Stage 39: lowercase letters rendered as they sit in a line of text.  All letters of a font share one
baseline and one scale (the font's ascender-to-descender span fills the image height; the widest letter
fits the width), so relative height and position - l tall, i short with a dot, g and j hanging below the
line - are kept.  The Stage 38 set scaled every letter on its own to a 20-pixel box, which erased them.
Same 299 families and the same family split as scripts/make_letters_families.py; same slant and stroke
variation; small jitter (size 0.92-1, +/-2 px horizontally, +/-1 px vertically).

    python scripts/make_letters_line.py --height 28 --out data/letters_line.npz
"""
import argparse, json, os, sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_letters import LETTERS, usable                      # noqa: E402
from make_letters_families import GF, SEL                       # noqa: E402

S, ORG, CAN = 48, (60, 40), 200


def glyph(font, text):
    im = Image.new("L", (CAN, CAN)); ImageDraw.Draw(im).text(ORG, text, fill=255, font=font)
    return im


def frame(font):
    """font frame at size S: top of ascenders, bottom of descenders, baseline row, widest letter"""
    a = np.asarray(glyph(font, "bdfhkl")); ys = np.nonzero(a.max(1) > 30)[0]; top = ys.min()
    a = np.asarray(glyph(font, "gjpqy")); bottom = np.nonzero(a.max(1) > 30)[0].max() + 1
    a = np.asarray(glyph(font, "x")); base = np.nonzero(a.max(1) > 30)[0].max() + 1
    wmax = 0
    for ch in LETTERS:
        xs = np.nonzero(np.asarray(glyph(font, ch)).max(0) > 30)[0]
        wmax = max(wmax, xs.max() - xs.min() + 1 if xs.size else 0)
    return int(top), int(bottom), int(base), int(wmax)


def render(font, fr, ch, rng, H, W):
    top, bottom, base, wmax = fr
    im = glyph(font, ch)
    if rng.random() < 0.4:
        im = im.filter(ImageFilter.MaxFilter(3))
    shear = rng.uniform(-0.25, 0.25)                                    # slant about the baseline
    im = im.transform(im.size, Image.AFFINE, (1, shear, -shear * base, 0, 1, 0), resample=Image.BILINEAR)
    a = np.asarray(im); xs = np.nonzero(a.max(0) > 30)[0]
    if xs.size == 0:
        return None
    crop = im.crop((int(xs.min()), top - 2, int(xs.max()) + 1, bottom + 2))
    # one scale per font (letters keep their relative size), allowing for the slant and thick strokes
    sc = min((H - 2) / (bottom - top + 4), (W - 2) / (wmax + 0.25 * (bottom - top) + 2)) * rng.uniform(0.92, 1.0)
    w, h = max(1, round(crop.size[0] * sc)), max(1, round(crop.size[1] * sc))
    crop = crop.resize((w, h), Image.BILINEAR)
    out = Image.new("L", (W, H))
    out.paste(crop, ((W - w) // 2 + int(rng.integers(-2, 3)), (H - h) // 2 + int(rng.integers(-1, 2))))
    return np.asarray(out, np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--height", type=int, default=28)
    ap.add_argument("--width", type=int, default=28)
    ap.add_argument("--per-font", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sel = json.load(open(SEL))
    fams = [s for s in sel if usable(os.path.join(GF, s["dir"], s["file"]))]
    order = np.random.default_rng(0).permutation(len(fams))             # same split as make_letters_families
    split = {"test": order[:60], "val": order[60:90], "train": order[90:]}
    rng = np.random.default_rng(a.seed)
    out = {}
    for name, idx in split.items():
        X, y, f = [], [], []
        for fi, k_f in enumerate(idx):
            font = ImageFont.truetype(os.path.join(GF, fams[k_f]["dir"], fams[k_f]["file"]), S)
            fr = frame(font)
            for k, ch in enumerate(LETTERS):
                for _ in range(a.per_font):
                    img = render(font, fr, ch, rng, a.height, a.width)
                    X.append(img if img is not None else np.zeros((a.height, a.width), np.uint8)); y.append(k); f.append(fi)
        out[f"X_{name}"] = np.array(X); out[f"y_{name}"] = np.array(y); out[f"font_{name}"] = np.array(f)
        out[f"families_{name}"] = np.array([fams[k]["family"] for k in idx])
    np.savez_compressed(a.out, **out)
    print({k: v.shape for k, v in out.items() if k.startswith("X_")})


if __name__ == "__main__":
    main()
