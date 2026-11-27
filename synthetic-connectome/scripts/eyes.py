"""Stage 97: VEGA's own eyes, built from what is known about the eye and V1.

Nothing here is a borrowed vision model.  The parts:

  retina (FIXED, ASSUMED from the literature, nothing learned)
    cones          linear light -> L, M, S cone signals (Hunt-Pointer-Estevez)
    adaptation     each cone divided by (itself + local mean light)  (Naka-Rushton)
    foveation      three 64x64 maps around the fixation point: fovea at full
                   sampling, parafovea at 1/2, periphery at 1/3 (acuity falls
                   with eccentricity; cortex gives the centre most room)
    ganglion cells centre minus surround, then split into ON and OFF cells:
                   midget   one cone type in the centre (L or M), small field
                   parasol  L+M, larger field, ON fields ~20% larger than OFF
                   small bistratified  S against L+M (blue-yellow)
  V1 (LEARNED, no labels)
    simple cells   sparse code of ganglion-cell patches from natural photos
    complex cells  local energy pooling of simple cells (FIXED)
  higher areas (LEARNED)
    two more layers, same weights at every eccentricity, ending in one vector
  words (LEARNED)
    a picture and the words said about it are pulled together (as a child hears
    words while looking); tested by picture choice on photos never seen.

Commands
  prep                     photos -> one array (kept out of the repository)
  v1      --seed N         learn simple cells, measure orientation tuning
  ground  --seed N --cond  train higher areas + words, run the tests
          cond: full | pixels (no retina, no learned V1) | randv1 (V1 not learned)
                | shuffled (captions paired with the wrong photos)
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

P = Path(__file__).resolve().parents[1]
FL = Path(os.environ.get("FLICKR", "/home/claude/corpora/flickr8k"))
OUT = Path(os.environ.get("EYES_OUT", "/home/claude/models/eyes"))
RES = Path(os.environ.get("EYES_RES", P / "results" / "eyes"))
SIDE = 192
NCH = 8          # ganglion channels per eccentricity
NV1 = 96         # simple cells
KS = 9           # simple-cell field, in ganglion cells


def prep():
    from PIL import Image
    d = FL / "images" / "Flicker8k_Dataset"
    split = {}
    for s, f in (("train", "Flickr_8k.trainImages.txt"), ("dev", "Flickr_8k.devImages.txt"),
                 ("test", "Flickr_8k.testImages.txt")):
        for n in (FL / "text" / f).read_text().split():
            split[n] = s
    caps = {}
    for line in (FL / "text" / "Flickr8k.token.txt").read_text().splitlines():
        k, c = line.split("\t")
        caps.setdefault(k.split("#")[0], []).append(c.lower())
    names = sorted(n for n in split if (d / n).exists())
    arr = np.zeros((len(names), SIDE, SIDE, 3), np.uint8)
    for i, n in enumerate(names):
        im = Image.open(d / n).convert("RGB")
        w, h = im.size
        s = SIDE / min(w, h)
        im = im.resize((max(SIDE, round(w * s)), max(SIDE, round(h * s))), Image.BICUBIC)
        w, h = im.size
        l, t = (w - SIDE) // 2, (h - SIDE) // 2
        arr[i] = np.asarray(im.crop((l, t, l + SIDE, t + SIDE)))
    np.save(FL / "photos192.npy", arr)
    json.dump({"names": names, "split": [split[n] for n in names], "captions": [caps[n] for n in names]},
              open(FL / "photos.json", "w"))
    print("photos", arr.shape, {s: sum(1 for n in names if split[n] == s) for s in ("train", "dev", "test")})


def fetch():
    """Research-use photo set, fetched from its public source; never copied into the repository."""
    import urllib.request, zipfile
    src = "https://github.com/jbrownlee/Datasets/releases/download/Flickr8k/"
    FL.mkdir(parents=True, exist_ok=True)
    for z, d in (("Flickr8k_text.zip", "text"), ("Flickr8k_Dataset.zip", "images")):
        if not (FL / d).exists():
            urllib.request.urlretrieve(src + z, FL / z)
            zipfile.ZipFile(FL / z).extractall(FL / d)
            (FL / z).unlink()
    prep()


def load():
    if not (FL / "photos192.npy").exists():
        fetch()
    arr = np.load(FL / "photos192.npy", mmap_mode="r")
    meta = json.load(open(FL / "photos.json"))
    return arr, meta


# --------------------------------------------------------------------------- retina

def gauss(torch, sigma, size=None):
    size = size or int(2 * math.ceil(3 * sigma) + 1)
    x = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-x ** 2 / (2 * sigma ** 2))
    g = g / g.sum()
    return g


def blur(torch, x, sigma):
    """Separable Gaussian blur of (B, C, H, W) with reflected edges."""
    F = torch.nn.functional
    g = gauss(torch, sigma).to(x.device)
    k = g.numel() // 2
    c = x.shape[1]
    x = F.pad(x, (k, k, k, k), mode="reflect")
    x = F.conv2d(x, g.view(1, 1, 1, -1).repeat(c, 1, 1, 1), groups=c)
    return F.conv2d(x, g.view(1, 1, -1, 1).repeat(c, 1, 1, 1), groups=c)


RGB2LMS = [[0.31399, 0.63951, 0.04650],      # Hunt-Pointer-Estevez after sRGB->XYZ (D65)
           [0.15537, 0.75789, 0.08670],
           [0.01775, 0.10945, 0.87257]]


class Retina:
    """Fixed model retina.  Input: uint8 photos (B, 192, 192, 3) and a fixation point.
    Output: three maps (fovea, parafovea, periphery), each (B, 8, 64, 64)."""
    # measured on training photos so that every channel has mean-square 1
    def __init__(self, torch, gains=None):
        self.torch = torch
        self.M = torch.tensor(RGB2LMS)
        self.gains = gains

    def cones(self, img):
        torch = self.torch
        x = (img.float() / 255.0) ** 2.2                       # undo display gamma: linear light
        lms = x @ self.M.T
        return lms.permute(0, 3, 1, 2).contiguous()            # (B, 3, H, W)

    def ganglion(self, lms):
        torch = self.torch
        lum = (lms[:, 0:1] + lms[:, 1:2]) / 2
        mean = blur(torch, lum, 8.0)                           # local light level
        c = lms / (lms + mean + 1e-3)                          # Naka-Rushton adaptation
        L, Mc, S = c[:, 0:1], c[:, 1:2], c[:, 2:3]
        LM = (L + Mc) / 2
        sur = blur(torch, LM, 1.5)
        mL, mM = L - sur, Mc - sur                             # midget: one cone centre, mixed surround
        pon = blur(torch, LM, 1.1) - blur(torch, LM, 3.3)      # parasol ON  (larger field)
        poff = blur(torch, LM, 0.9) - blur(torch, LM, 2.7)     # parasol OFF (smaller field)
        by = blur(torch, S, 1.0) - blur(torch, LM, 1.0)        # small bistratified: blue vs yellow
        r = torch.relu
        return torch.cat([r(mL), r(-mL), r(mM), r(-mM), r(pon), r(-poff), r(by), r(-by)], 1)

    def __call__(self, img, fix=None):
        torch = self.torch
        F = torch.nn.functional
        lms = self.cones(img)
        B = lms.shape[0]
        if fix is None:
            fix = torch.full((B, 2), SIDE // 2, dtype=torch.long)
        pad = 64
        big = F.pad(lms, (pad, pad, pad, pad), mode="reflect")
        fov, par = [], []
        for b in range(B):
            y, x = int(fix[b, 0]) + pad, int(fix[b, 1]) + pad
            fov.append(big[b, :, y - 32:y + 32, x - 32:x + 32])
            par.append(big[b, :, y - 64:y + 64, x - 64:x + 64])
        fov = torch.stack(fov)
        par = F.avg_pool2d(torch.stack(par), 2)
        per = F.avg_pool2d(lms, 3)
        out = [self.ganglion(z) for z in (fov, par, per)]
        if self.gains is not None:
            out = [o * self.gains.view(1, -1, 1, 1) for o in out]
        return out


def retina_gains(torch, arr, idx):
    R = Retina(torch)
    g = torch.tensor(np.asarray(arr[np.sort(idx[:400])]))
    o = torch.cat(R(g), 0)
    return 1.0 / o.pow(2).mean((0, 2, 3)).sqrt()


# --------------------------------------------------------------------------- V1

def gratings(torch, n_or=12, sfs=(0.05, 0.08, 0.12, 0.18, 0.25), n_ph=8, size=64, contrast=0.8):
    """Sine gratings as uint8 grey photos, already at retinal sampling (64x64)."""
    ys, xs = torch.meshgrid(torch.arange(size).float(), torch.arange(size).float(), indexing="ij")
    out = []
    for sf in sfs:
        for o in range(n_or):
            th = math.pi * o / n_or
            for p in range(n_ph):
                ph = 2 * math.pi * p / n_ph
                g = 0.5 + 0.5 * contrast * torch.sin(2 * math.pi * sf * (xs * math.cos(th) + ys * math.sin(th)) + ph)
                out.append(g)
    g = torch.stack(out)                                        # linear light
    return g.view(len(sfs), n_or, n_ph, size, size)


def tuning(torch, R, W, bias):
    """Orientation tuning of each simple cell to gratings passed through the retina.
    Returns selectivity (1 - circular variance), preferred orientation, preferred
    spatial frequency and phase modulation F1/F0 for every cell."""
    F = torch.nn.functional
    G = gratings(torch)
    S, O, Ph = G.shape[:3]
    lin = G.reshape(-1, 64, 64)
    lms = lin.unsqueeze(1).repeat(1, 3, 1, 1)                   # grey: equal cone signals
    gang = R.ganglion(lms)
    if R.gains is not None:
        gang = gang * R.gains.view(1, -1, 1, 1)
    c = 32 - KS // 2
    patch = gang[:, :, c:c + KS, c:c + KS].reshape(len(lin), -1)
    a = torch.relu(patch @ W.T - bias).view(S, O, Ph, -1)       # cells' responses
    mean = a.mean(2)                                            # over phase
    best_sf = mean.max(1).values.argmax(0)                      # per cell
    r = mean[best_sf, :, torch.arange(mean.shape[2])]           # (cells, O)
    th = torch.arange(O).float() * math.pi / O
    vec = torch.complex((r * torch.cos(2 * th)).sum(1), (r * torch.sin(2 * th)).sum(1))
    tot = r.sum(1) + 1e-9
    sel = vec.abs() / tot
    pref = (torch.angle(vec) / 2) % math.pi
    best_o = r.argmax(1)
    ph = a[best_sf, best_o, :, torch.arange(mean.shape[2])]     # (cells, Ph)
    k = torch.arange(Ph).float() * 2 * math.pi / Ph
    f1 = 2 * torch.complex((ph * torch.cos(k)).mean(1), (ph * torch.sin(k)).mean(1)).abs()
    f0 = ph.mean(1) + 1e-9
    live = r.max(1).values > 1e-4
    return {"sel": sel, "pref": pref, "sf": best_sf, "f1f0": f1 / f0, "live": live}


def patches(torch, R, arr, idx, rng, n):
    ii = np.sort(rng.choice(idx, 32, replace=False))
    img = torch.tensor(np.asarray(arr[ii]))
    fix = torch.tensor(rng.integers(48, SIDE - 48, (32, 2)))
    maps = torch.cat(R(img, fix), 0)                            # (96, 8, 64, 64)
    b = torch.tensor(rng.integers(0, maps.shape[0], n))
    y = torch.tensor(rng.integers(0, 64 - KS, n))
    x = torch.tensor(rng.integers(0, 64 - KS, n))
    ar = torch.arange(KS)
    pt = maps[b[:, None, None], :, (y[:, None] + ar)[:, :, None], (x[:, None] + ar)[:, None, :]]
    return pt.permute(0, 3, 1, 2).reshape(n, -1)                # (n, 8*9*9)


def learn_v1(seed, steps, lam=float(os.environ.get('V1_RATE', 0.08))):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    arr, meta = load()
    idx = np.array([i for i, s in enumerate(meta["split"]) if s == "train"])
    gains = retina_gains(torch, arr, idx)
    R = Retina(torch, gains)
    D = NCH * KS * KS
    W = torch.nn.Parameter(torch.randn(NV1, D) * 0.1)
    b = torch.zeros(NV1)
    c0 = torch.nn.Parameter(torch.zeros(D))                    # resting level of the ganglion cells
    opt = torch.optim.Adam([W, c0], 2e-3)
    W0 = torch.nn.functional.normalize(W.detach().clone(), dim=1)
    t0 = time.time()
    for step in range(steps + 1):
        x = patches(torch, R, arr, idx, rng, 512)
        Wn = torch.nn.functional.normalize(W, dim=1)
        a = torch.relu(x @ Wn.T - b)
        rec = a @ Wn + c0
        err = ((x - rec) ** 2).sum(1).mean()
        loss = err
        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                                   # homeostasis: each cell fires for ~lam of patches
            b += 0.05 * ((a > 0).float().mean(0) - lam) * (x @ Wn.T).std()
        if step % 500 == 0:
            print(f"v1 {step} err {err.item():.3f} of {(x**2).sum(1).mean().item():.3f} "
                  f"active {(a > 0).float().mean().item():.3f} {time.time()-t0:.0f}s", flush=True)
    Wn = torch.nn.functional.normalize(W.detach(), dim=1)
    bias = b.clone()
    out = {}
    xs = torch.cat([patches(torch, R, arr, idx, rng, 2000) for _ in range(5)])
    b0 = torch.quantile(xs @ W0.T, 1 - lam, dim=0)              # same firing rate for the unlearned cells
    out["rate"] = lam
    for name, w, bb in (("learned", Wn, bias), ("random", W0, b0)):
        t = tuning(torch, R, w, bb)
        live = t["live"]
        sel = t["sel"][live]
        out[name] = {"live": int(live.sum()), "median_selectivity": float(sel.median()) if len(sel) else 0.0,
                     "share_selective_over_0.3": float((sel > 0.3).float().mean()) if len(sel) else 0.0,
                     "median_f1f0": float(t["f1f0"][live].median()) if len(sel) else 0.0,
                     "pref_hist_6": torch.histc(t["pref"][live], 6, 0, math.pi).int().tolist(),
                     "sf_hist": torch.bincount(t["sf"][live], minlength=5).tolist()}
        print(name, out[name], flush=True)
    # colour: share of each cell's weight on the colour-opponent channels
    wv = Wn.view(NV1, NCH, KS, KS)
    e = wv.pow(2).sum((2, 3))
    out["weight_share"] = {"midget": float(e[:, :4].sum(1).mean()), "parasol": float(e[:, 4:6].sum(1).mean()),
                           "blue_yellow": float(e[:, 6:].sum(1).mean())}
    lm = wv[:, 0] - wv[:, 1] - wv[:, 2] + wv[:, 3]              # L against M
    out["red_green_cells"] = int((lm.pow(2).sum((1, 2)) > 0.25).sum())
    out["blue_yellow_cells"] = int((e[:, 6:].sum(1) > 0.5).sum())
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({"W": Wn, "b": bias, "gains": gains, "W0": W0, "b0": b0}, OUT / f"v1_s{seed}.pt")
    RES.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(RES / f"v1_s{seed}.json", "w"), indent=1)
    return out


# --------------------------------------------------------------------------- higher areas + words

STOP = set("a an the of in on at to and is are with for his her its their by as from into up down near "
           "while it this that there be has have one two three who which over through front back next other "
           "some very being".split())


def word_table():
    d = np.load(P / "data" / "semantic_vectors.npz", allow_pickle=True)
    return {w: i for i, w in enumerate(d["words"])}, d["vectors"].astype(np.float32)


def tokens(c):
    return [w for w in c.replace(".", " ").replace(",", " ").replace('"', " ").split() if w.isalpha()]


def build(torch, cond, v1, n_words, sem):
    nn = torch.nn
    F = nn.functional

    class Eyes(nn.Module):
        def __init__(s):
            super().__init__()
            if cond == "pixels":
                s.c0 = nn.Conv2d(3, NV1, KS, stride=2)
            else:
                s.register_buffer("W", (v1["W0"] if cond == "randv1" else v1["W"]).view(NV1, NCH, KS, KS).clone())
                s.register_buffer("b", (v1["b0"] if cond == "randv1" else v1["b"]).clone())
            s.c1 = nn.Conv2d(NV1, 128, 3, padding=1)
            s.c2 = nn.Conv2d(128, 256, 3, padding=1)
            s.ecc = nn.Parameter(torch.zeros(3, 256))           # which eccentricity a map came from
            s.out = nn.Linear(256 * 3, 256)
            s.drop = nn.Dropout(0.3)
            s.sem = nn.Embedding.from_pretrained(sem, freeze=True)
            s.wmap = nn.Sequential(nn.Linear(sem.shape[1], 512), nn.ReLU(), nn.Linear(512, 256))
            s.temp = nn.Parameter(torch.tensor(2.5))

        def see(s, maps):
            vs = []
            for k, m in enumerate(maps):
                if cond == "pixels":
                    simple = torch.relu(s.c0(m))
                else:
                    simple = torch.relu(F.conv2d(m, s.W, stride=2) - s.b.view(1, -1, 1, 1))
                cx = torch.sqrt(F.avg_pool2d(simple ** 2, 2) + 1e-6)      # complex cells: local energy
                cx = cx / (cx.mean((1, 2, 3), keepdim=True) + 1e-3)       # contrast gain control
                h = F.max_pool2d(torch.relu(s.c1(cx)), 2)
                h = torch.relu(s.c2(h))
                vs.append(h.mean((2, 3)) + s.ecc[k])
            return F.normalize(s.out(s.drop(torch.cat(vs, 1))), dim=1)

        def word(s, ids):
            return F.normalize(s.wmap(s.sem(ids)), dim=1)

        def say(s, ids, mask):
            e = s.wmap(s.sem(ids)) * mask.unsqueeze(2)
            return F.normalize(e.sum(1) / mask.sum(1, keepdim=True).clamp(min=1), dim=1)

    return Eyes()


def ground(seed, cond, epochs, limit=0):
    import torch
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    F = torch.nn.functional
    arr, meta = load()
    split = np.array(meta["split"])
    tr = np.where(split == "train")[0]
    if limit:
        tr = tr[:limit]
    te = np.where(split == "test")[0]
    dv = np.where(split == "dev")[0]
    w2i, vec = word_table()
    v1 = torch.load(OUT / f"v1_s{seed}.pt") if (OUT / f"v1_s{seed}.pt").exists() else None
    if v1 is None:
        learn_v1(seed, int(os.environ.get("V1_STEPS", 6000)))
        v1 = torch.load(OUT / f"v1_s{seed}.pt")
    R = Retina(torch, v1["gains"])

    # words: content words VEGA already knows from its reading
    caps = [[[w2i[w] for w in tokens(c) if w in w2i and w not in STOP] for c in cs] for cs in meta["captions"]]
    count = {}
    for i in tr:
        for w in {w for c in caps[i] for w in c}:
            count[w] = count.get(w, 0) + 1
    words = np.array(d_words := sorted(w for w, n in count.items() if n >= 20))
    inv = {i: w for w, i in w2i.items()}
    # words held out of all teaching: VEGA must find their pictures from meaning alone
    cand = [w for w in d_words if 40 <= count[w] <= 400]
    held = set(np.random.default_rng(0).choice(cand, 30, replace=False).tolist())
    print("words", len(words), "held out", sorted(inv[w] for w in held), flush=True)
    has = {}
    for i in range(len(caps)):
        ws = {w for c in caps[i] for w in c}
        nw = {w for w in ws if sum(w in c for c in caps[i]) >= 2}       # said in 2+ of 5 captions
        has[i] = (ws, nw)

    sem = torch.tensor(vec)
    sem = sem / sem.norm(dim=1, keepdim=True).clamp(min=1e-6)
    net = build(torch, cond, v1, len(w2i), sem)
    opt = torch.optim.AdamW([p for p in net.parameters() if p.requires_grad], 1e-3, weight_decay=0.05)
    total = epochs * (len(tr) // 64)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 2e-3, total_steps=total, pct_start=0.1)

    def maps_of(ii, train):
        order = np.argsort(ii)
        img = torch.tensor(np.asarray(arr[ii[order]]))[np.argsort(order)]
        if train:
            flip = torch.rand(len(ii)) < 0.5
            img = torch.where(flip.view(-1, 1, 1, 1), img.flip(2), img)
            fix = torch.tensor(rng.integers(64, SIDE - 64, (len(ii), 2)))     # where the eye lands
        else:
            fix = None
        if cond == "pixels":
            x = (img.float() / 255.0).permute(0, 3, 1, 2)
            fx = fix if fix is not None else torch.full((len(ii), 2), SIDE // 2)
            big = F.pad(x, (64, 64, 64, 64), mode="reflect")
            fov = torch.stack([big[b, :, int(fx[b, 0]) + 32:int(fx[b, 0]) + 96, int(fx[b, 1]) + 32:int(fx[b, 1]) + 96]
                               for b in range(len(ii))])
            par = F.avg_pool2d(torch.stack([big[b, :, int(fx[b, 0]):int(fx[b, 0]) + 128, int(fx[b, 1]):int(fx[b, 1]) + 128]
                                            for b in range(len(ii))]), 2)
            return [fov - 0.5, par - 0.5, F.avg_pool2d(x, 3) - 0.5]
        return R(img, fix)

    def cap_batch(ii, train):
        rows = []
        for i in ii:
            c = caps[i][rng.integers(5)] if train else caps[i][0]
            c = [w for w in c if w not in held]
            rows.append(c or [w2i["a"]])
        L = max(len(r) for r in rows)
        ids = torch.zeros(len(rows), L, dtype=torch.long)
        mask = torch.zeros(len(rows), L)
        for k, r in enumerate(rows):
            ids[k, :len(r)] = torch.tensor(r)
            mask[k, :len(r)] = 1
        return ids, mask

    pair = tr.copy()
    if cond == "shuffled":
        pair = rng.permutation(tr)                              # each photo hears another photo's words
    t0 = time.time()
    step = 0
    for ep in range(epochs):
        net.train()
        perm = rng.permutation(len(tr))
        for k in range(0, len(perm) - 63, 64):
            j = perm[k:k + 64]
            v = net.see(maps_of(tr[j], True))
            ids, mask = cap_batch(pair[j], True)
            c = net.say(ids, mask)
            lg = net.temp.exp().clamp(max=50) * v @ c.T
            y = torch.arange(len(j))
            loss = (F.cross_entropy(lg, y) + F.cross_entropy(lg.T, y)) / 2
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
            if step % 20 == 0:
                print(f"ep {ep} step {step}/{total} loss {loss.item():.3f} {time.time()-t0:.0f}s", flush=True)
    net.eval()

    def see_all(ii):
        with torch.no_grad():
            return torch.cat([net.see(maps_of(ii[k:k + 100], False)) for k in range(0, len(ii), 100)])

    def picture_choice(ii, V, wordset, n=4, trials=4000, strict=True):
        """A word is said; choose its photo among n unseen photos.  The right photo has the
        word in 2+ of its 5 captions; the wrong ones in none."""
        g = np.random.default_rng(1)
        ok = [w for w in wordset if sum(w in has[i][1] for i in ii) >= 3]
        hit = tot = 0
        per = {}
        with torch.no_grad():
            wv = {w: net.word(torch.tensor([w]))[0] for w in ok}
        pos = {w: [k for k, i in enumerate(ii) if w in has[i][1]] for w in ok}
        neg = {w: [k for k, i in enumerate(ii) if w not in has[i][0]] for w in ok}
        for _ in range(trials):
            w = ok[g.integers(len(ok))]
            a = pos[w][g.integers(len(pos[w]))]
            others = g.choice(neg[w], n - 1, replace=False)
            sc = V[[a] + list(others)] @ wv[w]
            h = int(sc.argmax() == 0)
            hit += h; tot += 1
            p = per.setdefault(inv[w], [0, 0]); p[0] += h; p[1] += 1
        return hit / tot, len(ok), per

    def sentence_choice(ii, V, n=10):
        """A whole caption is said; choose its photo among n unseen photos (and the reverse)."""
        g = np.random.default_rng(2)
        with torch.no_grad():
            ids, mask = cap_batch(ii, False)
            C = net.say(ids, mask)
        S = V @ C.T
        hit = hit2 = 0
        T = 3000
        for _ in range(T):
            pick = g.choice(len(ii), n, replace=False)
            s = S[pick][:, pick]
            hit += int(s[:, 0].argmax() == 0)
            hit2 += int(s[0].argmax() == 0)
        return hit / T, hit2 / T

    res = {"config": {"seed": seed, "cond": cond, "epochs": epochs, "train_photos": int(len(tr)),
                      "words": int(len(words)), "minutes": round((time.time() - t0) / 60, 1)}}
    Vt = see_all(te)
    taught = [w for w in d_words if w not in held]
    acc, nw, per = picture_choice(te, Vt, taught)
    res["picture_choice_taught_words"] = {"accuracy": acc, "chance": 0.25, "words": nw}
    acc_h, nh, per_h = picture_choice(te, Vt, sorted(held))
    res["picture_choice_untaught_words"] = {"accuracy": acc_h, "chance": 0.25, "words": nh,
                                           "per_word": {w: round(a / b, 2) for w, (a, b) in sorted(per_h.items())}}
    s1, s2 = sentence_choice(te, Vt)
    res["caption_to_photo_of_10"] = {"accuracy": s1, "chance": 0.1}
    res["photo_to_caption_of_10"] = {"accuracy": s2, "chance": 0.1}
    best = sorted(((a / b, w) for w, (a, b) in per.items() if b >= 15), reverse=True)
    res["best_words"] = [(w, round(a, 2)) for a, w in best[:15]]
    res["worst_words"] = [(w, round(a, 2)) for a, w in best[-15:]]
    print(json.dumps(res, indent=1), flush=True)
    RES.mkdir(parents=True, exist_ok=True)
    tag = os.environ.get("EYES_TAG", "")
    json.dump(res, open(RES / f"ground_{cond}{tag}_s{seed}.json", "w"), indent=1)
    OUT.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.half() if v.is_floating_point() else v for k, v in net.state_dict().items() if not k.startswith("sem")},
               OUT / f"eyes_{cond}{tag}_s{seed}.pt")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prep", "v1", "ground"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", default="full")
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "prep":
        prep()
    elif a.cmd == "v1":
        learn_v1(a.seed, a.steps)
    else:
        ground(a.seed, a.cond, a.epochs, a.limit)
