#!/usr/bin/env python3
"""Stage 73 data: new books for a grade 4-6 reading year.

Candidates: titles in the GITenberg repository list (gitenberg-dev/gitberg, gitenberg/data/GITenberg_repo_list.tsv;
Project Gutenberg books mirrored on GitHub) that look written for children (for children / boys / girls / young
folks, readers, fairy tales, nature stories, "Our Little ... Cousin", series such as Bobbsey Twins), excluding
every book already in the corpora. Each is downloaded once, its Gutenberg header and licence stripped
(grammar.book_text), checked to be English (share of common English words) and graded with Stage 53's
calibration (Flesch-Kincaid -> reader grade). Kept: grade 4.0-6.9, at least 15,000 words. Written to
/home/claude/corpora/books_year4 (not in the repository); the list with grades goes to data/grades/year4.json.

    python scripts/year4_books.py --max 400
"""
import argparse, json, os, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
import grammar as GR, graded_reading as GD   # noqa: E402
from synth import readability as RD   # noqa: E402
CORP = "/home/claude/corpora"
LIST = "/home/claude/downloads/gitberg_src/gitberg/gitenberg/data/GITenberg_repo_list.tsv"
OUTDIR = os.path.join(CORP, "books_year4")
PAT = re.compile(r"(for-Children|for-Boys|for-Girls|Young-Folks|Little-Folks|Boys-and-Girls|for-Young-People|for-the-Young|Children-s-|Child-s-|"
                 r"Our-Little-|Bobbsey|Rover-Boys|Tom-Swift|Boy-Scouts|Girl-Scouts|Camp-Fire-Girls|Motor-Boys|Uncle-Wiggily|Burgess|"
                 r"Fairy-Tales|Fairy-Book|Nature-Stories|Animal-Stories|Bird-Stories|Fourth-Reader|Fifth-Reader|Sixth-Reader|Told-to-)", re.I)
SKIP = re.compile(r"(Magazine|Vol\.|Volume|Issue|Journal|Astounding|Weekly|Punch|Index|Bulletin|Catalogue|Report)", re.I)
COMMON = set("the and of to a in that it was he his i you with for as on had she her at they be but not have is this by from".split())


def get(url):
    try:
        with urllib.request.urlopen(url, timeout=30) as r: return r.read()
    except Exception: return None


def fetch(item):
    i, n = item; d = os.path.join(OUTDIR, n)
    if os.path.isdir(d) and os.listdir(d): return n
    for f in (f"{i}.txt", f"{i}-0.txt", f"{i}-8.txt"):
        b = get(f"https://raw.githubusercontent.com/GITenberg/{n}/master/{f}")
        if b and len(b) > 60_000:
            os.makedirs(d, exist_ok=True); open(os.path.join(d, f), "wb").write(b); return n
    return None


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--max", type=int, default=400); a = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    have = set()
    for d in ("books", "books_more", "books_graded", "illustrated"):
        for x in os.listdir(os.path.join(CORP, d)): have.add(x.rsplit("_", 1)[-1]); have.add(re.sub(r"_\d+$", "", x).lower())
    rows = [l.rstrip("\n").split("\t") for l in open(LIST)]
    cand = [(i, n) for i, n in rows if PAT.search(n) and not SKIP.search(n) and i not in have and re.sub(r"_\d+$", "", n).lower() not in have
            and not any(t in n for t in GD.TEST_READERS + GD.TEST_OTHER)][:a.max]
    t0 = time.time()
    with ThreadPoolExecutor(8) as ex: got = [n for n in ex.map(fetch, cand) if n]
    print("candidates", len(cand), "downloaded", len(got), round(time.time() - t0), "s", flush=True)
    cal = json.load(open(os.path.join(ROOT, "data", "grades", "catalogue.json")))["calibration"]; keep, drop = [], {}
    for n in got:
        d = os.path.join(OUTDIR, n); txt = GR.book_text(d); words = re.findall(r"[a-z]+", txt.lower())
        if len(words) < 15_000: drop[n] = "short"; continue
        if sum(w in COMMON for w in words) / len(words) < 0.25: drop[n] = "not English"; continue
        fk = RD.fk_grade(txt); g = round(cal["intercept"] + cal["slope"] * fk, 1)
        if not 4.0 <= g < 7.0: drop[n] = f"grade {g}"; continue
        keep.append(dict(name=n, dir=d, fk=round(fk, 2), grade=g, band=GD.band_of(g), words=len(words)))
    json.dump(dict(source="GITenberg mirror of Project Gutenberg (raw.githubusercontent.com/GITenberg)", calibration=cal, kept=keep, dropped=drop),
              open(os.path.join(ROOT, "data", "grades", "year4.json"), "w"), indent=1)
    print("kept", len(keep), "words", sum(k["words"] for k in keep), "by band", {b: sum(k["band"] == b for k in keep) for b in ("4", "5", "6")})
    print("dropped", {r: sum(1 for v in drop.values() if v.startswith(r.split()[0])) for r in ("short", "not English", "grade")})


if __name__ == "__main__":
    main()
