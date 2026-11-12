#!/usr/bin/env python3
"""Stage 46: reading code and web pages as text.

Material: 28 open-source repositories (shallow clones in /home/claude/code_src; list in REPOS), files by
extension in 13 kinds.  Split by REPOSITORY, as fonts were split by family: TEST_REPOS are never read.

Token units: the model reads characters; every new token (an identifier, a number, a symbol, an indent)
recruits a unit the first time it is seen (exact - text, unlike print, arrives without noise).
Identifiers also activate units for their pieces (snake_case / camelCase split), so `read_file` and
`readFile` share `read` and `file`.

A. What kind of file.  13 kind units read the token units of a 40-line excerpt (log counts, unit length)
   and learn by the delta rule, taught by the file's extension - the label that comes with a file, as the
   word under a picture.  One pass; eta FITTED on validation files (training repos, other files).
   References: a fitted linear readout on the same token units (many passes), and on character 3-grams.
B. What a function is for.  Python functions with docstrings.  While reading the training repositories,
   word units of the docstring and token units of the code are linked by counting, function by function
   (delta P, as words and pictures in Stage 45).  Test: the docstring of a function from a never-read
   repository; which of 10 functions (it and 9 others from the same repositories) does it describe?
   Score = mean over the docstring's words of the strongest link to any piece in the function's code.
   Baselines: chance 10 %; surface overlap (share of docstring words that occur among the code's pieces).

    python scripts/read_code.py --seed 1   ->  results/read_code/s1.json
"""
import argparse, ast, json, os, re, sys
import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
SRC = "/home/claude/code_src"
OUT = os.path.join(ROOT, "results", "read_code")
EXT = {".py": "python", ".js": "javascript", ".mjs": "javascript", ".ts": "typescript", ".go": "go", ".rs": "rust",
       ".c": "c", ".h": "c", ".html": "html", ".css": "css", ".md": "markdown", ".sh": "shell", ".zsh": "shell",
       ".json": "json", ".yml": "yaml", ".yaml": "yaml", ".toml": "toml"}
KINDS = sorted(set(EXT.values()))
TEST_REPOS = {"encode_httpx", "psf_requests", "expressjs_express", "sindresorhus_got", "spf13_cobra", "BurntSushi_ripgrep",
              "jqlang_jq", "mdn_learning-area", "nvm-sh_nvm", "h5bp_html5-boilerplate", "Textualize_rich"}
EXTRA = {"python_cpython"}       # Python standard library (Lib/, no tests): extra reading for part B only
TOKEN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|\d+(?:\.\d+)?|\S")
STOP = set("a an the of to and or in on for is are be by with this that it as at from if not no its can will "
           "which when should may into then than also any all each but so use used using we you your our".split())


def tokens(text):
    out = []
    for line in text.split("\n"):
        ind = len(line) - len(line.lstrip(" \t"))
        if ind and line.strip():
            out.append("<indent>")
        out += TOKEN.findall(line)
        out.append("<nl>")
    return out


def pieces(tok):
    if not re.match(r"[A-Za-z_]", tok):
        return []
    p = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", tok).lower().split("_")
    return [x for x in p if len(x) > 1]


class TokenUnits:
    def __init__(self):
        self.id = {}

    def unit(self, t, learn=True):
        if t not in self.id:
            if not learn:
                return None
            self.id[t] = len(self.id)
        return self.id[t]

    def read(self, toks, learn=True):
        act = {}
        for t in toks:
            for x in [t] + ["~" + p for p in pieces(t)]:
                u = self.unit(x, learn)
                if u is not None:
                    act[u] = act.get(u, 0) + 1
        return act


def files(split, extra=False):
    out = []
    for repo in sorted(os.listdir(SRC)):
        p = os.path.join(SRC, repo)
        if not os.path.isdir(p) or (repo in TEST_REPOS) != (split == "test") or ((repo in EXTRA) != extra and repo in EXTRA):
            continue
        for root, _, fs in os.walk(p):
            if "/.git" in root or "node_modules" in root or "/test" in root:
                continue
            for f in fs:
                e = os.path.splitext(f)[1]; fp = os.path.join(root, f)
                if e in EXT and 200 < os.path.getsize(fp) < 400_000:
                    out.append((repo, fp, EXT[e]))
    return out


def excerpt(fp, rng, n=40):
    try:
        lines = open(fp, encoding="utf-8", errors="ignore").read().split("\n")
    except OSError:
        return ""
    s = int(rng.integers(0, max(1, len(lines) - n)))
    return "\n".join(lines[s:s + n])


def vec(act, n):
    v = np.zeros(n)
    for u, c in act.items():
        if u < n:
            v[u] = np.log1p(c)
    return v / (np.linalg.norm(v) + 1e-9)


# ---------------------------------------------------------------------------------------------- A
def part_a(a, rng):
    tr, te = files("train"), files("test")
    perm = rng.permutation(len(tr)); nval = len(tr) // 6
    val = [tr[i] for i in perm[:nval]]; tr = [tr[i] for i in perm[nval:]]
    TU = TokenUnits()
    Xtr_act = [TU.read(tokens(excerpt(fp, rng))) for _, fp, _ in tr]
    n = len(TU.id)
    Xtr = np.array([vec(x, n) for x in Xtr_act]); ytr = np.array([KINDS.index(k) for *_, k in tr])
    Xva = np.array([vec(TU.read(tokens(excerpt(fp, rng)), learn=False), n) for _, fp, _ in val]); yva = np.array([KINDS.index(k) for *_, k in val])
    tex = [excerpt(fp, rng) for _, fp, _ in te]
    Xte = np.array([vec(TU.read(tokens(t), learn=False), n) for t in tex]); yte = np.array([KINDS.index(k) for *_, k in te])

    def delta(eta):
        W = np.zeros((len(KINDS), n)); b = np.zeros(len(KINDS))
        for x, y in zip(Xtr, ytr):
            z = W @ x + b; z -= z.max(); p = np.exp(z); p /= p.sum()
            err = -p; err[y] += 1
            W += eta * np.outer(err, x); b += 0.1 * eta * err
        return lambda X: (X @ W.T + b).argmax(1)
    best = max(((np.mean(delta(e)(Xva) == yva), e) for e in (0.3, 1.0, 3.0, 10.0)))
    pred = delta(best[1])(Xte)
    from sklearn.linear_model import LogisticRegression
    from sklearn.feature_extraction.text import TfidfVectorizer
    vz = TfidfVectorizer(analyzer="char", ngram_range=(3, 3), min_df=2, sublinear_tf=True, max_features=50000)
    trtext = [excerpt(fp, np.random.default_rng(7)) for _, fp, _ in tr]; Ttr = vz.fit_transform(trtext)
    vtext = [excerpt(fp, np.random.default_rng(8)) for _, fp, _ in val]
    # references: C chosen on validation files, as eta was; also the accuracy each C would give on the test
    refs = {}
    for name, Xa, Xv, Xt in (("token_units", Xtr, Xva, Xte), ("char_3grams", Ttr, vz.transform(vtext), vz.transform(tex))):
        fits = {C: LogisticRegression(C=C, max_iter=3000).fit(Xa, ytr) for C in (0.1, 1.0, 10.0)}
        Cb = max(fits, key=lambda C: fits[C].score(Xv, yva))
        refs[name] = dict(C=Cb, test=round(float(fits[Cb].score(Xt, yte)), 4), test_balanced=None,
                          test_by_C={str(C): round(float(m.score(Xt, yte)), 4) for C, m in fits.items()}, model=fits[Cb], X=Xt)
    per = {KINDS[k]: round(float(np.mean(pred[yte == k] == k)), 3) for k in range(len(KINDS)) if np.any(yte == k)}
    conf = {}
    for t_, p_ in zip(yte, pred):
        if t_ != p_:
            conf[f"{KINDS[t_]}->{KINDS[p_]}"] = conf.get(f"{KINDS[t_]}->{KINDS[p_]}", 0) + 1
    # balanced accuracy (each kind counts equally: test files are dominated by a few kinds)
    bal = lambda p: float(np.mean([np.mean(p[yte == k] == k) for k in range(len(KINDS)) if np.any(yte == k)]))
    return dict(train_files=len(tr), test_files=len(te), token_units=n, eta=best[1], validation=round(float(best[0]), 4),
                kind_units=round(float(np.mean(pred == yte)), 4), kind_units_balanced=round(bal(pred), 4),
                references={k: dict(C=v["C"], test=v["test"], test_balanced=round(bal(v["model"].predict(v["X"])), 4), test_by_C=v["test_by_C"]) for k, v in refs.items()},
                test_files_per_kind={KINDS[k]: int(np.sum(yte == k)) for k in range(len(KINDS)) if np.any(yte == k)},
                per_kind=per, top_confusions=sorted(conf.items(), key=lambda x: -x[1])[:8])


# ---------------------------------------------------------------------------------------------- B
def functions(split, extra=False):
    out = []
    for repo, fp, kind in files(split, extra):
        if kind != "python":
            continue
        try:
            src = open(fp, encoding="utf-8", errors="ignore").read(); tree = ast.parse(src)
        except (SyntaxError, ValueError):
            continue
        lines = src.split("\n")
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                doc = ast.get_docstring(node)
                if not doc or len(doc.split()) < 4 or not node.body:
                    continue
                body = node.body[1:] if isinstance(node.body[0], ast.Expr) else node.body
                if not body:
                    continue
                code = "\n".join(lines[body[0].lineno - 1: getattr(node, "end_lineno", body[-1].lineno)])
                first = doc.strip().split("\n\n")[0]
                words = [w for w in re.findall(r"[a-z]+", first.lower()) if w not in STOP and len(w) > 1]
                # the function's name is withheld: the question is whether the BODY shows what it is for
                cp = set()
                for t in tokens(code):
                    cp |= set(pieces(t))
                if len(words) >= 2 and len(cp) >= 3:
                    out.append(dict(repo=repo, name=node.name, words=sorted(set(words)), code=sorted(cp)))
    return out


def part_b(a, rng, extra=False):
    tr, te = functions("train") + (functions("train", True) if extra else []), functions("test")
    # links between docstring words and code pieces, counted function by function (delta P)
    nw, nc, nwc = {}, {}, {}
    for f in tr:
        for w in f["words"]:
            nw[w] = nw.get(w, 0) + 1
        for c in f["code"]:
            nc[c] = nc.get(c, 0) + 1
        for w in f["words"]:
            for c in f["code"]:
                nwc[(w, c)] = nwc.get((w, c), 0) + 1
    N = len(tr)

    def link(w, c):
        n_w = nw.get(w, 0)
        if n_w == 0 or n_w == N:
            return 0.0
        return nwc.get((w, c), 0) / n_w - (nc.get(c, 0) - nwc.get((w, c), 0)) / (N - n_w)

    def score_links(f, g):
        ws = [w for w in f["words"] if w in nw]
        return np.mean([max(link(w, c) for c in g["code"]) for w in ws]) if ws else 0.0

    def score_surface(f, g):
        cs = set(g["code"])
        return np.mean([any(w == c or (len(w) > 3 and (c.startswith(w[:-1]) or w.startswith(c[:-1]))) for c in cs) for w in f["words"]])

    res = dict(train_functions=N, test_functions=len(te), docstring_words_with_links=len(nw))
    for name, sc in (("links_learned_from_reading", score_links), ("surface_overlap", score_surface),
                     ("both", lambda f, g: score_links(f, g) + 0.5 * score_surface(f, g))):
        hits = []
        r2 = np.random.default_rng(a.seed + 77)
        for i, f in enumerate(te):
            others = r2.choice([j for j in range(len(te)) if j != i], 9, replace=False)
            cand = [i] + list(others)
            s = np.array([sc(f, te[j]) for j in cand])
            hits.append(float(s[0] > np.max(s[1:])) + 0.5 * float(s[0] == np.max(s[1:]) and np.sum(s == s[0]) == 2))
        res[name] = round(float(np.mean(hits)), 4)
    # examples of learned links
    ex = {}
    for w in ("request", "header", "file", "path", "string", "list", "error", "close", "url", "response"):
        if w in nw:
            cs = sorted(nc, key=lambda c: -link(w, c))[:5]
            ex[w] = [(c, round(link(w, c), 2)) for c in cs]
    res["strongest_links"] = ex
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    res = dict(seed=a.seed, kinds=KINDS, test_repos=sorted(TEST_REPOS))
    res["A_kind_of_file"] = part_a(a, rng); print("A", {k: v for k, v in res["A_kind_of_file"].items() if not isinstance(v, (dict, list))}, flush=True)
    res["B_what_a_function_is_for"] = part_b(a, rng); print("B", {k: v for k, v in res["B_what_a_function_is_for"].items() if k != "strongest_links"}, flush=True)
    res["B_after_also_reading_the_standard_library"] = part_b(a, rng, True)
    print("B+", {k: v for k, v in res["B_after_also_reading_the_standard_library"].items() if k != "strongest_links"}, flush=True)
    json.dump(res, open(os.path.join(OUT, f"s{a.seed}.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
