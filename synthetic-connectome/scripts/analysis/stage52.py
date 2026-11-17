"""Stage 52 summary: the gated memory by amount of text read - half the 32 books (0.35 M words), the 32 books
(0.7 M), half the big corpus (2.4 M), the big corpus (4.9 M). results/grammar_memory -> stage52_summary.json"""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "grammar_memory")
RUNS = (("0.35 M", "gru_f0.5"), ("0.7 M", "gru"), ("2.4 M", "gru_big_f0.5"), ("4.9 M", "gru_big"))


def ms(v, pct=True):
    v = np.asarray([x for x in v if x is not None], float) * (100 if pct else 1)
    if len(v) == 0: return None
    return [round(float(v.mean()), 1 if pct else 3), round(float(v.std(ddof=1)), 1 if pct else 3) if len(v) > 1 else None, len(v)]


def main():
    out = {}
    for label, name in RUNS:
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, f"{name}_s[0-9].json")))]
        if not runs: continue
        E2 = [r["E2_controlled"] for r in runs]
        o = dict(file=name, networks=len(runs), train_words=runs[0].get("train_words"),
                 val_logp_per_word=ms([r["val_logp_per_word"] for r in runs], False),
                 C_swapped=ms([r["C"]["two_neighbours_swapped"] for r in runs]),
                 D_blimp_stage51_vocab=ms([r["D"].get("known_to_stage51_vocab", r["D"]["known_words_only"]) for r in runs]),
                 D_blimp_all=ms([r["D"]["all"] for r in runs]),
                 E_books_sensitivity=ms([r["E"]["sensitivity_auc"] for r in runs]),
                 E_books_right=ms([r["E"]["all"] for r in runs]),
                 E2_sensitivity=ms([e["sensitivity_auc"] for e in E2]),
                 F_surprise=ms([r["F"]["more_surprised_by_substitute"] for r in runs]))
        for k in ("pp_same_number", "pp_attractor", "rel_same_number", "rel_attractor"):
            o[f"E2_{k}"] = ms([e[k]["sensitivity_auc"] for e in E2])
            o[f"E2_{k}_plural_right"] = ms([e[k]["plural_subj"] for e in E2])
        out[label] = o
    json.dump(out, open(os.path.join(R, "stage52_summary.json"), "w"), indent=1)
    for k in next(iter(out.values())):
        if k in ("file",): continue
        print(f"{k:32s}", "  ".join(f"{lab}: {o.get(k)}" for lab, o in out.items()))


if __name__ == "__main__":
    main()
