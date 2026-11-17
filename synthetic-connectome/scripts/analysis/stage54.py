"""Stage 54 summary: learned / taught / always-open slots (results/subject_slot/*_s*.json -> stage54_summary.json),
with Stage 52's plain gated memory (4.9 M words, four passes, full softmax) for reference."""
import glob, json, os
import numpy as np
ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")


def ms(v, pct=True):
    v = np.asarray([x for x in v if x is not None], float) * (100 if pct else 1)
    if len(v) == 0: return None
    return [round(float(v.mean()), 1 if pct else 3), round(float(v.std(ddof=1)), 1 if pct else 3) if len(v) > 1 else None, len(v)]


def main():
    out = {}
    sets = [("plain memory (Stage 52)", os.path.join(ROOT, "grammar_memory", "gru_big_s[0-9].json")),
            ("always-open slot", os.path.join(ROOT, "subject_slot", "open_s[0-9].json")),
            ("learned slot", os.path.join(ROOT, "subject_slot", "learned_s[0-9].json")),
            ("taught slot", os.path.join(ROOT, "subject_slot", "taught_s[0-9].json"))]
    for name, pat in sets:
        runs = [json.load(open(f)) for f in sorted(glob.glob(pat))]
        if not runs: continue
        E2 = [r["E2_controlled"] for r in runs]
        o = dict(networks=len(runs), val_logp_per_word=ms([r["val_logp_per_word"] for r in runs], False),
                 word_order=ms([r["C"]["two_neighbours_swapped"] for r in runs]),
                 blimp_same_pairs=ms([r["D"].get("known_to_stage51_vocab", r["D"]["known_words_only"]) for r in runs]),
                 books_agreement=ms([r["E"]["sensitivity_auc"] for r in runs]), surprise=ms([r["F"]["more_surprised_by_substitute"] for r in runs]))
        for k in ("pp_same_number", "pp_attractor", "rel_same_number", "rel_attractor"):
            o[k] = ms([e[k]["sensitivity_auc"] for e in E2]); o[k + "_plural_right"] = ms([e[k]["plural_subj"] for e in E2])
        if "gates" in runs[0]:
            o["gates"] = {kind: {slot: {w: round(float(np.mean([r["gates"][kind][slot][w] for r in runs])), 2) for w in runs[0]["gates"][kind][slot]}
                                 for slot in runs[0]["gates"][kind]} for kind in ("pp", "rel")}
        out[name] = o
    json.dump(out, open(os.path.join(ROOT, "subject_slot", "stage54_summary.json"), "w"), indent=1)
    keys = ["networks", "val_logp_per_word", "word_order", "blimp_same_pairs", "books_agreement", "surprise", "pp_same_number", "pp_attractor",
            "rel_same_number", "rel_attractor", "rel_attractor_plural_right"]
    for k in keys:
        print(f"{k:28s}", " | ".join(f"{n}: {o.get(k)}" for n, o in out.items()))
    for n, o in out.items():
        if "gates" in o: print(n, "gates rel:", o["gates"]["rel"])


if __name__ == "__main__":
    main()
