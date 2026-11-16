"""Stage 51 summary: trigram (Stage 50) vs simple recurrent vs gated recurrent memory, four networks each,
plus the gated network trained on half the books. results/grammar_memory/*_s*.json -> stage51_summary.json"""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "grammar_memory")


def ms(v, pct=True):
    v = np.asarray([x for x in v if x is not None], float) * (100 if pct else 1)
    if len(v) == 0: return None
    return [round(float(v.mean()), 1 if pct else 3), round(float(v.std(ddof=1)), 1 if pct else 3) if len(v) > 1 else None, len(v)]


def main():
    out = {}
    for name in ("trigram", "srn", "gru", "gru_f0.5"):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, f"{name}_s[0-9].json")))]
        if not runs: continue
        E2 = [r["E2_controlled"] for r in runs]
        o = dict(networks=len(runs),
                 val_logp_per_word=ms([r["val_logp_per_word"] for r in runs], False),
                 C_swapped=ms([r["C"]["two_neighbours_swapped"] for r in runs]), C_shuffled=ms([r["C"]["shuffled"] for r in runs]),
                 D_blimp_all=ms([r["D"]["all"] for r in runs]), D_blimp_known=ms([r["D"]["known_words_only"] for r in runs]),
                 E_books_right=ms([r["E"]["all"] for r in runs]), E_books_sensitivity=ms([r["E"]["sensitivity_auc"] for r in runs]),
                 E_books_sensitivity_dist1plus=ms([r["E"]["sensitivity_auc_distance_1plus"] for r in runs]),
                 E_books_attractor=ms([r["E"]["with_attractor"][0] for r in runs]),
                 E2_right=ms([e["all"] for e in E2]), E2_sensitivity=ms([e["sensitivity_auc"] for e in E2]),
                 F_surprise=ms([r["F"]["more_surprised_by_substitute"] for r in runs]), F_extra_nats=ms([r["F"]["mean_extra_surprise_nats"] for r in runs], False))
        for k in ("pp_same_number", "pp_attractor", "rel_same_number", "rel_attractor"):
            o[f"E2_{k}"] = dict(right=ms([e[k]["right"] for e in E2]), singular_subj=ms([e[k]["singular_subj"] for e in E2]),
                                plural_subj=ms([e[k]["plural_subj"] for e in E2]), sensitivity=ms([e[k]["sensitivity_auc"] for e in E2]))
        out[name] = o
    json.dump(out, open(os.path.join(R, "stage51_summary.json"), "w"), indent=1)
    keys = ["val_logp_per_word", "C_swapped", "D_blimp_known", "E_books_right", "E_books_sensitivity", "E2_sensitivity", "F_surprise"]
    for k in keys:
        print(f"{k:28s}", "  ".join(f"{n}: {o[k]}" for n, o in out.items()))
    for k in ("pp_same_number", "pp_attractor", "rel_same_number", "rel_attractor"):
        print(f"E2_{k:24s}", "  ".join(f"{n}: {o['E2_' + k]['sensitivity']}" for n, o in out.items()))


if __name__ == "__main__":
    main()
