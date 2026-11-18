"""Stage 58 summary: Stage 57's check group with and without word endings (results/spelling -> stage58_summary.json)."""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")


def ms(v):
    v = 100 * np.asarray([x for x in v if x is not None], float)
    return [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1) if len(v) > 1 else None, len(v)] if len(v) else None


def main():
    out = {}
    for name, pat in (("whole words (Stage 57)", "frames/check_s[0-9].json"), ("words + endings", "spelling/check_s[0-9].json")):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, pat)))]
        if not runs: continue
        o = dict(networks=len(runs), built={k: dict(right=ms([r["built"][k]["right"] for r in runs]), plural=ms([r["built"][k]["plural_right"] for r in runs]),
                                                    singular=ms([r["built"][k]["singular_right"] for r in runs])) for k in runs[0]["built"]},
                 guess_number_held_out_nouns=ms([np.mean(list(r["guess_number_held_out_nouns"].values())) for r in runs]),
                 blimp=ms([r["D"]["known_to_stage51_vocab"] for r in runs]), blimp_all=ms([r["D"]["all"] for r in runs]),
                 books_agreement=ms([r["E"]["sensitivity_auc"] for r in runs]), word_order=ms([r["C"]["two_neighbours_swapped"] for r in runs]),
                 surprise=ms([r["F"]["more_surprised_by_substitute"] for r in runs]), val_logp=round(float(np.mean([r["val_logp_per_word"] for r in runs])), 3))
        out[name] = o
    json.dump(out, open(os.path.join(R, "spelling", "stage58_summary.json"), "w"), indent=1)
    first = next(iter(out.values()))
    for k in first["built"]:
        print(f"{k:36s}", " | ".join(f"{n}: {o['built'][k]['right']} (pl {o['built'][k]['plural'][0]}, sg {o['built'][k]['singular'][0]})" for n, o in out.items()))
    for k in ("guess_number_held_out_nouns", "blimp", "blimp_all", "books_agreement", "word_order", "surprise", "val_logp"):
        print(f"{k:36s}", " | ".join(f"{n}: {o[k]}" for n, o in out.items()))


if __name__ == "__main__":
    main()
