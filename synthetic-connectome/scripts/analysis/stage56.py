"""Stage 56 summary: many-noun pages, text only vs picture check (results/picture_check/*_s*.json -> stage56_summary.json)."""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results")


def ms(v):
    v = 100 * np.asarray([x for x in v if x is not None], float)
    return [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1) if len(v) > 1 else None, len(v)] if len(v) else None


def main():
    out = {}
    for c in ("text", "check"):
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "picture_check", f"{c}_s[0-9].json")))]
        if not runs: continue
        o = dict(networks=len(runs), built={k: dict(right=ms([r["built"][k]["right"] for r in runs]), plural=ms([r["built"][k]["plural_right"] for r in runs]),
                                                    singular=ms([r["built"][k]["singular_right"] for r in runs])) for k in runs[0]["built"]},
                 blimp=ms([r["D"]["known_to_stage51_vocab"] for r in runs]), books_agreement=ms([r["E"]["sensitivity_auc"] for r in runs]),
                 E2_sensitivity={k: ms([r["E2_controlled"][k]["sensitivity_auc"] for r in runs]) for k in ("pp_same_number", "pp_attractor", "rel_same_number", "rel_attractor")},
                 val_logp=round(float(np.mean([r["val_logp_per_word"] for r in runs])), 3))
        if "guess_number_held_out_nouns" in runs[0]:
            o["guess_number_held_out_nouns"] = {k: ms([r["guess_number_held_out_nouns"][k] for r in runs]) for k in runs[0]["guess_number_held_out_nouns"]}
        out[c] = o
    ref = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "picture_pages", "text_s[0-9].json")))]
    out["stage55_text_17_nouns"] = dict(E2_by_noun={k: ms([r["E2_by_noun"][k][0] for r in ref]) for k in ref[0]["E2_by_noun"]},
                                        E2_sensitivity={k: ms([r["E2_controlled"][k]["sensitivity_auc"] for r in ref]) for k in ("pp_same_number", "pp_attractor", "rel_same_number", "rel_attractor")},
                                        blimp=ms([r["D"]["known_to_stage51_vocab"] for r in ref]))
    json.dump(out, open(os.path.join(R, "picture_check", "stage56_summary.json"), "w"), indent=1)
    for k in out.get("text", out.get("check"))["built"]:
        print(f"{k:42s}", " | ".join(f"{c}: {out[c]['built'][k]['right']}" for c in ("text", "check") if c in out))
    for k in ("blimp", "books_agreement", "val_logp"):
        print(f"{k:42s}", " | ".join(f"{c}: {out[c].get(k)}" for c in out))
    for c in ("text", "check", "stage55_text_17_nouns"):
        if c in out: print(c, "E2", out[c]["E2_sensitivity"])
    if "check" in out and "guess_number_held_out_nouns" in out["check"]: print("guess", out["check"]["guess_number_held_out_nouns"])


if __name__ == "__main__":
    main()
