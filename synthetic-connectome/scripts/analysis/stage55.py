"""Stage 55 summary: books / text pages / picture pages / picture pages + questions
(results/picture_pages/*_s*.json -> stage55_summary.json)."""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "picture_pages")
CONDS = ("books", "text", "pictures", "questions")


def ms(v):
    v = 100 * np.asarray([x for x in v if x is not None], float)
    return [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1) if len(v) > 1 else None, len(v)] if len(v) else None


def main():
    out = {}
    for c in CONDS:
        runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, f"{c}_s[0-9].json")))]
        if not runs: continue
        o = dict(networks=len(runs), questions_asked=int(np.mean([r["questions_asked"] for r in runs])),
                 val_logp=round(float(np.mean([r["val_logp_per_word"] for r in runs])), 3),
                 blimp=ms([r["D"]["known_to_stage51_vocab"] for r in runs]), books_agreement=ms([r["E"]["sensitivity_auc"] for r in runs]),
                 word_order=ms([r["C"]["two_neighbours_swapped"] for r in runs]))
        for view in ("pages_new_with_picture", "pages_new_text_only"):
            o[view] = {k: ms([r[view][k] for r in runs]) for k in runs[0][view]}
        o["E2_by_noun"] = {k: ms([r["E2_by_noun"][k][0] for r in runs]) for k in runs[0]["E2_by_noun"]}
        o["E2_sensitivity"] = {k: ms([r["E2_controlled"][k]["sensitivity_auc"] for r in runs]) for k in ("pp_same_number", "pp_attractor", "rel_same_number", "rel_attractor")}
        out[c] = o
    json.dump(out, open(os.path.join(R, "stage55_summary.json"), "w"), indent=1)
    rows = [("pages with picture: all", "pages_new_with_picture", "all"), ("pages with picture: rel_obj attractor", "pages_new_with_picture", "rel_obj_attractor"),
            ("looks at the actor at the verb", "pages_new_with_picture", "looks_at_actor_when_verb_comes"),
            ("pages text only: all", "pages_new_text_only", "all"), ("pages text only: rel_obj attractor", "pages_new_text_only", "rel_obj_attractor"),
            ("pages text only: pp attractor", "pages_new_text_only", "pp_attractor")]
    for name, view, k in rows:
        print(f"{name:40s}", " | ".join(f"{c}: {out[c][view].get(k)}" for c in out))
    for k in out[next(iter(out))]["E2_by_noun"]:
        print(f"{k:40s}", " | ".join(f"{c}: {out[c]['E2_by_noun'][k]}" for c in out))
    for k in ("blimp", "books_agreement", "word_order", "val_logp", "questions_asked"):
        print(f"{k:40s}", " | ".join(f"{c}: {out[c][k]}" for c in out))


if __name__ == "__main__":
    main()
