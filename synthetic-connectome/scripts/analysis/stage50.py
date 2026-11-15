"""Stage 50 summary over four networks (results/grammar/s0-3.json -> stage50_summary.json)."""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "grammar")


def ms(v):
    v = 100 * np.asarray(v, float); return [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1)]


def main():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "s[0-9].json")))]
    o = dict(networks=[r.get("seed", 0) for r in runs], K=[r["K"] for r in runs], mix=[r["mix"] for r in runs], corpus=runs[0]["words"])
    A = [r["A_categories"] for r in runs]
    o["A"] = {k: ms([a[k] for a in A]) for k in ("purity", "purity_random", "v_measure", "v_measure_random")}
    o["B"] = {k: ms([r["B_new_words"][k] for r in runs]) for k in ("1", "3", "10", "chance_most_common_class")}
    o["C"] = {t: {m: ms([r["C_word_order_never_read_books"][t][m] for r in runs]) for m in ("categories", "words_trigram", "both")} for t in ("two_neighbours_swapped", "shuffled")}
    o["D_all"] = {m: ms([r["D_blimp"]["overall"][m] for r in runs]) for m in ("categories", "words", "both", "known")}
    o["D_known"] = dict(pairs=runs[0]["D_blimp"]["pairs_with_known_words_only"]["pairs"], **{m: ms([r["D_blimp"]["pairs_with_known_words_only"][m] for r in runs]) for m in ("categories", "words", "both")})
    phen = runs[0]["D_blimp"]["by_phenomenon"]
    o["D_known_by_phenomenon"] = {k: dict(field=phen[k]["field"], pairs=phen[k]["known_pairs"], both=round(100 * float(np.mean([r["D_blimp"]["by_phenomenon"][k]["both_known_words_only"] for r in runs])), 1))
                                  for k in phen if phen[k]["known_pairs"] >= 100}
    print(json.dumps({k: v for k, v in o.items() if k != "D_known_by_phenomenon"}, indent=1))
    for k, v in sorted(o["D_known_by_phenomenon"].items(), key=lambda kv: -kv[1]["both"]):
        print(f"  {v['both']:5.1f}  {k} ({v['field']}, {v['pairs']} pairs)")
    json.dump(o, open(os.path.join(R, "stage50_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
