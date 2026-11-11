"""Stage 42 summary over networks (results/word_units/s*.json -> stage42_summary.json)."""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "word_units")


def ms(v):
    v = 100 * np.asarray([x for x in v if x is not None], float)
    return [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1) if len(v) > 1 else None] if len(v) else None


def main():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "s[0-9].json")))]
    out = dict(networks=[r["seed"] for r in runs])
    for N in runs[0]["vocab"]:
        out[N] = {}
        for k in runs[0]["vocab"][N]:
            o = {}
            for style in ("lower", "Capital", "ALLCAPS"):
                g = lambda key: ms([r["vocab"][N][k][style][key] for r in runs])
                o[style] = dict(word=g("word"), word_each_word_equally=g("word_types"), every_letter_right_alone=g("all_alone"),
                                every_letter_right_with_prediction=g("all_pred"), letters_alone=g("letters_alone"),
                                letters_with_prediction=g("letters_pred"), letters_from_word=g("letters_word"),
                                capital_I_alone=g("capI_alone"), capital_I_from_word=g("capI_word"),
                                capital_I_count=int(np.sum([r["vocab"][N][k][style]["capI_n"] for r in runs])),
                                known_word_called_unknown=g("known_flagged_unknown"))
            o["unknown_words_flagged"] = ms([r["vocab"][N][k]["unknown_words_flagged"] for r in runs])
            out[N][k] = o
            print(f"{N:>5} words, {k:>5} showings: " + " | ".join(f"{s} {o[s]['word']} (letters only {o[s]['every_letter_right_alone']}, +pred {o[s]['every_letter_right_with_prediction']})" for s in ("lower", "Capital", "ALLCAPS"))
                  + f" | unknown flagged {o['unknown_words_flagged']}, known called unknown {o['lower']['known_word_called_unknown']}")
    json.dump(out, open(os.path.join(R, "stage42_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
