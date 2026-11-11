"""Stage 43 summary over networks (results/primer/s*.json -> stage43_summary.json)."""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "primer")


def ms(v, pct=True):
    v = np.asarray(v, float) * (100 if pct else 1)
    return [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1) if len(v) > 1 else None]


def walk(runs, path):
    out = []
    for r in runs:
        x = r
        for k in path:
            x = x[k]
        out.append(x)
    return out


def main():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "s[0-9].json")))]
    o = dict(networks=[r["seed"] for r in runs], book=runs[0]["book"])
    A = "A_reading_from_nothing"
    o[A] = {k: ms(walk(runs, (A, k)), pct=k not in ("units", "duplicates", "words_with_a_unit", "first_sightings", "known_tokens"))
            for k in runs[0][A] if k != "by_tenth_of_book"}
    o[A]["by_tenth_of_book"] = np.round(100 * np.mean(walk(runs, (A, "by_tenth_of_book")), 0), 1).tolist()
    for sec, subs in (("B_second_copy", ("without_word_prediction", "with_word_prediction", "reference_units_taught_with_names")),
                      ("C_second_reader", ("without_word_prediction", "with_word_prediction"))):
        o[sec] = {s: {k: ms(walk(runs, (sec, s, k))) for k in runs[0][sec][s] if k != "tokens" and runs[0][sec][s][k] is not None} for s in subs}
    o["C_second_reader"]["next_word_guessed"] = {k: ms(walk(runs, ("C_second_reader", "next_word_guessed", k))) for k in ("right", "most_common_word_baseline")}
    for sec in ("D_before_reading_on", "D_after_reading_on"):
        o[sec] = dict(last_20_lessons={k: ms(walk(runs, (sec, "last_20_lessons", k))) for k in ("known_share", "known_read_right", "known_called_unknown", "unknown_flagged", "letter_by_letter_with_prediction")},
                      next_word_guessed={k: ms(walk(runs, (sec, "next_word_guessed", k))) for k in ("right", "most_common_word_baseline")})
    o["D_after_reading_on"]["units"] = ms(walk(runs, ("D_after_reading_on", "units")), pct=False)
    o["D_after_reading_on"]["words_with_a_unit"] = ms(walk(runs, ("D_after_reading_on", "words_with_a_unit")), pct=False)
    print(json.dumps(o, indent=1))
    json.dump(o, open(os.path.join(R, "stage43_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
