"""Stage 48 summary over networks (results/sleep/s*.json -> stage48_summary.json); gains paired against 'none'."""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "sleep")
KINDS = ("none", "awake", "merge", "replay", "merge+replay")


def main():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "s[0-9].json")))]
    o = dict(networks=[r["seed"] for r in runs], replay_eta=[r["replay_eta"] for r in runs], day1=[r["day1"] for r in runs])
    get = {"B second copy, read right": lambda d: d["B_second_copy"]["known_read_right"], "B called unknown": lambda d: d["B_second_copy"]["known_called_unknown"],
           "C Second Reader, read right": lambda d: d["C_second_reader"]["known_read_right"], "C unknown words flagged": lambda d: d["C_second_reader"]["unknown_flagged"],
           "A2 same book, read right": lambda d: d["A2_same_book_same_font"]["known_read_right"],
           "L letters, other fonts (slow units)": lambda d: d["L_letters_other_fonts_slow_units"], "L letters, book font (slow units)": lambda d: d["L_letters_book_font_slow_units"]}
    for k in KINDS:
        d = {}
        for name, f in get.items():
            v = 100 * np.array([f(r[k]) for r in runs]); base = 100 * np.array([f(r["none"]) for r in runs]); g = v - base
            d[name] = dict(mean=round(float(v.mean()), 1), sd=round(float(v.std(ddof=1)), 1), gain=round(float(g.mean()), 1), gain_sd=round(float(g.std(ddof=1)), 1), per_network=v.round(1).tolist())
        d["units"] = [r[k]["units"] for r in runs]; d["duplicates"] = [r[k]["duplicates"] for r in runs]
        for extra in ("units_merged_away", "merges_of_the_same_word", "groups_merging_different_words", "new_units_while_rereading"):
            if extra in runs[0][k]:
                d[extra] = [r[k][extra] for r in runs]
        o[k] = d
        print(k, {n: (x["mean"], x["gain"], x["gain_sd"]) for n, x in d.items() if isinstance(x, dict)}, "dups", d["duplicates"], {e: d[e] for e in d if e in ("units_merged_away", "merges_of_the_same_word", "groups_merging_different_words", "new_units_while_rereading")})
    json.dump(o, open(os.path.join(R, "stage48_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
