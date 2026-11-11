"""Stage 44 summary over networks (results/primer_font/s*.json -> stage44_summary.json)."""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "primer_font")


def main():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "s[0-9].json")))]
    o = dict(networks=[r["seed"] for r in runs], settings=[(r["eta_font"], r["conf"]) for r in runs], validation=[r["validation"] for r in runs],
             fonts=[dict(A=r["font_A"], B=r["fonts_B"]) for r in runs])
    for sec in [k for k in runs[0] if k[:2] in ("A_", "B_", "C_")]:
        d = {}
        for k in runs[0][sec]:
            v = [r[sec][k] for r in runs if r[sec][k] is not None]
            if not v:
                continue
            if isinstance(v[0], list):
                d[k] = np.round(100 * np.mean(v, 0), 1).tolist()
            else:
                pct = k not in ("units", "duplicates", "words_with_a_unit")
                v = np.array(v, float) * (100 if pct else 1)
                d[k] = [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1)]
        o[sec] = d
        print(sec, d)
    for a_, b_, key in (("B_units_off_reading_off", "B_units_on_reading_on", "known_read_right"), ("B_units_off_reading_off", "B_units_on_reading_on", "every_letter_right"),
                        ("A_off", "A_on", "every_letter_right"), ("C_second_reader_off", "C_second_reader_on", "known_read_right")):
        dlt = 100 * np.array([r[b_][key] - r[a_][key] for r in runs])
        o[f"gain_{b_}_{key}"] = [round(float(dlt.mean()), 1), round(float(dlt.std(ddof=1)), 1), dlt.round(1).tolist()]
        print("paired gain", b_, key, o[f"gain_{b_}_{key}"])
    json.dump(o, open(os.path.join(R, "stage44_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
