"""Stage 45 summary over networks (results/picture_book/s*.json -> stage45_summary.json)."""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "picture_book")


def ms(v):
    v = 100 * np.asarray(v, float)
    return [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1)]


def main():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "s[0-9].json")))]
    o = dict(networks=[r["seed"] for r in runs], styles=[r["style"] for r in runs], picture_thresholds=[r["picture_threshold"] for r in runs],
             things_named_in_primer=runs[0]["things_pictured_and_named_in_the_primer"])
    for b in ("wordbook", "primer", "both", "control"):
        o[b] = {}
        for part in ("all_34", "primer_things"):
            g = lambda f: [f(r[b][part]) for r in runs]
            o[b][part] = dict(naming=ms(g(lambda x: x["naming"])), choosing=ms(g(lambda x: x["choosing"])),
                              naming_other_styles=ms(g(lambda x: x["naming_other_styles"])),
                              reading_without_picture=ms(g(lambda x: x["reading_hard_font"]["without_picture"])),
                              reading_with_picture=ms(g(lambda x: x["reading_hard_font"]["with_picture"])),
                              reading_gain=ms(g(lambda x: x["reading_hard_font"]["with_picture"] - x["reading_hard_font"]["without_picture"])),
                              picture_units=[round(float(np.mean(g(lambda x: x["picture_units"]))), 1)])
        pt = np.mean([[r[b]["all_34"]["naming_per_thing"][k] for k in r[b]["all_34"]["naming_per_thing"]] for r in runs], 0)
        o[b]["naming_per_thing"] = dict(zip(runs[0][b]["all_34"]["naming_per_thing"], np.round(pt, 2).tolist()))
        print(b, o[b]["all_34"], "| primer things", o[b]["primer_things"]["naming"], o[b]["primer_things"]["choosing"])
    json.dump(o, open(os.path.join(R, "stage45_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
