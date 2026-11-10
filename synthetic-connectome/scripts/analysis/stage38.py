"""Stage 38 summary: letter units on the learned letter layers (pixels / dense edge cells / sparse edge
cells) and on the edge cells directly, on 60 never-seen font families; all recordings with thresholds
held fixed during reading (files *_i<inhib>; the drift-era numbers are kept for the record).

    python scripts/analysis/stage38.py   ->  results/letters_fam/stage38_summary.json
"""
import glob, json, os
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "letters_fam")


def main():
    out = {}
    for f in sorted(glob.glob(os.path.join(R, "*", "score_i*_fixed.json"))):
        r = json.load(open(f))[0]
        key = f"{r['input']}, inhibition x{r['inhib']:g}"
        out.setdefault(key, dict(networks=[], layer=[], edge_cells_direct=[], layer_and_edges=[], cells_firing=[], hardest=[]))
        o = out[key]
        o["networks"].append(r["seed"]); o["layer"].append(r["letter_units_on_layer"]["test_new_families"])
        o["edge_cells_direct"].append(r["letter_units_on_v1_cells"]["test_new_families"])
        o["layer_and_edges"].append(r["letter_units_on_layer_and_v1"]["test_new_families"])
        o["cells_firing"].append(r["cells_firing_test"])
        pl = r["letter_units_on_layer"]["per_letter"]; o["hardest"].append(sorted(pl.items(), key=lambda x: x[1])[:5])
        o.setdefault("confusions", []).append(r["letter_units_on_layer"]["top_confusions"][:5])
    for k, o in out.items():
        v = 100 * np.array(o["layer"])
        o["layer_mean_sd"] = [round(v.mean(), 1), round(v.std(ddof=1), 1) if len(v) > 1 else None]
        print(f"{k:28s} n={len(v)} layer {o['layer_mean_sd']} | edge cells direct {np.mean(o['edge_cells_direct']):.3f} | cells firing {np.mean(o['cells_firing']):.1f}")
    json.dump(out, open(os.path.join(R, "stage38_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
