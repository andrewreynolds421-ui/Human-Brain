"""Stage 46 summary over samplings (results/read_code/s*.json -> stage46_summary.json)."""
import glob, json, os
import numpy as np
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "results", "read_code")


def ms(v):
    v = 100 * np.asarray(v, float)
    return [round(float(v.mean()), 1), round(float(v.std(ddof=1)), 1)]


def main():
    runs = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(R, "s[0-9].json")))]
    A = [r["A_kind_of_file"] for r in runs]
    o = dict(samplings=[r["seed"] for r in runs], test_repos=runs[0]["test_repos"], A=dict(
        kind_units=ms([a["kind_units"] for a in A]), kind_units_balanced=ms([a["kind_units_balanced"] for a in A]),
        fitted_token_units=ms([a["references"]["token_units"]["test"] for a in A]), fitted_token_units_balanced=ms([a["references"]["token_units"]["test_balanced"] for a in A]),
        fitted_char_3grams=ms([a["references"]["char_3grams"]["test"] for a in A]), fitted_char_3grams_balanced=ms([a["references"]["char_3grams"]["test_balanced"] for a in A]),
        validation_same_repos=ms([a["validation"] for a in A]),
        per_kind={k: round(100 * float(np.mean([a["per_kind"][k] for a in A])), 1) for k in A[0]["per_kind"]},
        test_files_per_kind=A[0]["test_files_per_kind"], top_confusions=A[0]["top_confusions"]))
    for key in ("B_what_a_function_is_for", "B_after_also_reading_the_standard_library"):
        B = [r[key] for r in runs]
        o[key] = dict(train_functions=B[0]["train_functions"], test_functions=B[0]["test_functions"],
                      **{k: ms([b[k] for b in B]) for k in ("links_learned_from_reading", "surface_overlap", "both")},
                      strongest_links=B[0]["strongest_links"])
    print(json.dumps({k: v for k, v in o.items() if k != "test_repos"}, indent=1)[:3000])
    json.dump(o, open(os.path.join(R, "stage46_summary.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
