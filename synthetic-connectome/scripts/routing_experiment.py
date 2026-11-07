#!/usr/bin/env python3
"""Can the spiking learner learn RASPUTIN's routing decision from reward?

SYNTHETIC test set written for this experiment (not real user data): short
messages for three destinations — answer directly (RASPUTIN), ASCLEPIUS
(health), PLUTUS (money).  Messages are encoded by synth/textcode.py (each
content word drives a fixed random 5% of the cue cells); the learner has one
action pool per destination and is rewarded +1 when its choice matches the
label, -1 otherwise.  Accuracy is measured on held-out phrasings without
reward, so any success must come from words shared with training messages.

    python scripts/routing_experiment.py --train 300 --n-scale 0.015 --seed 41
"""
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from brian2 import ms                                         # noqa: E402
from synth.learner import LearnerColumn, build_learner_network, QUICK_SETTINGS   # noqa: E402

DATA = {  # synthetic
    "RASPUTIN": dict(
        train=["tell me a joke", "what time is it", "summarize this conversation", "remind me what we talked about",
               "write a short poem about rain", "who wrote hamlet", "explain how rainbows form",
               "draft an email to my landlord", "translate hello into spanish", "what does recursion mean",
               "give me a fun fact", "help me name my project", "rename the dashboard widget",
               "open the project notes", "what is the capital of peru", "write a thank you note"],
        test=["tell me a fun story", "what does entropy mean", "write a poem about the sea",
              "help me draft a note to my landlord", "explain how tides form", "who wrote the odyssey"]),
    "ASCLEPIUS": dict(
        train=["what's my resting heart rate", "how did i sleep last night", "log my weight today",
               "track my blood pressure", "how many steps did i walk", "is my heart rate trend normal",
               "show my sleep score this week", "i have a headache and fever", "log my run distance",
               "track my calories", "my blood sugar reading was high", "how is my sleep trend",
               "chart my weight over the month", "record my workout", "fitbit sync my steps", "log my symptoms"],
        test=["what was my heart rate during the run", "chart my sleep this month", "log my blood pressure reading",
              "how many steps this week", "track my weight trend", "i have a fever and cough"]),
    "PLUTUS": dict(
        train=["how is the s&p doing", "what's the price of nvidia stock", "show my portfolio",
               "track my spending this month", "what's the market outlook", "did any senators buy stock",
               "chart my budget", "how much did i spend on groceries", "what are bond yields doing",
               "log a purchase of fifty dollars", "show congress stock trades", "is the market up today",
               "what's apple stock at", "track my savings", "my budget for rent", "show market movers"],
        test=["how is apple stock doing", "chart my spending this week", "what is the market outlook today",
              "show my portfolio value", "did congress members trade stock", "how much did i spend on rent"]),
}
LABELS = list(DATA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=300)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--n-scale", type=float, default=0.015)
    ap.add_argument("--cue-frac", type=float, default=0.05)
    ap.add_argument("--test-every", type=int, default=100)
    ap.add_argument("--cue-gain", type=float, default=1.0)
    ap.add_argument("--code", default="union", choices=["union", "kwta"])
    ap.add_argument("--cue-ms", type=float, default=200.0)
    ap.add_argument("--data", default="synthetic", help="'synthetic' or 'clinc' (data/routing_clinc3.json)")
    ap.add_argument("--sensory", default=None, help="trained sensory layer (.npz) between text and the brain")
    ap.add_argument("--sens-gain", type=float, default=1.0)
    ap.add_argument("--sens-lr", type=float, default=1.0)
    ap.add_argument("--n-test", type=int, default=150)
    ap.add_argument("--balanced", action="store_true", help="wrong = -1/(K-1), so chance averages 0 reward")
    ap.add_argument("--slow", action="store_true", help="previous rule: eta 6.55, no uncertainty boost")
    ap.add_argument("--wta-ref", type=float, default=0.0, help="selection-input scaling reference pool size (0 = off)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--preset", default="human_h01_dendritic", help="column preset (human_h01_dendritic_v2 = stage 25)")
    ap.add_argument("--selection", default="wta", help="wta or bg (basal-ganglia loop)")
    ap.add_argument("--bg", default="{}", help="JSON BasalGanglia options (e.g. results/hom/configs.json r25_hom)")
    ap.add_argument("--da", default=None, help="JSON DopamineSystem options; turns on the dopamine population")
    ap.add_argument("--ckpt", default=None, help="checkpoint stem: state saved every 25 trials, resumed if present")
    a = ap.parse_args()
    t0 = time.time()
    col = build_learner_network(n_scale=a.n_scale, seed=a.seed, preset=a.preset)
    L = LearnerColumn(col, seed=a.seed, structural=True, surprise_gain=3.0, fast=True, lean=True,
                      genome=dict(eta=6.55 if a.slow else QUICK_SETTINGS["eta"], wta_ref_pool=a.wta_ref),
                      uncert_gain=0.0 if a.slow else QUICK_SETTINGS["uncert_gain"],
                      n_actions=3, pattern_cues=True, sensory=a.sensory, sens_gain=a.sens_gain, sens_lr=a.sens_lr, cue_frac=a.cue_frac, cue_gain=a.cue_gain, cue_code=a.code, cue_ms=a.cue_ms,
                      selection=a.selection, bg_kw=json.loads(a.bg), da_kw=None if a.da is None else json.loads(a.da))
    L.net.run(500 * ms)
    rng = np.random.default_rng(a.seed)
    if a.data == "clinc":
        C = json.load(open(os.path.join(os.path.dirname(__file__), "..", "data", "routing_clinc3.json")))
        labs = C["labels"]
        train = [(m, labs.index(l)) for m, l in C["train"]]
        test = [(m, labs.index(l)) for m, l in (C["test"][i] for i in rng.choice(len(C["test"]), a.n_test, replace=False))]
    else:
        labs = LABELS
        train = [(m, k) for k, lab in enumerate(LABELS) for m in DATA[lab]["train"]]
        test = [(m, k) for k, lab in enumerate(LABELS) for m in DATA[lab]["test"]]
    curve, tests, choices = [], [], []
    start, wall_before = 0, 0.0
    PY_STATE = ("rbar", "rslow", "n_rewards", "history", "present", "t_ms", "cum_weight_change", "last_surprise")

    def save_ckpt(done):
        """network state (Brian2 store to file) + the learner's Python-side state + this script's state"""
        import pickle
        L.net.store("ckpt", filename=a.ckpt + ".brian")
        st = dict(done=done, rng=rng.bit_generator.state, curve=curve, tests=tests, choices=choices,
                  wall=wall_before + time.time() - t0, learner={k: getattr(L, k) for k in PY_STATE if hasattr(L, k)},
                  sp=(getattr(L, "sp_rule").__dict__ if getattr(L, "sp_rule", None) is not None else None))
        pickle.dump(st, open(a.ckpt + ".tmp", "wb")); os.replace(a.ckpt + ".tmp", a.ckpt + ".pkl")

    if a.ckpt and os.path.exists(a.ckpt + ".pkl") and os.path.exists(a.ckpt + ".brian"):
        import pickle
        st = pickle.load(open(a.ckpt + ".pkl", "rb"))
        L.net.restore("ckpt", filename=a.ckpt + ".brian")
        rng.bit_generator.state = st["rng"]; curve, tests, choices = st["curve"], st["tests"], st["choices"]
        for k, v in st["learner"].items():
            setattr(L, k, v)
        if st.get("sp") is not None:
            L.sp_rule.__dict__.update(st["sp"])
        L.pending = None
        start, wall_before = st["done"], st["wall"]
        print(f"resumed from checkpoint after {start} training trials", flush=True)

    def evaluate(after):
        ok = []
        for m, k in test:
            r = L.trial(m); ok.append(int(r["choice"] == k + 1))
        L.pending = None                      # no reward on test trials
        acc = float(np.mean(ok)); tests.append(dict(after=after, acc=acc))
        print(f"  held-out accuracy after {after} training trials: {acc:.2f} (chance {1 / len(labs):.2f}, n={len(test)})", flush=True)

    if start == 0:
        evaluate(0)
    for t in range(start, a.train):
        m, k = train[rng.integers(len(train))]
        r = L.trial(m)
        choices.append((k + 1, r["choice"], r["pool_spikes"]))
        ok = r["choice"] == k + 1
        L.reward(1.0 if ok else (-1.0 / (len(labs) - 1) if a.balanced else -1.0))
        curve.append(int(ok))
        if (t + 1) % 25 == 0:
            print(f"trial {t + 1}: training accuracy (last 25) {np.mean(curve[-25:]):.2f}", flush=True)
        if (t + 1) % a.test_every == 0:
            evaluate(t + 1)
        if a.ckpt and (t + 1) % 25 == 0:
            save_ckpt(t + 1)
    res = dict(seed=a.seed, n_scale=a.n_scale, neurons=col.N, pools={k: int(v.size) for k, v in L.pools.items()},
               cue_cells=int(L.cue_cells_all.size), curve=curve, choices=choices, cue_frac=a.cue_frac, cue_gain=a.cue_gain, code=a.code, cue_ms=a.cue_ms, data=a.data, sensory=a.sensory, sens_gain=a.sens_gain, sens_lr=a.sens_lr, wta_ref=a.wta_ref, balanced=a.balanced, slow=a.slow, tests=tests, wall_s=round(wall_before + time.time() - t0), preset=a.preset, selection=a.selection, bg=json.loads(a.bg), da=a.da,
               note="synthetic messages written for this experiment")
    print(json.dumps({k: v for k, v in res.items() if k != "curve"}))
    if a.out:
        json.dump(res, open(a.out, "w"))
    if a.ckpt:
        for ext in (".pkl", ".brian"):
            if os.path.exists(a.ckpt + ext):
                os.remove(a.ckpt + ext)


if __name__ == "__main__":
    main()
