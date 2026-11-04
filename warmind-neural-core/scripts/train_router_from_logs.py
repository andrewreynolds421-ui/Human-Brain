#!/usr/bin/env python3
"""Train the spiking learner on RASPUTIN's real routing history (offline, shadow only).

Reads warmind.db `tool_calls`: for each user message sent to RASPUTIN, the
label is the submind RASPUTIN consulted first in its first response
(round 0), or RASPUTIN itself if it consulted none.  Destinations with fewer
than --min-examples messages are folded into RASPUTIN.  Messages are split by
time: the brain trains by reward on the older 80% (each repeated --epochs
times in shuffled order) and is tested, without reward, on the newest 20%.

Nothing here acts on RASPUTIN; it only measures whether the brain could make
the decision.  Results go to --out (JSON) and the console.

    python scripts/train_router_from_logs.py --db warmind.db --epochs 2
    python scripts/train_router_from_logs.py --dry-run      # just show the dataset
"""
import argparse, collections, json, os, sqlite3, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))


def load(db, min_examples):
    con = sqlite3.connect(db)
    rows = con.execute("SELECT user_message, tool_name, consulted, created_at FROM tool_calls "
                       "WHERE target = 'RASPUTIN' AND round = 0 ORDER BY created_at, id").fetchall()
    con.close()
    by_msg = collections.OrderedDict()
    for msg, tool, consulted, ts in rows:
        key = (msg, ts[:16])                          # same utterance, same minute = one turn
        lab = by_msg.get(key)
        if lab is None or lab == "RASPUTIN":
            by_msg[key] = (consulted or "").upper() if tool == "consult_submind" and consulted else (lab or "RASPUTIN")
    items = [(m, lab) for (m, _), lab in by_msg.items()]
    counts = collections.Counter(lab for _, lab in items)
    keep = {lab for lab, n in counts.items() if n >= min_examples} | {"RASPUTIN"}
    items = [(m, lab if lab in keep else "RASPUTIN") for m, lab in items]
    return items, collections.Counter(lab for _, lab in items)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=os.path.join(HERE, "..", "warmind.db"))
    ap.add_argument("--min-examples", type=int, default=15)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--max-train", type=int, default=2000, help="cap on training trials")
    ap.add_argument("--n-scale", type=float, default=0.03)
    ap.add_argument("--cue-frac", type=float, default=0.15)
    ap.add_argument("--wta-ref", type=float, default=30.5)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--sensory", default="auto",
                    help="sensory layer .npz between text and the brain; 'auto' = train one on your own logged "
                         "messages first (no labels used), 'none' = word code straight into the brain")
    ap.add_argument("--sensory-cells", type=int, default=400)
    ap.add_argument("--sens-lr", type=float, default=0.33)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "router_shadow_result.json"))
    a = ap.parse_args()

    items, counts = load(a.db, a.min_examples)
    print(f"{len(items)} routed messages; destinations: {dict(counts)}")
    if len(items) < 50:
        print("Too few logged messages to train on yet (need ~50+; more is better).")
        return
    labels = sorted(counts, key=lambda k: (k != "RASPUTIN", k))
    cut = int(len(items) * 0.8)
    train, test = items[:cut], items[cut:]
    base = max(counts.values()) / len(items)
    print(f"train {len(train)}, test {len(test)}; always guessing the most common destination scores {base:.0%}")
    if a.dry_run:
        return

    from neural_core import _find_synth_path
    sys.path.insert(0, _find_synth_path())
    from brian2 import ms
    from synth.learner import LearnerColumn, build_learner_network, QUICK_SETTINGS
    t0 = time.time()
    sensory = None
    if a.sensory == "auto":
        from synth.sensory import SensoryLayer
        sens_path = os.path.join(os.path.dirname(a.out), "router_sensory.npz")
        if os.path.exists(sens_path):
            sensory = SensoryLayer.load(sens_path)
        else:
            sensory = SensoryLayer(n_in=1000, n_e=a.sensory_cells, seed=a.seed)
            msgs = [m for m, _ in train]
            reps = max(1, 8000 // max(len(msgs), 1))
            print(f"training the sensory layer on {len(msgs)} messages x{reps} (labels not used) ...", flush=True)
            sensory.train([msgs[i] for i in np.random.default_rng(a.seed).permutation(len(msgs) * reps) % len(msgs)],
                          log_every=1000)
            sensory.save(sens_path)
    elif a.sensory != "none":
        from synth.sensory import SensoryLayer
        sensory = SensoryLayer.load(a.sensory)
    col = build_learner_network(n_scale=a.n_scale, seed=a.seed)
    L = LearnerColumn(col, seed=a.seed, structural=True, surprise_gain=3.0, fast=True, lean=True,
                      genome=dict(eta=QUICK_SETTINGS["eta"], wta_ref_pool=a.wta_ref),
                      uncert_gain=QUICK_SETTINGS["uncert_gain"], n_actions=len(labels),
                      pattern_cues=True, cue_frac=a.cue_frac, sensory=sensory, sens_lr=a.sens_lr)
    L.net.run(500 * ms)
    rng = np.random.default_rng(a.seed)
    order = [train[i] for e in range(a.epochs) for i in rng.permutation(len(train))][: a.max_train]
    curve = []
    for n, (m, lab) in enumerate(order):
        r = L.trial(m)
        ok = labels[r["choice"] - 1] == lab
        L.reward(1.0 if ok else -1.0 / (len(labels) - 1))      # balanced: chance averages zero
        curve.append(int(ok))
        if (n + 1) % 50 == 0:
            print(f"trial {n + 1}/{len(order)}: training accuracy (last 50) {np.mean(curve[-50:]):.2f}", flush=True)
    hits, confusion = [], collections.Counter()
    for m, lab in test:
        r = L.trial(m)
        guess = labels[r["choice"] - 1]
        hits.append(int(guess == lab)); confusion[(lab, guess)] += 1
    L.pending = None
    acc = float(np.mean(hits))
    res = dict(messages=len(items), destinations=dict(counts), labels=labels, train_trials=len(order),
               test_accuracy=acc, majority_baseline=base, chance=1 / len(labels),
               confusion={f"{k[0]}->{k[1]}": v for k, v in confusion.items()},
               training_curve=curve, wall_s=round(time.time() - t0))
    print(f"held-out (newest 20%) accuracy {acc:.0%}; chance {1 / len(labels):.0%}; most-common-guess {base:.0%}")
    json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
