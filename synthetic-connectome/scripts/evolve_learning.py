#!/usr/bin/env python3
"""Evolutionary search over reward-learning settings (not over biology).

Genome = learning-rule settings of scripts/reward_learning.py.  Each
generation evaluates ``--pop`` genomes (mutants of the current best two;
random draws in generation 0, plus a no-learning control), each on a new
network seed, for ``--trials`` trials.  Fitness = accuracy over the last 60
trials, minus a penalty when firing leaves healthy ranges (any pyramidal
population > 20 Hz, or 9/46d L5 outside 0.5-15 Hz).  Results are appended to
results/evolve_learning.jsonl; rerunning resumes from that file.

    python scripts/evolve_learning.py --generations 8 --pop 4 --workers 2
"""
import argparse, json, os, sys, time
import numpy as np
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(__file__))
OUT = os.path.join(os.path.dirname(__file__), "..", "results", os.environ.get("EVOLVE_OUT", "evolve_learning.jsonl"))

RANGES = dict(eta=(0.1, 20.0, "log"), tau_elig_ms=(100.0, 2000.0, "log"), a_minus=(0.7, 1.5, "lin"),
              baseline_trials=(3.0, 60.0, "log"), w_max_factor=(1.5, 30.0, "log"),
              istdp_eta=(0.0, 0.1, "zlog"), local_plastic=(0, 1, "bit"))
if os.environ.get("EVOLVE_V2"):
    # round 2: action selection + stimulus strength + gating options
    RANGES.update(wta=(1.0, 30.0, "log"), cue_fibres=(500.0, 3000.0, "log"), gate=(0, 1, "bit"), normalize=(0, 1, "bit"))
SEED_GENOMES = [dict(eta=6.55, tau_elig_ms=263, a_minus=0.73, baseline_trials=10, w_max_factor=30, istdp_eta=0.0163,
                     local_plastic=0, wta=8.0, cue_fibres=1000, gate=1, normalize=0),
                dict(eta=2.92, tau_elig_ms=224, a_minus=0.73, baseline_trials=3, w_max_factor=17.1, istdp_eta=0.0163,
                     local_plastic=0, wta=8.0, cue_fibres=1000, gate=0, normalize=0)]


def random_genome(rng):
    g = {}
    for k, (lo, hi, kind) in RANGES.items():
        if kind == "log":
            g[k] = float(np.exp(rng.uniform(np.log(lo), np.log(hi))))
        elif kind == "lin":
            g[k] = float(rng.uniform(lo, hi))
        elif kind == "zlog":
            g[k] = 0.0 if rng.random() < 0.5 else float(np.exp(rng.uniform(np.log(1e-3), np.log(hi))))
        else:
            g[k] = int(rng.random() < 0.5)
    return g


def mutate(g, rng, strength=0.5):
    g = dict(g)
    for k, (lo, hi, kind) in RANGES.items():
        if rng.random() > 0.6:
            continue
        if kind == "log":
            g[k] = float(np.clip(g[k] * np.exp(rng.normal(0, strength)), lo, hi))
        elif kind == "lin":
            g[k] = float(np.clip(g[k] + rng.normal(0, strength * 0.2 * (hi - lo)), lo, hi))
        elif kind == "zlog":
            if g[k] == 0.0:
                g[k] = 0.0 if rng.random() < 0.7 else 1e-3
            else:
                g[k] = 0.0 if rng.random() < 0.2 else float(np.clip(g[k] * np.exp(rng.normal(0, strength)), 1e-4, hi))
        else:
            g[k] = int(1 - g[k]) if rng.random() < 0.25 else g[k]
    return g


def fitness(r):
    acc = float(np.mean(r["curve"][-60:]))
    pen = 0.0
    for k, v in r["rates"].items():
        if k.endswith("E") and v > 20:
            pen += 0.1 * (v - 20) / 20
    l5 = r["rates"].get("9/46d:L5E", 5)
    if l5 < 0.5 or l5 > 15:
        pen += 0.2
    return acc - pen


def evaluate(args):
    genome, trials, seed, tag = args
    import reward_learning as rl
    t0 = time.time()
    try:
        r = rl.run(genome, trials, seed)
        r["fitness"] = fitness(r)
    except Exception as e:                    # a genome that breaks the simulation scores 0
        r = dict(genome=genome, error=f"{type(e).__name__}: {e}", fitness=0.0, curve=[], rates={})
    r["wall_s"] = round(time.time() - t0); r["tag"] = tag; r["seed"] = seed
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--generations", type=int, default=8)
    ap.add_argument("--pop", type=int, default=4)
    ap.add_argument("--trials", type=int, default=160)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    done = [json.loads(l) for l in open(OUT)] if os.path.exists(OUT) else []
    gen0 = max([d["generation"] for d in done], default=-1) + 1
    ctx = get_context("spawn")
    with ctx.Pool(a.workers, maxtasksperchild=1) as pool:
        for gen in range(gen0, gen0 + a.generations):
            # rank genomes by their MEAN fitness over every evaluation (new network each time)
            by = {}
            for d in done:
                if "error" in d:
                    continue
                key = json.dumps({k: d["genome"].get(k) for k in RANGES}, sort_keys=True)
                by.setdefault(key, []).append(d["fitness"])
            ranked = sorted(by.items(), key=lambda kv: -np.mean(kv[1]))
            if not ranked:
                genomes = ([dict(SEED_GENOMES[0], eta=0.0)] + SEED_GENOMES + [random_genome(rng) for _ in range(a.pop)])[: a.pop] \
                    if os.environ.get("EVOLVE_V2") else [dict(eta=0.0)] + [random_genome(rng) for _ in range(a.pop - 1)]
            else:
                parents = [json.loads(k) for k, _ in ranked[:2]]
                genomes = [parents[0]] + [mutate(parents[k % len(parents)], rng, strength=0.6 if gen < 3 else 0.35)
                                          for k in range(a.pop - 1)]          # re-test the leader every generation
                print(f"gen {gen} leader: mean fitness {np.mean(ranked[0][1]):.3f} over {len(ranked[0][1])} networks", flush=True)
            jobs = [(g, a.trials, 1000 * gen + k + 1, f"g{gen}i{k}") for k, g in enumerate(genomes)]
            for r in pool.imap_unordered(evaluate, jobs):
                r["generation"] = gen
                rec = {k: v for k, v in r.items()}
                with open(OUT, "a") as fh:
                    fh.write(json.dumps(rec) + "\n")
                done.append(rec)
                g = r["genome"]
                print(f"gen {gen} {r['tag']}: fitness {r['fitness']:.3f}  acc last60 {np.mean(r['curve'][-60:]) if r['curve'] else 0:.2f}  "
                      f"eta {g.get('eta', 0):.2f} tau_e {g.get('tau_elig_ms', 0):.0f} a- {g.get('a_minus', 0):.2f} "
                      f"base {g.get('baseline_trials', 0):.0f} wmax {g.get('w_max_factor', 0):.1f} istdp {g.get('istdp_eta', 0):.4f} "
                      f"local {g.get('local_plastic', 0)} wta {g.get('wta', 0):.1f} cue {g.get('cue_fibres', 1000):.0f} "
                      f"gate {g.get('gate', 0)} norm {g.get('normalize', 0)} dom {r.get('dominance')}  sel {r.get('selectivity')}  {r['wall_s']}s"
                      + (f"  ERROR {r['error']}" if "error" in r else ""), flush=True)


if __name__ == "__main__":
    main()
