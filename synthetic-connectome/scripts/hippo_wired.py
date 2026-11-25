#!/usr/bin/env python3
"""Stage 81: a hippocampus wired from data instead of textbook boxes.

Stages 75-80 had the right order of stages (entorhinal -> dentate -> CA3 -> CA1) but CA3 and CA1 were not
populations: the dentate code was itself the storage address of a memory table, pattern completion was two fixed
lookups, and sparsity was a hard top-5 % cut. Here each subfield is a population sized from human counts, each
pathway has the shape the anatomy gives it, and the memory lives in the synapses where the brain keeps it:

  EC II  (64)     cortex -> entorhinal layer II: the "what" key (learned, as before)
  DG     (1,472)  perforant path EC II -> granule cells (fixed, dense); basket-cell feedback inhibition holds
                  activity near 2 % (sparsity emerges from inhibition, no hard cut)
  CA3    (265)    mossy fibres DG -> CA3: fixed, sparse, strong - each CA3 cell gets 46 granule inputs;
                  perforant path EC II -> CA3 (learned, dense); feedback inhibition holds ~5 % active.
                  At storage the mossy fibres dominate (they "detonate" a new CA3 pattern); at recall the cue
                  arrives through the perforant path only, and the CA3 recurrent collaterals complete it.
  CA3 -> CA3      recurrent collaterals: fast autoassociative synapses written during the story (delta rule)
  EC III (352)    cortex -> entorhinal layer III: the content to remember (learned)
  CA1    (1,576)  driven by EC III directly (temporoammonic path, learned) and by CA3 through the Schaffer
                  collaterals, whose fast synapses are written by the difference between the two (CA1 compares
                  what CA3 predicts with what the cortex says now: the mismatch is what gets learned)
  SUB    (442)    CA1 -> subiculum -> entorhinal layer V -> cortex: the recalled content goes back out
Sizes: human counts (West & Gundersen 1990; EC from Gomez-Isla et al. 1996) scaled so EC II = 64 cells.
Pathway numbers: rat (Amaral et al. 1990, via Treves & Rolls 1992; Acsady et al. 1998). Activity levels:
model literature (DG ~2 %, Treves & Rolls 1992), CA3 5 % ASSUMED. Rate neurons, not spiking (as Stages 75-80).

WiredCore is the drop-in: store(key_emb, content_emb, gate) per event; recall(cue_emb) -> content_emb.
"""
import math
import numpy as np

EC2 = 64
HUMAN = dict(EC2=652_500, EC3=3_590_500, DG=15_000_000, CA3=2_700_000, CA1=16_000_000, SUB=4_500_000)
SIZE = {k: int(round(v * EC2 / HUMAN["EC2"])) for k, v in HUMAN.items()}           # EC2 64, EC3 352, DG 1472, CA3 265, CA1 1569, SUB 441
MF_IN = 46            # granule inputs per CA3 pyramid (rat; Amaral et al. 1990)
DG_ACTIVE = 0.02      # Treves & Rolls 1992 worked example; DG "ultra-sparse" (Diamantaki et al. 2016)   LITERATURE MODEL VALUE
CA3_ACTIVE = 0.05     # ASSUMED
RC_STEPS = 3          # recurrent settling steps at recall                                                  ASSUMED
PROVENANCE = dict(sizes="human counts West & Gundersen 1990 (DG, CA3-2, CA1, subiculum) and Gomez-Isla 1996 / "
                        "Insausti (EC II 652,500, EC III 3,590,500), scaled to EC II = 64",
                  mf_in="46 mossy-fibre inputs per CA3 cell, rat (Amaral et al. 1990)",
                  pp_rc="perforant path ~3,750 and recurrent ~12,000 synapses per CA3 cell (rat) exceed the scaled "
                        "populations, so both are dense here",
                  dg_active="2 % (model literature); CA3 5 % ASSUMED; inhibition by a basket-cell feedback loop",
                  neurons="rate model, not spiking")


def build_core(torch, E, seed, mode="wired"):
    """mode: wired | no_rc (no recurrent completion) | dense_mf (mossy fibres all-to-all, weak) | shared (recall cue takes the storage route)"""
    nn = torch.nn

    def inhibit(x, frac, steps=12):
        """basket-cell feedback inhibition: a shared inhibitory signal rises with the fraction of active cells
        until about `frac` of them stay above it (subtractive, so the surviving activity is graded)"""
        with torch.no_grad():
            lo = torch.zeros_like(x[..., :1]); hi = x.max(-1, keepdim=True).values.clamp(min=1e-6)
            for _ in range(steps):                                # the inhibition settles (bisection = its fixed point)
                th = (lo + hi) / 2; f = (x > th).float().mean(-1, keepdim=True)
                lo = torch.where(f > frac, th, lo); hi = torch.where(f > frac, hi, th)
            th = (lo + hi) / 2
        return torch.relu(x - th)

    unit = lambda a: a / (a.norm(dim=-1, keepdim=True) + 1e-6)

    class WiredCore(nn.Module):
        def __init__(s):
            super().__init__()
            g = torch.Generator().manual_seed(7700 + seed); S = SIZE
            s.ec2 = nn.Linear(E, S["EC2"]); s.ec3 = nn.Linear(E, S["EC3"])
            s.register_buffer("pp_dg", torch.randn(S["EC2"], S["DG"], generator=g) / S["EC2"] ** 0.5)       # fixed
            mf = torch.zeros(S["CA3"], S["DG"])
            for i in range(S["CA3"]): mf[i, torch.randperm(S["DG"], generator=g)[:MF_IN]] = 1.0
            if mode == "dense_mf": mf = torch.ones_like(mf) * MF_IN / S["DG"]
            s.register_buffer("mf", mf)                                                                    # fixed, sparse
            s.mf_gain = nn.Parameter(torch.tensor(3.0))                                                      # strong synapses
            s.pp3 = nn.Linear(S["EC2"], S["CA3"]); s.rc_gain = nn.Parameter(torch.tensor(1.0)); s.cue_gain = nn.Parameter(torch.tensor(1.0))
            s.ta = nn.Linear(S["EC3"], S["CA1"])                                                             # temporoammonic EC III -> CA1
            s.sub = nn.Linear(S["CA1"], S["SUB"]); s.out = nn.Linear(S["SUB"], E)                            # CA1 -> subiculum -> EC V -> cortex
            s.sparsity = {}

        def dg(s, k):
            d = inhibit(torch.relu(k @ s.pp_dg), DG_ACTIVE); s.sparsity["DG"] = float((d > 0).float().mean()); return unit(d)

        def ca3_store(s, e):
            """the CA3 pattern for a new event: mossy fibres (dominant) + perforant path"""
            k = s.ec2(e); x = s.mf_gain * s.dg(k) @ s.mf.t() / MF_IN ** 0.5 + s.pp3(k)
            c = inhibit(x, CA3_ACTIVE); s.sparsity["CA3"] = float((c > 0).float().mean()); return unit(c)

        def ca3_cue(s, e, R):
            """recall: the cue arrives through the perforant path only; the recurrent collaterals complete it"""
            c0 = s.pp3(s.ec2(e)); c = unit(inhibit(c0, CA3_ACTIVE))
            if mode != "no_rc" and R is not None:
                for _ in range(RC_STEPS): c = unit(inhibit(s.rc_gain * torch.einsum("bij,bj->bi", R, c) + s.cue_gain * unit(c0), CA3_ACTIVE))
            return c

        def ca1_target(s, v):
            return torch.tanh(s.ta(s.ec3(v)))

        def readout(s, a):
            return s.out(torch.relu(s.sub(a)))

        def new_memory(s, B):
            """the fast synapses, kept as their history of updates: R = sum g (c - Rc) c^T and Sc = sum g (a1 - Sc c) c^T
            are applied to a pattern x as sum_s err_s (c_s . x) - the same numbers without the 265 x 1,569 matrices"""
            return dict(C=[], eR=[], eS=[], st={})

        @staticmethod
        def apply(mem, which, x):
            if not mem["C"]: return None
            st = mem["st"]                                                            # stacked once per stored event, reused by every recall
            if "C" not in st: st["C"] = torch.stack(mem["C"], 1)
            if which not in st: st[which] = torch.stack(mem[which], 1)
            return torch.einsum("bt,btm->bm", torch.einsum("btn,bn->bt", st["C"], x), st[which])

        def store(s, mem, key_e, val_e, gate, c=None):
            """one event: CA3 autoassociation (recurrent collaterals) and CA3 -> CA1 (Schaffer), both delta rule;
            gate (B,1) = how strongly this moment is stored. Returns the CA3 pattern used."""
            if c is None: c = s.ca3_store(key_e)
            pr = s.apply(mem, "eR", c); ps = s.apply(mem, "eS", c); a1 = s.ca1_target(val_e)
            mem["eR"].append(gate * (c - (pr if pr is not None else 0)))                  # R += g (c - Rc) c^T
            mem["eS"].append(gate * (a1 - (ps if ps is not None else 0)))                 # CA1: EC III minus Schaffer prediction
            mem["C"].append(c); mem["st"] = {}; return c

        def ca3_input(s, e):
            k = s.ec2(e); return s.mf_gain * s.dg(k) @ s.mf.t() / MF_IN ** 0.5 + s.pp3(k)

        def ca3_cue_mem(s, e, mem):
            c0 = s.ca3_input(e) if mode == "shared" else s.pp3(s.ec2(e))      # shared: the cue takes the storage route (DG + mossy fibres + PP)
            c = unit(inhibit(c0, CA3_ACTIVE))
            if mode != "no_rc" and mem["C"]:
                for _ in range(RC_STEPS): c = unit(inhibit(s.rc_gain * s.apply(mem, "eR", c) + s.cue_gain * unit(c0), CA3_ACTIVE))
            return c

        def recall(s, mem, cue_e):
            c = s.ca3_cue_mem(cue_e, mem); a = s.apply(mem, "eS", c)
            if a is None: a = torch.zeros(c.shape[0], SIZE["CA1"])
            return s.readout(a), c

    return WiredCore()


def pretrain_loop(torch, core, table, steps=3000, bs=256, seed=0):
    """the monosynaptic loop EC III -> CA1 -> subiculum -> EC V learns, slowly and before any story, to give back
    the cortical pattern it was given (as in complementary-learning-systems models of the hippocampus, e.g. Schapiro
    et al. 2017) - so a recalled CA1 pattern reinstates the cortex's own representation of the word"""
    params = list(core.ec3.parameters()) + list(core.ta.parameters()) + list(core.sub.parameters()) + list(core.out.parameters())
    opt = torch.optim.Adam(params, lr=1e-3); g = torch.Generator().manual_seed(9100 + seed); table = table.detach()
    for i in range(steps):
        e = table[torch.randint(0, table.shape[0], (bs,), generator=g)]
        r = core.readout(core.ca1_target(e)); loss = (1 - torch.nn.functional.cosine_similarity(r, e, dim=-1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        e = table[:4096]; r = core.readout(core.ca1_target(e)); sim = torch.nn.functional.normalize(r, dim=-1) @ torch.nn.functional.normalize(e, dim=-1).t()
        return dict(loop_cos=round(float(torch.diagonal(sim).mean()), 3), loop_top1=round(float((sim.argmax(1) == torch.arange(e.shape[0])).float().mean()), 3))


def selftest(steps=1500, seed=0):
    """the core alone: stories of random key -> content pairs, later facts about a key replace earlier ones;
    recall each key's latest content among 8 candidates"""
    import torch
    torch.manual_seed(seed); E, B, NW = 48, 32, 200
    words = torch.nn.functional.normalize(torch.randn(NW, E), dim=-1)
    rng = np.random.default_rng(seed); res = {}
    for mode in ("wired", "no_rc", "dense_mf"):
        core = build_core(torch, E, seed, mode); opt = torch.optim.Adam(core.parameters(), lr=1e-3); accs = []
        for step in range(steps):
            T = 12; keys = rng.integers(0, 40, (B, T)); vals = rng.integers(40, NW, (B, T))
            mem = core.new_memory(B)
            for t in range(T): core.store(mem, words[keys[:, t]], words[vals[:, t]], torch.ones(B, 1))
            qi = rng.integers(0, T, B); qk = keys[np.arange(B), qi]
            latest = np.array([vals[b][np.nonzero(keys[b] == qk[b])[0][-1]] for b in range(B)])
            cands = np.concatenate([latest[:, None], rng.integers(40, NW, (B, 7))], 1)
            z, _ = core.recall(mem, words[qk]); ce = words[cands]
            lg = 5 * torch.einsum("be,bce->bc", torch.nn.functional.normalize(z, dim=-1), ce)
            loss = torch.nn.functional.cross_entropy(lg, torch.zeros(B, dtype=torch.long)); opt.zero_grad(); loss.backward(); opt.step()
            accs.append(float((lg.argmax(1) == 0).float().mean()))
        res[mode] = dict(acc_last200=round(float(np.mean(accs[-200:])), 3), sparsity=core.sparsity)
        print(mode, res[mode], flush=True)
    return res


if __name__ == "__main__":
    import sys
    print("sizes", SIZE)
    selftest(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
