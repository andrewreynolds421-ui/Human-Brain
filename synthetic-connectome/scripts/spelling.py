#!/usr/bin/env python3
"""Stage 58: seeing the word's ending.

Stage 57: with varied frames the verb follows the subject, but for nouns never drilled the model often does not
know whether the noun is plural - it sees each word whole and cannot see the -s. Here each word is read as the
whole word PLUS its last one, two and three letters (as the letter and word units of Stages 38-42 would give:
the ending is part of what is seen). Everything else is Stage 57's check group (same pages, frames, held-out
nouns and frames, four networks).

    python scripts/spelling.py run --seed 0
"""
import argparse, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import frames as FR, picture_check as PC                                       # noqa: E402


def spelled(net):
    """replace the word embedding with word + ending embeddings"""
    torch = net.torch; nn = torch.nn; itos = net.itos

    class Spell(nn.Module):
        def __init__(s, base):
            super().__init__(); s.base = base; s.tabs = nn.ModuleList(); ids = []
            for k in (1, 2, 3):
                ends = sorted({w[-k:] for w in itos[4:] if len(w) > k})
                idx = {e: i + 1 for i, e in enumerate(ends)}
                ids.append(torch.tensor([idx.get(w[-k:], 0) if (n >= 4 and len(w) > k) else 0 for n, w in enumerate(itos)]))
                s.tabs.append(nn.Embedding(len(ends) + 1, base.embedding_dim, padding_idx=0))
            s.register_buffer("ids", torch.stack(ids))

        def forward(s, x):
            return s.base(x) + sum(t(s.ids[k][x]) for k, t in enumerate(s.tabs))
    net.m.e = Spell(net.m.e)
    params = list(net.m.parameters()) + list(net.num.parameters()) + list(net.look.parameters())
    net.opt = torch.optim.Adam(params, lr=2e-3)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("cmd", choices=("run",)); ap.add_argument("--seed", type=int, default=0); a = ap.parse_args()
    a.cond = "check"
    PC.make_page = FR.make_page; PC.built_tests = FR.built_tests
    PC.DATA = "/home/claude/models/spelling"; PC.OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "spelling")
    extra = set(FR.PREPS) | set(FR.HELD_PREPS) | set(FR.REL) | set(FR.HELD_REL) | {w for p in FR.PRED + FR.HELD_PRED for w in p}
    PC.PP.PREPS = sorted(set(PC.PP.PREPS) | extra)
    base_init = PC.CheckNet.__init__

    def init(self, vocab, seed, check):
        base_init(self, vocab, seed, check); spelled(self)
    PC.CheckNet.__init__ = init
    PC.run(a)


if __name__ == "__main__":
    main()
