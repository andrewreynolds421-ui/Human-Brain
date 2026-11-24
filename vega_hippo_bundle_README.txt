VEGA hippocampus bundle (24 November 2026)

Everything scripts/hippo_pointer.py, scripts/hippo_babi.py and scripts/babi_eval.py open, at their original
paths. Unpack at / (tar -xzf vega_hippo_bundle.tar.gz -C /) so the files land in /home/claude/..., or unpack
elsewhere and make that folder's home/claude appear as /home/claude.

Contents
  home/claude/synthetic-connectome/scripts/        all project code (the three scripts import ~10 other modules)
  home/claude/synthetic-connectome/synth/          helper package
  home/claude/synthetic-connectome/data/noun_pairs.json, data/pictures.npz   read when the reading network is built
  home/claude/synthetic-connectome/README.md, METHODS.md, AUDIT.md, HANDOFF.md
  home/claude/models/day_night/year4_cycle/        the reading networks (Stage 73), 4 networks
  home/claude/models/hippocampus/                  Stage 75-77 hippocampus models (pointer_s* = Stage 77)
  home/claude/models/babi_all/                     Stage 79 hippocampus models (all 20 bAbI tasks)
  home/claude/corpora/books/, corpora/books_more/  Project Gutenberg books (public domain) - the reading
                                                   network's vocabulary is rebuilt from them at load time
  home/claude/corpora/babi/                        bAbI v1.2 (Weston et al. 2015; Facebook, CC BY 3.0 - see LICENSE.txt)

Not included
  /home/claude/models/childes/adult_speech.pkl - not opened by these three scripts, and derived from CHILDES
  (TalkBank), which may not be republished; this release is public.
  Stage 80 models (training when the bundle was made).

Needs: Python 3 with torch and numpy (tested with the workspace's versions). Run from
/home/claude/synthetic-connectome with PYTHONPATH=. e.g.
  python scripts/babi_eval.py --seed 0 --tasks 1

How to rejoin (this branch): the bundle is split into four pieces under 100 MB (GitHub's limit for one file).
  cat vega_hippo_bundle.tar.gz.part0* > vega_hippo_bundle.tar.gz
  sha256sum -c vega_hippo_bundle.tar.gz.sha256      # checks the whole file and each piece
  tar -xzf vega_hippo_bundle.tar.gz -C /
