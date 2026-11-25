# VEGA runs on GitHub's machines

Edit `jobs.json` on this branch and push: `.github/workflows/vega.yml` starts one job per entry (all at once, up to
6 h each) and each job pushes its results to `runs/<name>/` (result JSON, model `.pt`, last part of the log,
`status.txt`). Entry fields: `name`, `flags` (for scripts/hippo_babi.py), `seed` (reading network 0-3),
`questions`, `env` (extra variables, e.g. `DUMP=1`).
