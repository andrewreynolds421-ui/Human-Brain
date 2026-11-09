# Stage 37 supporting checks (scratch scripts, kept for traceability)

Run from the repository root with `PYTHONPATH=.`; `scale.py`, `online2.py` and `gap.py` read the
helper definitions out of `conj.py` / `online2.py` in this folder, so first fix their `/tmp/` paths
to `results/words/stage37_checks/`. The main Stage 37 table is `scripts/analysis/stage37.py`.

## conj.py: best fitted linear predictor (true next letters, 2,000 words), next-letter top-1 / bits
    s1 | context: known 0.275/3.42b held 0.155/3.75b | conj: known 0.203/3.85b held 0.110/4.12b | conj+ctx: known 0.294/3.37b held 0.156/3.73b
    s2 | context: known 0.314/3.35b held 0.147/3.89b | conj: known 0.211/3.82b held 0.122/4.16b | conj+ctx: known 0.324/3.31b held 0.156/3.86b
    s3 | context: known 0.310/3.34b held 0.167/3.85b | conj: known 0.205/3.79b held 0.138/4.14b | conj+ctx: known 0.326/3.30b held 0.170/3.82b
    s4 | context: known 0.282/3.42b held 0.178/3.82b | conj: known 0.203/3.84b held 0.170/4.10b | conj+ctx: known 0.299/3.37b held 0.185/3.80b

## scale.py: same, next-letter top-1 against amount of reading
    s1   2000 words read | context known 0.287 held 0.148 | conj+ctx known 0.302 held 0.154
    s1   8000 words read | context known 0.282 held 0.168 | conj+ctx known 0.312 held 0.176
    s1  40000 words read | context known 0.292 held 0.178 | conj+ctx known 0.331 held 0.177
    s2   2000 words read | context known 0.313 held 0.148 | conj+ctx known 0.322 held 0.154
    s2   8000 words read | context known 0.311 held 0.169 | conj+ctx known 0.345 held 0.171
    s2  40000 words read | context known 0.323 held 0.167 | conj+ctx known 0.366 held 0.178

## online2.py: letter accuracy, fitted (true letters) vs online percept-taught rule, unscaled inputs, 20,000 words
    s1 context   FITTED: known 0.874 held 0.852 | online eta 0.1: 0.868 0.842 | 0.3: 0.869 0.834 | 1.0: 0.861 0.842
    s1 conj+ctx  FITTED: known 0.883 held 0.860 | online eta 0.1: 0.871 0.845 | 0.3: 0.870 0.833 | 1.0: 0.863 0.846
    s2 context   FITTED: known 0.884 held 0.829 | online eta 0.1: 0.853 0.809 | 0.3: 0.853 0.808 | 1.0: 0.848 0.806
    s2 conj+ctx  FITTED: known 0.896 held 0.831 | online eta 0.1: 0.857 0.806 | 0.3: 0.873 0.830 | 1.0: 0.851 0.810

## gap.py: where the gap to the fitted predictor goes (context, 20,000 words, 4 networks, letter accuracy)
    fitted, true letters                   known 87.1 ± 1.0  held 84.0 ± 1.1
    fitted, winner of evidence             known 86.0 ± 0.9  held 82.3 ± 1.7
    online, fixed rate 0.1                 known 86.0 ± 0.6  held 82.0 ± 2.0
    online, settling rate 1/(1+t/5000)     known 84.9 ± 2.1  held 81.0 ± 2.0
    online, settling rate 1/(1+t/20000)    known 85.0 ± 1.8  held 81.2 ± 2.6
