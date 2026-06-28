# Replication package — Reference Proximity Is Not Diff Faithfulness

This package contains the data, scripts, and outputs for the paper's
controlled-perturbation study (RQ1), premise-swap NLI diagnostics (RQ2),
human triangulation audit (RQ3), and robustness checks (RQ4).

## Contents

```
data/raw/                 Eight-language commit corpus (500 commits/language).
                          Java, C#, C++, Python, JavaScript are from MCMD;
                          Go, PHP, Rust were collected from GitHub in the same
                          tokenized format. Fields: diff_id, repo, sha, time, diff, msg.
scripts/                  Numbered pipeline (01–20): perturbation construction,
                          metric scoring, NLI calibration, scaled triad generation,
                          DeBERTa sensitivity, prompt ablation, intent classifier,
                          RQ1 CSV audit, threshold/artifact diagnostics.
runs/                     Per-stage outputs (summary.md + CSVs) for every iteration,
                          including iter13_scaled_triad (BART), iter14_deberta_sanity,
                          iter08_calibration (tau*, AUC), iter15/16/18/19/20.
human_eval/               RQ3 human audit:
  out/human_final_consolidated.csv   Final 80 adjudicated labels (record of truth).
  out/찐_rater_final_filled.csv       Rater 1 labels (80).
  out/rater2_blind_filled_1_80.csv    Rater 2 blind labels (80).
  out/adjudication_disagreements_filled_1_11.csv   11 adjudicated disagreements.
  out/key.csv, cells.csv              Hidden source/NLI map + post-stratification weights.
  analyze_human_eval.py               Agreement / kappa / supported-rate analysis.
  DESIGN.md, RUBRIC.md, README_RATER.md   Audit design + rater instructions.
figures/                  Figure/table generators.
```

## Reproducing the key numbers

- RQ1 (metric blindness, Table 2): `scripts/18_rq1_csv_audit.py`; see `runs/iter18_rq1_csv_audit/`.
- Calibration (tau* = +0.56, AUC 0.958): `runs/iter08_calibration/summary.md`.
- RQ2 pass-rates (Table 3): `runs/iter13_scaled_triad/summary.md` (BART) and
  `runs/iter14_deberta_sanity/summary.md` (DeBERTa).
- RQ3 human audit: `cd human_eval && python3 analyze_human_eval.py` over `out/`.
- RQ4 robustness (Table 4): `runs/iter15/16/19/20`.

All manuscript numbers are computed from these released files. The diff text uses
the MCMD tokenization (`<nl>` line breaks, spaces around punctuation).
