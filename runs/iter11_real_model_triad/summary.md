# Iter 11 — Multi-model + diff-grounded NLI

Samples: 80 (reused iter10 selection)

## Gold ↔ Generated: NLI pass-rates at paper's τ = +0.56

| model              | n | signed ≥ +0.56 | signed ≥ +0.88 | mean signed | mean BERTScore |
|--------------------|---|----------------|----------------|-------------|----------------|
| codellama-7b       |  80 |   6/80 (7.5%) |   4/80 (5.0%) | +0.038 | +0.860 |
| qwen2.5-coder-7b   |  80 |   7/80 (8.8%) |   3/80 (3.8%) | +0.054 | +0.864 |
| deepseek-6.7b      |  80 |   8/80 (10.0%) |   5/80 (6.2%) | -0.038 | +0.847 |

## Diff ↔ X: does the source-of-truth entail each reference or candidate?

| probe                    | n | signed ≥ +0.56 | mean signed | mean entail |
|--------------------------|---|----------------|-------------|-------------|
| diff → gold              |  80 | 10/80 (12.5%) | +0.114 | +0.268 |
| diff → gen (codellama-7b     ) |  80 | 37/80 (46.2%) | +0.520 | +0.549 |
| diff → gen (qwen2.5-coder-7b ) |  80 | 31/80 (38.8%) | +0.387 | +0.455 |
| diff → gen (deepseek-6.7b    ) |  80 | 28/80 (35.0%) | +0.386 | +0.464 |

## Interpretation cheatsheet

- If `diff→gold` also fails (signed near 0 or <<0.56): **reference itself is not diff-entailing** → construct-validity critique is empirical. Gold = author *intent*, not diff *content*.
- If `diff→gold` passes but `diff→gen` fails across all models: **models underfit content**, gold is fine, reference framework salvageable.
- If `diff→gold` and `diff→gen` both pass: NLI works on both — iter10's failure is a **calibration-shift** issue between paired and free-form regimes.
