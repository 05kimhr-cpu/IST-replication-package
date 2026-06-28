# Iter 15 — Prompt-sensitivity ablation (CodeLlama-7B, N = 1600)

Threshold τ = +0.56 (paper's BART calibration).

## Gold → Generated

| prompt   | pass ≥ +0.56 (95% CI)     | mean signed |
|----------|---------------------------|-------------|
| intent   | 165/1600 (10.3%)  [8.9–11.9] | -0.006 |
| content  | 34/1600 (2.1%)  [1.5–3.0] | -0.008 |
| baseline | 156/1600 (9.8%)  [8.4–11.3] | -0.013 |

## Diff → Generated

| prompt   | pass ≥ +0.56 (95% CI)     | mean signed |
|----------|---------------------------|-------------|
| intent   | 732/1600 (45.8%)  [43.3–48.2] | +0.466 |
| content  | 742/1600 (46.4%)  [43.9–48.8] | +0.499 |
| baseline | 595/1600 (37.2%)  [34.9–39.6] | +0.401 |
