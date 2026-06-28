# Iter 14 — NLI backbone sensitivity: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

τ = +0.56 (paper's BART-calibrated threshold reused as-is for cross-comparison)
N = 1600

## Gold → Generated (DeBERTa signed)

| model            | pass ≥ +0.56 (95% CI)     | mean signed |
|------------------|---------------------------|-------------|
| codellama-7b     | 135/1600 (8.4%)  [7.2–9.9] | -0.026 |
| qwen2.5-coder-7b | 104/1600 (6.5%)  [5.4–7.8] | +0.004 |
| deepseek-6.7b    | 122/1600 (7.6%)  [6.4–9.0] | -0.075 |

## Diff → X (DeBERTa signed)

| probe                  | pass ≥ +0.56 (95% CI)     | mean signed |
|------------------------|---------------------------|-------------|
| diff → gold            | 417/1600 (26.1%)  [24.0–28.3] | +0.231 |
| diff → gen (codellama-7b  ) | 723/1600 (45.2%)  [42.8–47.6] | +0.440 |
| diff → gen (qwen2.5-coder-7b) | 887/1600 (55.4%)  [53.0–57.9] | +0.547 |
| diff → gen (deepseek-6.7b ) | 708/1600 (44.2%)  [41.8–46.7] | +0.438 |
