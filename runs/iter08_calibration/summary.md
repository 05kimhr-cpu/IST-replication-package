# Iter 8 — NLI probe calibration

- Candidates: **5814** (2907 meaning-preserving + 2907 meaning-changing)
- Sources: iter 5b (verb pairs, 1630×2) + iter 7 (noun pairs, 1277×2)
- Task: binary classifier — flag candidate as 'faithful to gold'

## Overall operating points

| score   | AUC  | best-F1 τ | best F1 | prec  | recall | FPR≤0.05 τ | TPR@FPR0.05 |
|---------|-----:|----------:|--------:|------:|-------:|-----------:|------------:|
| entail  | 0.9624 | +0.610 | 0.8971 | 0.8763 | 0.9188 | +0.900 | 0.7860 |
| signed  | 0.9579 | +0.560 | 0.8908 | 0.8865 | 0.8951 | +0.880 | 0.7736 |

## Per-kind AUC and best operating point (signed score)

| kind | n+ | n- | AUC | best-F1 τ | best F1 | prec | recall |
|------|---:|---:|----:|----------:|--------:|-----:|-------:|
| noun_close_entity | 563 | 563 | 0.8772 | +0.560 | 0.8123 | 0.8024 | 0.8224 |
| noun_diagnostic | 253 | 253 | 0.8100 | +0.880 | 0.7556 | 0.7204 | 0.7945 |
| noun_directional | 68 | 68 | 0.9682 | -0.680 | 0.9286 | 0.9028 | 0.9559 |
| noun_infrastructure | 393 | 393 | 0.8791 | +0.640 | 0.8316 | 0.8610 | 0.8041 |
| verb | 1630 | 1630 | 0.9988 | -0.400 | 0.9859 | 0.9859 | 0.9859 |

## Per-language F1 at overall best-F1 threshold (signed score)

Threshold used: τ = +0.560

| language | n+ | n- | AUC | F1@τ | prec | recall |
|----------|---:|---:|----:|-----:|-----:|-------:|
| cpp | 267 | 267 | 0.9685 | 0.9088 | 0.9037 | 0.9139 |
| cs | 338 | 338 | 0.9663 | 0.9061 | 0.9286 | 0.8846 |
| go | 399 | 399 | 0.9428 | 0.8738 | 0.8631 | 0.8847 |
| java | 352 | 352 | 0.9573 | 0.8899 | 0.9083 | 0.8722 |
| js | 398 | 398 | 0.9530 | 0.8856 | 0.8585 | 0.9146 |
| php | 379 | 379 | 0.9600 | 0.8912 | 0.8960 | 0.8865 |
| py | 385 | 385 | 0.9648 | 0.8932 | 0.8852 | 0.9013 |
| rust | 389 | 389 | 0.9549 | 0.8866 | 0.8691 | 0.9049 |

