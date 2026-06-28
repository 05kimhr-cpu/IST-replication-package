# Iter 19 - Threshold and paired NLI diagnostics

This CPU-only diagnostic reuses iter13 BART-MNLI and iter14 DeBERTa
per-row signed scores. It adds no new model inference.

## Paired result at tau = +0.56

| backbone | model | both pass | gold only | gen only | both fail | gen-gold pp |
|----------|-------|----------:|----------:|---------:|----------:|------------:|
| BART-MNLI | codellama-7b | 163 | 138 | 432 | 867 | 18.4 |
| BART-MNLI | qwen2.5-coder-7b | 170 | 131 | 442 | 857 | 19.4 |
| BART-MNLI | deepseek-6.7b | 135 | 166 | 404 | 895 | 14.9 |
| DeBERTa-v3 | codellama-7b | 237 | 180 | 486 | 697 | 19.1 |
| DeBERTa-v3 | qwen2.5-coder-7b | 304 | 113 | 583 | 600 | 29.4 |
| DeBERTa-v3 | deepseek-6.7b | 229 | 188 | 479 | 704 | 18.2 |

Interpretation: for every model/backbone pair, more rows are
generated-only passes than gold-only passes. The headline
diff->gold < diff->generated ordering is therefore a paired
row-level effect, not only an aggregate pass-rate artefact.

See threshold_sweep.csv for the same comparison over tau values
-0.50, 0.00, 0.25, 0.56, 0.75, and 0.90.
