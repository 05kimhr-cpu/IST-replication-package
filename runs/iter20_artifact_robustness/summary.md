# Iter 20 - Artifact-only robustness diagnostics

These diagnostics reuse released raw diffs and per-row NLI scores. They do
not rerun NLI, generation, or any GPU model.

## Full-diff-length subset

- Full sample: 1600 rows.
- Full raw diff <= 1,500 chars: 1022 rows (63.9%).
- Full raw diff > 1,500 chars: 578 rows (36.1%).

At tau=+0.56, the complete-diff subset preserves the rank ordering
diff->gold < diff->generated for every model and for both BART-MNLI
and DeBERTa. See subset_pass_rates.csv.

## Literal-copy diagnostics

| model | exact substring n | exact substring % | mean token overlap | median token overlap | mean gen tokens | corr(length, diff-NLI) |
|-------|------------------:|------------------:|-------------------:|---------------------:|----------------:|-----------------------:|
| codellama-7b | 0 | 0.0 | 0.472 | 0.462 | 8.48 | -0.042 |
| qwen2.5-coder-7b | 0 | 0.0 | 0.384 | 0.375 | 9.95 | 0.070 |
| deepseek-6.7b | 0 | 0.0 | 0.457 | 0.455 | 6.54 | 0.096 |

Interpretation: exact full-message copying from the diff is rare/absent
under this automatic check. Token overlap is expected because commit
messages legitimately reuse identifiers and action words from diffs;
therefore this diagnostic rules out literal regurgitation but does not
prove semantic adequacy.
