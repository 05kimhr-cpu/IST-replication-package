# Iter 16 — Author-intent marker classification

Inputs: iter13 N=1600 rows, gold message text.
Classes are NOT mutually exclusive (a commit can be both 'ticket' and 'hotfix').
`has_intent = 1` iff ≥1 tag matches.

Total rows: 1600
Rows with ≥1 intent tag: 159 (9.9%)
Plain rows: 1441 (90.1%)

## Tag frequency (overall)

| tag      | count | share of N=1600 |
|----------|-------|-----------------|
| ticket   |    23 | 1.4%          |
| revert   |    15 | 0.9%          |
| hotfix   |    13 | 0.8%          |
| release  |     9 | 0.6%          |
| merge    |     2 | 0.1%          |
| trailer  |     0 | 0.0%          |
| bracket  |   109 | 6.8%          |
| ce_flag  |     8 | 0.5%          |
| nfc      |     1 | 0.1%          |
| branch   |     1 | 0.1%          |

## Tagged vs plain: diff→gold pass-rate (τ = +0.56)

| group       |    n   | pass (95% CI)                        | mean signed  |
|-------------|--------|--------------------------------------|--------------|
| has_intent  |    159 | 19/159 (11.9%) [7.8–17.9] | -0.008       |
| plain       |   1441 | 282/1441 (19.6%) [17.6–21.7] | +0.120       |

Two-proportion z = -2.333, p = 0.0196 (two-sided)

## Per-tag diff→gold pass-rate

| tag      |    n   | diff→gold pass ≥ +0.56 (95% CI)     | mean signed  |
|----------|--------|--------------------------------------|--------------|
| ticket   |     23 | 1/23 (4.3%) [0.8–21.0] | -0.064       |
| revert   |     15 | 0/15 (0.0%) [0.0–20.4] | -0.455       |
| hotfix   |     13 | 0/13 (0.0%) [0.0–22.8] | -0.074       |
| release  |      9 | 2/9 (22.2%) [6.3–54.7] | +0.077       |
| merge    |      2 | 0/2 (0.0%) [0.0–65.8] | +0.187       |
| trailer  |   0   | 0/0 (—)                              | —            |
| bracket  |    109 | 17/109 (15.6%) [10.0–23.6] | +0.069       |
| ce_flag  |      8 | 0/8 (0.0%) [0.0–32.4] | -0.348       |
| nfc      |      1 | 0/1 (0.0%) [0.0–79.3] | -0.912       |
| branch   |      1 | 0/1 (0.0%) [0.0–79.3] | -0.040       |

## Does intent-tagging affect gold→gen pass-rate? (per model)

| model            | intent pass-rate         | plain pass-rate          | p (z-test) |
|------------------|--------------------------|--------------------------|------------|
| codellama-7b     | 16/159 (10.1%)          | 140/1441 (9.7%)          | 0.889   |
| qwen2.5-coder-7b | 8/159 (5.0%)          | 105/1441 (7.3%)          | 0.292   |
| deepseek-6.7b    | 17/159 (10.7%)          | 139/1441 (9.6%)          | 0.673   |

## Interpretation

If `has_intent` group has materially *lower* diff→gold pass-rate than
`plain` (and z-test is significant), that supports §7: the gold is diff-weak
precisely because it encodes intent markers that are not in the diff.

