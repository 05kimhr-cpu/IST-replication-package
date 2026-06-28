# Diff-description accuracy audit - analysis
- Rated: **80** / 80 main items
- Rater: single independent blind human. Treat as a calibration audit, not a multi-rater gold standard.
- Human input: the shown diff is capped to the same 1,500-character premise used by the BART-MNLI diff diagnostic.

## 1. Primary outcome: human supported-rate

| source | supported-rate weighted | supported-rate raw | n raw | NLI pass-rate weighted |
|---|--:|--:|--:|--:|
| diff->gold | 37.3% | 42.5% | 40 | 18.8% |
| diff->generated | 24.5% | 27.5% | 40 | 36.4% |

Weighted rates use `cells.csv` post-stratification. Raw rates are diagnostic because the sample is intentionally balanced across source/NLI/intent cells.

## 2. Label distribution

| source | supported | neutral | contradicted | n |
|---|--:|--:|--:|--:|
| gold | 17 | 23 | 0 | 40 |
| gen | 11 | 28 | 1 | 40 |
| overall | 28 | 51 | 1 | 80 |

## 3. Reason distribution

| source | label | reason | n |
|---|---|---|--:|
| gen | contradicted | different_change | 1 |
| gen | neutral | diff_insufficient | 7 |
| gen | neutral | external_context | 4 |
| gen | neutral | missing_key_detail | 1 |
| gen | neutral | near_miss | 10 |
| gen | neutral | too_abstract | 6 |
| gen | supported | exact_core | 11 |
| gold | neutral | diff_insufficient | 2 |
| gold | neutral | external_context | 18 |
| gold | neutral | missing_key_detail | 1 |
| gold | neutral | near_miss | 2 |
| gold | supported | exact_core | 17 |

## 4. Human supported vs NLI pass @ tau=+0.56
- Raw binary agreement: **67.5%**
- Cohen's kappa (binary): **0.350**
- 3-class Cohen's kappa: **0.288**

| human \ NLI | supported | neutral | contradicted |
|---|---|---|---|
| supported | 22 | 3 | 3 |
| neutral | 21 | 25 | 5 |
| contradicted | 0 | 1 | 0 |

## 5. Per-cell supported-rate and NLI agreement

| cell | n | human supported-rate | NLI agreement |
|---|--:|--:|--:|
| gen|fail|intent | 10 | 20.0% | 80.0% |
| gen|fail|plain | 10 | 10.0% | 90.0% |
| gen|pass|intent | 10 | 30.0% | 30.0% |
| gen|pass|plain | 10 | 50.0% | 50.0% |
| gold|fail|intent | 10 | 10.0% | 90.0% |
| gold|fail|plain | 10 | 30.0% | 70.0% |
| gold|pass|intent | 10 | 50.0% | 50.0% |
| gold|pass|plain | 10 | 80.0% | 80.0% |

## 6. Intra-rater test-retest reliability
- Not available (out/retest.csv absent or unrated).
