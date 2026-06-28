# Diff-description accuracy audit - analysis
- Rated: **80** / 80 main items
- Rating source: `out/human_final_consolidated.csv`.
- Human labels: final consolidated labels after adjudication.
- Human input: the shown diff is capped to the same 1,500-character premise used by the BART-MNLI diff diagnostic.

## 1. Primary outcome: human supported-rate

| source | supported-rate weighted | supported-rate raw | n raw | NLI pass-rate weighted |
|---|--:|--:|--:|--:|
| diff->gold | 44.5% | 45.0% | 40 | 18.8% |
| diff->generated | 27.6% | 30.0% | 40 | 36.4% |

Weighted rates use `cells.csv` post-stratification. Raw rates are diagnostic because the sample is intentionally balanced across source/NLI/intent cells.

## 2. Label distribution

| source | supported | neutral | contradicted | n |
|---|--:|--:|--:|--:|
| gold | 18 | 21 | 1 | 40 |
| gen | 12 | 26 | 2 | 40 |
| overall | 30 | 47 | 3 | 80 |

## 3. Reason distribution

| source | label | reason | n |
|---|---|---|--:|
| gen | contradicted | different_change | 2 |
| gen | neutral | diff_insufficient | 9 |
| gen | neutral | external_context | 4 |
| gen | neutral | missing_key_detail | 1 |
| gen | neutral | near_miss | 8 |
| gen | neutral | too_abstract | 4 |
| gen | supported | exact_core | 12 |
| gold | contradicted | different_change | 1 |
| gold | neutral | diff_insufficient | 2 |
| gold | neutral | external_context | 17 |
| gold | neutral | missing_key_detail | 1 |
| gold | neutral | near_miss | 1 |
| gold | supported | exact_core | 18 |

## 4. Human supported vs NLI pass @ tau=+0.56
- Raw binary agreement: **62.5%**
- Cohen's kappa (binary): **0.250**
- 3-class Cohen's kappa: **0.269**

| human \ NLI | supported | neutral | contradicted |
|---|---|---|---|
| supported | 22 | 5 | 3 |
| neutral | 20 | 23 | 4 |
| contradicted | 1 | 1 | 1 |

## 5. Per-cell supported-rate and NLI agreement

| cell | n | human supported-rate | NLI agreement |
|---|--:|--:|--:|
| gen|fail|intent | 10 | 30.0% | 70.0% |
| gen|fail|plain | 10 | 20.0% | 80.0% |
| gen|pass|intent | 10 | 30.0% | 30.0% |
| gen|pass|plain | 10 | 40.0% | 40.0% |
| gold|fail|intent | 10 | 10.0% | 90.0% |
| gold|fail|plain | 10 | 40.0% | 60.0% |
| gold|pass|intent | 10 | 50.0% | 50.0% |
| gold|pass|plain | 10 | 80.0% | 80.0% |

## 6. Intra-rater test-retest reliability
- Not available (out/retest.csv absent or unrated).
