# Iter 7 findings — target-noun paired perturbation

## Headline
The verb-side findings from iter 2–5 generalize to target-noun
substitutions. The specifics differ but the overall story — overlap and
embedding metrics are blind to meaning-direction, while a generic NLI
probe discriminates — is preserved.

## Setup
- 8 languages × 500 gold messages (= 4,000).
- 26 anchor noun forms across 4 triplet kinds:
  - `close_entity`: function/method/variable, class/type/module, parameter/argument/constant
  - `diagnostic`: error/exception/warning, bug/issue/feature
  - `infrastructure`: test/check/config, log/trace/output, comment/note/command
  - `directional`: import/include/export, input/entry/output
- Paired rows with anchor hit: **1,277**.

## Results (1,277 rows, 8 languages pooled)

| metric    | mean delta | mean|delta| | |delta|<0.01 | syn>dis | strong>0.5 |
|-----------|-----------:|------------:|-------------:|--------:|-----------:|
| BLEU-4    |     +0.0000|      0.0000 |   **1277/1277 (100%)** |   0/1277 (0%) | 0/1277 (0%) |
| ROUGE-L   |     +0.0000|      0.0000 |   **1277/1277 (100%)** |   0/1277 (0%) | 0/1277 (0%) |
| CHRF++    |     +0.0027|      0.0113 |   874/1277 (68%)       | 1152/1277 (90%) | 0/1277 (0%) |
| METEOR    |     +0.0001|      0.0002 |   1273/1277 (100%)     |    9/1277 (1%)  | 0/1277 (0%) |
| BERTScore |     +0.0042|      0.0067 |   1020/1277 (80%)      |  942/1277 (74%) | 0/1277 (0%) |
| NLI entail|     +0.4820|      0.4979 |     41/1277 (3%)       | **1203/1277 (94%)** | 654/1277 (51%) |
| NLI signed|     +0.7817|      0.8119 |     29/1277 (2%)       | **1199/1277 (94%)** | 768/1277 (60%) |

## Per-kind breakdown (key takeaway)

| kind               | n   | NLI syn>dis | NLI strong>0.5 |
|--------------------|----:|------------:|---------------:|
| close_entity       | 563 | 551/563 (98%) |  363/563 (64%) |
| diagnostic         | 253 | 241/253 (95%) |   70/253 (28%) |
| infrastructure     | 393 | 339/393 (86%) |  274/393 (70%) |
| **directional**    |  68 | **68/68 (100%)** | **61/68 (90%)** |

The directional kind (import/include/export, input/entry/output) is the
cleanest: NLI gets 100% and strongly discriminates in 90% of pairs. This
is the noun equivalent of verb antonyms and shows the NLI probe is
capable of crisp discrimination when the two substitutes differ in a
direction-like way.

The diagnostic kind (error/exception/warning, bug/issue/feature) is
fuzzier: "warning" and "error" are on a continuum, not opposites, so the
NLI probe's "strong" threshold (|signed_delta|>0.5) only fires in 28% of
pairs. This is an honest limitation: our probe reliably detects strong
referent changes but is weaker when the two candidate nouns are on an
adjacent semantic axis.

## What this adds to the paper
1. Blindness generalizes across two error classes (verbs in iter 2–5,
   nouns in iter 7) with uniform BLEU/ROUGE zero-delta and uniform
   BERTScore near-zero behavior across 8 languages.
2. The NLI probe is not verb-specific. It works on noun swaps too, with
   the expected caveat that fuzzier pairs (diagnostic) give smaller
   margins.
3. The by-kind breakdown is paper-friendly: it documents *when* the probe
   is strongest and *when* it is weakest, which is exactly what a
   practitioner needs.

## BERTScore observation worth a sentence
On noun substitutions BERTScore shows a mild directional preference for
synonyms (74% syn>dis) — unlike verbs where it was 50/50. Likely cause:
contextual embeddings of nouns are more clustered within topic (e.g.,
"function" and "method" really are neighbors) than for action verbs,
where antonyms often share context. The magnitude is still too small to
be usable (mean|delta| ≈ 0.007, far below score variance), but the
directional sign flip between verb and noun results is worth flagging.

## Open questions after iter 7
1. Are there perturbation classes we haven't covered that would flip the
   story? E.g., negation insertion, scope swap.
2. Would our NLI probe work on *real* CMG model outputs, not only
   perturbed-gold? That is the realism test.
3. Calibration: at what NLI entailment threshold should a practitioner
   call a candidate "faithful enough"?

## Recommended iter 8 (decision)
**NLI probe calibration and construct-validity robustness checks.**

Reasons:
- Calibration is cheap: reuses iter 5b + iter 7 data, produces paper-
  quality operating-point tables and a single ROC-style figure.
- Per-kind, threshold, and weak-axis diagnostics handle the most likely
  reviewer attack on construct validity without changing the study design.
- Real-model-output realism (iter 9 or later) is higher-value but also
  substantially more engineering effort (LLM inference + prompt design +
  result curation). Defer until calibration + robustness checks are in the bag.
