# Iter 8 findings — NLI probe calibration + construct-validity infrastructure

## Headline
The NLI probe is paper-ready. Calibration over the 5,814 labeled
candidates from iter 5b + iter 7 gives **AUC = 0.962** overall, with
clean per-language stability (F1 0.87–0.91 across 8 languages at the
same operating threshold). This file records the automatic calibration
results only; no human-evaluation artefacts are part of the release.

## Calibration (5,814 candidates)

### Operating points (signed score, entail − contradict)

| regime                | τ      | F1     | precision | recall | FPR    | TPR    |
|-----------------------|-------:|-------:|----------:|-------:|-------:|-------:|
| best F1               | +0.560 | 0.8908 | 0.8865    | 0.8951 | 0.1146 | 0.8951 |
| FPR ≤ 0.05 (conservative) | +0.880 | —     | —         | —      | 0.0500 | 0.7736 |

AUC = 0.9579 (signed), 0.9624 (entail).

### Per-kind AUC (signed score)

| kind                | n+   | AUC    | best F1 | comment                                                    |
|---------------------|-----:|-------:|--------:|------------------------------------------------------------|
| verb                | 1630 | 0.9988 | 0.9859  | near-perfect: verbs have crisp antonym pairs               |
| noun_directional    |   68 | 0.9682 | 0.9286  | small n but crispest noun kind (import↔export etc.)        |
| noun_infrastructure |  393 | 0.8791 | 0.8316  | —                                                          |
| noun_close_entity   |  563 | 0.8772 | 0.8123  | function/method vs function/variable                       |
| noun_diagnostic     |  253 | 0.8100 | 0.7556  | error/warning on a continuum — the probe's weak spot       |

### Per-language stability (signed, τ = +0.560)

F1 ranges 0.87 (go) – 0.91 (cpp). AUC ranges 0.94 (go) – 0.97 (cpp).
**No language is an outlier** — the probe's discrimination is not driven
by a single language's idiom.

## What this unlocks for the paper

1. **Practitioner recommendation is now concrete.** Instead of saying
   "NLI probes help," the paper can say: "treat signed_score ≥ +0.56 as
   a faithful-enough heuristic; at this threshold 89% of meaning-
   preserving candidates are flagged and only 11% of meaning-changing
   candidates sneak through. For high-precision gating, move to ≥ +0.88
   (FPR ≤ 5%, TPR 77%)."
2. **Per-kind table is paper-friendly.** It simultaneously shows the
   probe's strength (verbs, directional nouns) and its honest limitation
   (diagnostic nouns near adjacent semantic axes), which pre-empts the
   "you only showed this works where it works" critique.
3. **ROC figure is ready.** `roc.png` in the run directory. Minor visual
   polish might improve it, but the content is complete.

## Construct-validity caveat

The paired labels are produced by curated lexical triplets, not by an
independent semantic validation pass. The automatic calibration should
therefore be interpreted as a stress test for metric sensitivity under
controlled edits. Vocabulary quality remains a limitation, especially
for idiomatic near-synonyms in commit-message English.

## Open questions after iter 8

1. **Real-model realism**: does the probe show the same ordering on
   actual CMG model output rather than perturbed-gold candidates?
   Iter 5b/7/8 used perturbed-gold; later iterations address this with
   three 7B code models.
2. **Plots, threats-to-validity text, related-work citations** — all
   plumbing for manuscript.
3. Whether to explore negation insertion and scope swap as additional
   perturbation classes. These would be cheap (script patterns) but
   the marginal reviewer value over "verbs + nouns" is debatable.

## Recommended iter 9 (decision)

**Iter 9 = paper-consolidation pass:** complete the plots (delta
distributions + sign bars for BLEU/ROUGE/CHRF/METEOR/BERTScore, NLI
discrimination histogram, per-language bar), write the threats-to-
validity section prose, and stub the related-work section with the
5–6 anchor citations (Mathur 2020, BLEURT, Reiter 2018, Dong 2022,
Liu 2020, Post Kaster on BLEU). All of this is low-engineering,
high-leverage for manuscript readiness.

**Iter 10 = real-model realism:** generate commit messages for 100
diffs with one small LLM, score with full metric stack + NLI probe,
compare orderings. This is engineering-heavy (prompt design, inference
harness, curation) and should only start once iter 9 is in the bag.

Rationale: the empirical story is now strong enough to write up. The
risk is that iter 10 runs long and delays a submittable draft. Iter 9
produces manuscript-ready artefacts from material we already have, so
even if iter 10 slips we can submit. If iter 10 succeeds it becomes
Section 6 of the paper; if not, it's future work.
