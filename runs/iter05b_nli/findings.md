# Iteration 5 findings (coverage + NLI probe)

## 5a Anchor coverage in the corpus
Across 8 languages × 500 gold commit messages = 4,000 messages:
- **1,470 messages (36.8%)** contain at least one of our 52 anchor verb forms.
- Per-language coverage ranges from 30.0% (cpp) to 39.6% (php).
- Most common anchors: `add` (586), `added` (230), `remove` (166), `allow` (144).

This establishes that the class of messages on which the metrics we study
are demonstrably blind to meaning direction is not a marginal slice — it is
about one third of all commits in the corpus.

## 5b NLI discriminative probe — **decisive positive result**
Model: `facebook/bart-large-mnli` (general-purpose NLI, not domain-tuned).
Premise = gold, hypothesis = candidate. Probabilities over
{contradiction, neutral, entailment}.

### Overall (1,630 paired rows across 8 languages)
- mean entailment for synonym candidate: **0.942**
- mean entailment for antonym  candidate: **0.011**
- mean (entail − contradict) for synonym: +0.906
- mean (entail − contradict) for antonym:  −0.970
- fraction where entail_syn > entail_ant: **1,628 / 1,630 (99.9%)**
- strong discrimination (entail_delta > 0.5): 1,565 / 1,630 (96.0%)
- decisive signed_delta > 1.0 (scale is [-2, +2]): 1,568 / 1,630 (96.2%)

### Per language
All 8 languages exceed 99% on entail_syn > entail_ant and ≥92% decisive
signed_delta. The pattern is uniform.

## Putting it together — paper now has both halves
- **Negative**: 5 widely-reported metrics (BLEU, ROUGE-L, CHRF++, METEOR,
  BERTScore) cannot discriminate meaning-preserving from meaning-changing
  single-token edits in CMG messages across 8 languages (iter 2–4).
- **Positive**: A simple off-the-shelf NLI probe discriminates ~100% of the
  same paired comparisons (iter 5b).
- **Relevance**: 36.8% of commits use one of the anchor verbs in question,
  so the blindness is not confined to a niche slice (iter 5a).

The paper has the shape of a measurement-validity study with a concrete
actionable proposal, which is exactly IST territory.

## Open gaps before manuscript
1. Broaden the perturbation taxonomy beyond single verbs (target-noun
   swap, scope swap, negation, order flip). This guards against the
   reviewer criticism "verbs are a narrow slice of CMG errors."
2. Demonstrate relevance on real CMG model outputs (not only perturbed
   gold). An independent LLM baseline scored against gold with both the
   standard metric stack and the NLI probe would give a "real-world"
   section.
3. Automatic construct-validity checks for the perturbation vocabulary:
   per-kind breakdowns, threshold sensitivity, and inspection of weak
   lexical axes such as diagnostic nouns.
4. Calibration + threshold analysis for the NLI probe: at what threshold
   does it call a candidate faithful, and what's the operating-point
   trade-off?
5. Threats to validity section: tokenization, model version pinning,
   dataset style (MCMD), absence of multi-reference messages.

## Recommended iter 6 (decision)
The most IST-reviewer-satisfying next move is a **paper outline assembly
iteration**: plug every result we have into a manuscript skeleton, see
where sections are thin, and decide iter 7's gap-filler from that. This
avoids premature experimentation and produces an artifact the authors
can iterate on.

After iter 6 (outline), the likely iter 7 is one of:
- broaden perturbation (target-noun) — narrowest and fastest
- real model output study  — broadest and slowest
- automatic construct-validity diagnostics — cheapest to run without
  changing the experimental design
