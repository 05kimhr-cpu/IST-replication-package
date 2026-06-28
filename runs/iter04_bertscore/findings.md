# Iteration 4 findings

## Headline
**BERTScore also fails the paired meaning-direction test.**

Across 1,630 paired substitutions in 8 languages:
- `exact_zero` BERTScore delta: 0 / 1630 (it is a floating-point cosine, so
  exact zero is not expected; that's fine)
- `|delta| < 0.01`: **1,357 / 1,630 (83%)**
- sign split: syn > ant in **825/1630**, syn < ant in **805/1630** → essentially
  a coin flip
- `mean|delta|`: 0.0051 (with typical BERTScore F1 around 0.93–0.97)

Per-language, every language shows the same pattern: near-zero mean, tiny
absolute deviation, near-even sign split.

## Interpretation for the paper
This is the critical extension of iter 3. The working assumption in much of
CMG evaluation literature is that embedding-based metrics (BERTScore in
particular) capture semantic content that surface overlap cannot. Under our
controlled paired design, that assumption **does not hold for single-token
meaning-direction edits in real commit messages**.

The failure is especially clean because:
1. The substitutions are real English action verbs (`add`/`remove`,
   `enable`/`disable`, `show`/`hide`, ...).
2. They differ only in one position per message.
3. The reference and the two candidates are otherwise identical.

Contextual embeddings appear to encode "action verb acting on this object"
but not the semantic direction of the action. This matches known BERTScore
failure modes for negation and antonymy in MT, but had not been
systematically demonstrated for CMG.

## What this lets us say for the paper now
A sharp, well-controlled empirical claim:

> *Across 1,630 controlled single-word paired substitutions in commit
> messages spanning 8 programming languages, five widely-reported CMG
> evaluation metrics (BLEU-4, ROUGE-L, CHRF++, METEOR, BERTScore) all fail
> to reliably distinguish meaning-preserving from meaning-changing edits.
> BLEU-4 and ROUGE-L give identical scores in 100% of pairs; CHRF++ gives
> |delta|<0.01 in 78% of pairs; METEOR in 84%; BERTScore in 83%. BERTScore's
> sign of delta is essentially uniform (825 vs 805 of 1,630).*

## Paper outline — after iter 4 (v1)

```
1. Introduction
   - CMG evaluation relies heavily on BLEU/ROUGE/BERTScore (cite prior
     CMG works).
   - What does "a 0.8 BLEU" mean for commit messages?

2. Paired perturbation protocol
   - Anchor verb substitution with matched 1-word syn/ant pairs.
   - Rationale: identical word count, identical substitution position.
   - Vocabulary: 13 anchor verb families, 52 surface forms.
   - Construct validity: curated list, manual check on ~5%.

3. Empirical results
   - 8 languages × 1,630 paired rows × 5 metrics.
   - Table 1: per-metric, per-language discrimination stats.
   - Figure 1: BERTScore delta distributions per language.
   - Figure 2: sign-of-delta fractions per metric.

4. Why it happens
   - n-gram arithmetic for BLEU/ROUGE.
   - Character overlap inventory for CHRF.
   - WordNet coverage for METEOR.
   - Contextual-embedding similarity for BERTScore: verbs with opposite
     meanings still occupy similar contextual neighborhoods in CMG domain.

5. Towards discriminative evaluation (iter 5+)
   - Probe 1: NLI-based candidate-vs-reference (will test in iter 5).
   - Probe 2: diff-grounded verification (DGF family; out of scope for
     this paper but discussed).

6. Threats to validity
   - Construct: anchor curation; automatic per-kind robustness checks.
   - Internal: tokenization, metric impl version pinning.
   - External: MCMD dataset style; other CMG corpora may differ.
   - Replication: all scripts deterministic.

7. Implications and recommendations
   - Do not interpret 0.01 BERTScore differences as meaning differences.
   - Report paired probes alongside absolute scores.
   - Develop diagnostic benchmarks.

8. Conclusion.
```

## Iter 5 plan (decision made: go for positive probe + realism check)

Two pieces, in order:
1. **Realism check (half day)**: report anchor verb coverage in the corpus
   so we can argue "this class of error matters, N% of gold messages
   contain one of our anchor verbs."
2. **Positive probe (NLI)**: evaluate a small NLI model on the same
   paired rows. If it discriminates syn/ant (entailment high for syn,
   low for ant), we have a positive section 5 result. If it also fails,
   the paper's Section 5 becomes "we tried a natural fix, it didn't
   work — here's why we think a CMG-specific probe is needed."
3. Model choice: `roberta-large-mnli` or `facebook/bart-large-mnli`,
   run with bidirectional NLI, take entailment probability.

Status: preliminary. Results hold up statistically and across all 8
languages. Reframing not needed at this point; the paper's core claim is
now robust.
