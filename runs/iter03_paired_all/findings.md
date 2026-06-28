# Iteration 3 findings

## Scale
8 languages (cpp, cs, go, java, js, php, py, rust), 500 samples each, 52
paired perturbation triplets. Total paired rows with anchor hit: **1,630**.

## Headline result: meaning-direction blindness is near-universal across non-embedding metrics

| metric | exact_zero delta | |delta| < 0.01 | mean|delta| if nonzero |
|--------|------------------|----------------|-------------------------|
| BLEU   | **1630 / 1630**  | 1630 / 1630    | 0.000  |
| ROUGE-L| **1630 / 1630**  | 1630 / 1630    | 0.000  |
| CHRF   | 606 / 1630 (37%) | 1268 / 1630 (78%) | 0.017 |
| METEOR | 1372 / 1630 (84%)| 1377 / 1630 (84%) | 0.090 |

### Interpretation
1. **BLEU and ROUGE-L are categorically blind.** A 1-word substitution at
   a fixed position in the candidate produces the same n-gram breakage
   regardless of whether the substitute preserves or flips meaning. This is
   a property of word-level n-gram metrics and it holds uniformly across
   8 programming languages.
2. **CHRF responds on character n-grams, but marginally.** When syn and ant
   substitutes share overlapping character substrings with the anchor in
   different amounts, CHRF diverges — but the effect is small
   (mean|delta|_if_nonzero ≈ 0.017, compared to typical CHRF scores of
   0.5–0.9). 78% of pairs are indistinguishable under |delta|<0.01.
3. **METEOR leverages WordNet and does discriminate**, but only in ~16% of
   pairs and with moderate effect (mean|delta|_if_nonzero ≈ 0.090). In the
   other 84% of pairs, WordNet does not register syn-anchor as a synonym
   while marking ant-anchor as such, so the discriminative signal is
   absent.

### Direction of METEOR deltas
Mean METEOR(syn) − METEOR(ant) = +0.014 across all languages, i.e. when
METEOR moves, it tends to reward the synonym more than the antonym — the
right direction. But it rarely moves.

## What this lets us say for the paper
Strong empirical claim ready for a paper:

> *Across 1,630 controlled 1-word paired substitutions in CMG messages
> spanning 8 programming languages, the four most commonly reported
> evaluation metrics (BLEU, ROUGE-L, CHRF, METEOR) fail to reliably
> distinguish meaning-preserving from meaning-changing edits. BLEU and
> ROUGE-L give identical scores in 100% of pairs; CHRF and METEOR
> discriminate in fewer than 22% of pairs.*

Weaker but still useful:
- Consistency of the pattern across 8 languages strengthens construct
  validity of the finding (not a language quirk).
- The exact-zero result for BLEU/ROUGE is immune to noise — it is a
  structural consequence of word-level n-gram overlap.

## Open questions for iter 4
1. **Embedding metrics**: do BERTScore and CodeBERTScore discriminate more?
   This is the natural next question and the paper needs its answer to say
   what practitioners should use instead.
2. **DGF-family metrics** (from the TEMP work): should we include the
   reference DGF heuristic as an additional comparator? Caveat: we must
   not import TEMP code; we can only re-implement from the public DGF
   paper/guidelines if we want them in the benchmark. For iter 4 we skip
   and revisit once the embedding-metric result is in.
3. **Real model outputs**: the current design compares gold against
   perturbed gold. Iter 5 should verify that real CMG model outputs exhibit
   the 1-verb-error class often enough to make this blindness practically
   consequential.
4. **Construct-validity stress tests**: which automatic checks can show
   that synonym substitutes preserve the intended edit direction while
   antonym/disjoint substitutes change it? For iter 4 we trust the
   curated triplets; later iterations should add threshold sensitivity,
   per-kind breakdowns, and robustness checks over the perturbation
   vocabulary.

## Reframing notes
Current paper frame (post iter 3):

> Title working draft: *"Evaluation Blindness: Widely-Used CMG Metrics Do
> Not Discriminate Meaning-Direction in Single-Token Edits"*
>
> Key contributions:
>   1. A paired perturbation protocol for CMG that isolates meaning
>      direction while controlling word count.
>   2. Empirical evidence across 8 languages that BLEU, ROUGE-L, CHRF,
>      METEOR fail to discriminate.
>   3. (Pending iter 4) Quantification of how embedding-based metrics
>      close the gap.
>   4. (Pending iter 5) Evidence that real CMG model outputs contain the
>      class of single-verb errors this blindness prevents detecting.

Reframing triggers for iter 4+:
- **If BERTScore discriminates cleanly**: paper becomes a clean guidance
  story ("use BERTScore for CMG, here's the measurement reason").
- **If BERTScore also fails on the same paired probe**: paper becomes an
  indictment of the entire current metric toolkit, motivating a new
  diagnostic benchmark (which could cite DGF as one candidate direction
  without reproducing TEMP work).
- **If BERTScore discriminates partially**: paper adds a "when does which
  metric help" section.
