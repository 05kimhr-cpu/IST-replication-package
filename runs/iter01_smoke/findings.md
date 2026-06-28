# Iteration 1 findings

## Summary
Small smoke run (Python, 50 samples, 4 perturbations, 2 metrics) produced a
stronger signal than expected. The *expected* ordering of mean scores was:

    trivial (≈1.0) > paraphrase (mid) > meaning_change (low)

The *observed* ordering on ROUGE-L is:

    trivial (≈1.0) > meaning_change (0.879) > paraphrase (0.867)

And on BLEU-4:

    whitespace (1.0) > trailing_period (0.89) > meaning_change (0.85) > paraphrase (0.80)

## Two concrete findings
1. **Fragility asymmetry between metrics.** A cosmetic trailing-period flip
   does not move ROUGE-L (1.0) but moves BLEU-4 to 0.89 on average. Two
   metrics that are commonly reported side-by-side disagree on whether a
   meaning-irrelevant edit counts.

2. **Discrimination failure on 1-token semantic flips.** A 1-token *antonym*
   swap (e.g. `add` → `remove`) and a 1-token *synonym* swap
   (e.g. `add` → `introduce`) are the same *lexical* edit distance.  They are
   *opposite* in meaning implication. BLEU-4 and ROUGE-L both score the
   meaning-flipping edit **at least as high** as the meaning-preserving
   edit. A CMG practitioner using these metrics cannot distinguish "the
   model got the direction of the change wrong" from "the model said the
   same thing in other words."

## Caveats (must address in iter 2)
- n=14 for antonym, n=18 for synonym — small. Expand perturbation vocabulary
  to raise applicability.
- Python only. Does the pattern hold across 8 languages? — iter 2.
- Gold-as-candidate. Does the effect persist when the candidate is a real
  model output? — iter 3.
- Only BLEU-4 and ROUGE-L. Embedding metrics (BERTScore, CodeBERTScore) may
  discriminate meaning flips.
- Lexical distance of swap pairs is not matched: `add→introduce` has zero
  character overlap, while `add→remove` shares no chars either, but word
  length differs. Iter 2 should match perturbation pairs on length and
  character-edit-distance to isolate the meaning dimension.

## Reframing options for iter 2

### Option A — double down on "discrimination failure" (strongest if it scales)
Headline: *Surface-overlap metrics assign indistinguishable scores to
meaning-preserving and meaning-changing single-token edits in CMG messages.*

Plan:
- Build matched 1-token perturbation pairs (antonym vs synonym) controlling
  for character-edit-distance and word length.
- Scale to all 8 languages.
- Add METEOR, CHRF++, BERTScore; see which (if any) metrics actually
  distinguish the two classes.
- If embedding metrics discriminate: paper becomes a guidance study
  ("use embedding metrics, here's why"). If they also fail: paper becomes
  a stronger indictment, motivating a diagnostic benchmark.

### Option B — broaden to fragility spectrum (safer, less punchy)
Keep both the fragility axis (BLEU vs ROUGE on trivial edits) and the
discrimination axis, report the full matrix across languages and metrics.
Less of a single headline finding.

### Option C — pivot to the assumption under iter 1
Iter 1 used gold-as-candidate. The discrimination-failure observation also
holds for gold-vs-perturbed-gold regardless of the diff. One could argue
that the diff is *implicit* in "what counts as a meaning-preserving swap",
but that is a stretch. Iter 2 should ground the perturbation design in the
*diff* so the claim "meaning-preserving/changing" is anchored in code
reality, not a hand-curated word list.

## Recommended iter 2 (if user agrees)
Take Option A plus the iter-1 caveats, starting with:

1. Expand perturbation vocabulary: pull all action-verb tokens from the
   8-language message corpus, manually (or LLM-assisted) build 30–50
   matched synonym/antonym pairs, filter by character-edit-distance band.
2. Scale smoke to all 8 languages, still gold-as-candidate, compute BLEU-4,
   ROUGE-L, METEOR, CHRF++ (still no heavy embedding metrics).
3. Primary analysis: for each metric, is the *distribution* of scores on
   synonym swaps distinguishable from antonym swaps? Effect size, paired
   test per sample.
4. Threshold for pressing onward vs pivoting: if effect sizes are small
   and noisy across languages, pivot toward the fragility axis (Option B).
   If the discrimination failure replicates cleanly, that becomes the
   paper's core claim.

## Status
Preliminary. Do not cite externally. Results in `results.csv`, `summary.md`.
