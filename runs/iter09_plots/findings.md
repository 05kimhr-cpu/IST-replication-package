# Iter 9 findings — paper-consolidation pass

## Artefacts produced

### Plots (`runs/iter09_plots/`)
- **fig1_delta_panel.png** — 2×3 grid of per-pair delta distributions
  (BLEU-4, ROUGE-L, CHRF++, METEOR, BERTScore F1, NLI signed).
  Verbs (blue) and nouns (orange) overlaid. Visual evidence of the
  blindness claim in one image.
- **fig2_sign_bars.png** — fraction of pairs with syn > ant/dis by
  metric. NLI entail/signed = 1.00 (verbs) / 0.94 (nouns). BERTScore
  = 0.51 (verbs) / 0.74 (nouns). BLEU/ROUGE = 0.00 (all ties).
- **fig3_nli_per_kind.png** — AUC + best-F1 by perturbation kind,
  monotonic: verb 0.999 > directional 0.968 > infrastructure 0.879 >
  close_entity 0.877 > diagnostic 0.810. Tells the "strength and
  honest limitation" story at a glance.

### Prose stubs (`docs/`)
- **threats_to_validity.md** — §7 draft covering construct, internal,
  external, probe-generalization, and label-validity threats.
- **related_work_stub.md** — ~14 anchor citations with the sentence
  each supports. Citation pass (bibtex + venue verification) is
  iter 11+.

## One finding worth a sentence in §4.4

CHRF++ on nouns has mean |delta| = 0.011 but prefers the synonym
in 90% of pairs (vs. 14% for verbs). Root cause is character-n-gram
overlap between software-jargon nouns (e.g. "function" and "method"
share the characters "t", "o"). This is **not** semantic detection —
the magnitude is too small to be usable (< 0.02) — but it is a real
and paper-worthy artefact: CHRF++'s sign on nouns is driven by
incidental character proximity, not by meaning.

Paper phrasing: *"CHRF++ on nouns shows a directional preference for
synonyms, but the magnitude is below a single score bin (< 0.02) and
is attributable to character proximity of software-jargon nouns
rather than semantic detection. The signal is structurally incapable
of surfacing meaning."*

## METEOR note

METEOR gives literal zero delta in 1273/1277 noun pairs. Visualizing
this as "1% syn wins" in the sign-bar chart is technically correct
but misleading — "99% ties" is the right interpretation. The paper
text should clarify that BLEU/ROUGE/METEOR bars near 0.00 mean the
metric is producing *identical* scores for both candidates in nearly
every pair, not that the metric prefers the meaning-changing
candidate.

## Manuscript readiness assessment

| paper element             | state    | remaining work                        |
|---------------------------|----------|---------------------------------------|
| §1 Introduction           | outline  | write prose, insert 5–7 citations     |
| §2 Related Work           | stub     | citation pass + prose                 |
| §3 Protocol               | ready    | convert paper-outline.md into prose   |
| §4 Blindness results      | ready    | tables + fig1 + fig2 + prose          |
| §5 NLI probe              | ready    | tables + fig3 + ROC + prose           |
| §6 Coverage               | ready    | 2 paragraphs + one table              |
| §7 Threats to validity    | drafted  | polish automatic robustness caveats   |
| §8 Implications           | outline  | write prose                           |
| §9 Conclusion             | outline  | short paragraph                       |

Overall: experiments and figures are done. What remains is manuscript
prose plus automatic robustness diagnostics. The paper could be submitted in
its current empirical state; iter 10 (real-model realism) would
strengthen the external-validity paragraph but is not a blocker.

## Recommended iter 10 (decision)

**Real-model realism probe**: generate commit messages for ~100 MCMD
diffs using one small LLM from `../LLM_Models/` (likely CodeLlama-7B
or Qwen-2.5-Coder-7B), score with the full metric stack + NLI probe,
and compare rankings to pairs where the model output happens to be
paraphrase-y vs hallucinated.

Rough shape:
1. Sample 100 diffs with non-trivial gold messages (≥ 8 tokens).
2. Prompt one model: `{diff}\nWrite a one-line commit message:`
3. Get model output. Score (gold, model_output) with 5 metrics + NLI.
4. Optional: also produce a manually-disturbed version of the model
   output (single-token antonym swap) to confirm the metric/probe
   behavior transfers to real generations.

Engineering: new script `10_real_model_eval.py` + one inference pass
(~30 min on GPU). The harder part is prompt curation and deciding
what "ground truth" means when the gold and model message are both
candidates with different emphases.

**Alternative iter 10 (if real-model is too slow)**: negation-insertion
perturbation class (`~X` → `do not X`). Cheaper, still novel, complements
the verb + noun story.

Tentative decision: try real-model first; fall back to negation if
the inference pipeline takes more than half a day.

## Open paper issues (not blocking)

1. Figure 1 could add a vertical line marking τ = +0.56 on the NLI
   panel to connect to the calibration story. Minor visual polish.
2. Whether to merge fig1 + fig3 into a single multi-panel figure to
   save figure budget in 10-page IST format. Depends on IST column
   layout; defer to layout pass.
3. Decide whether to include per-language plot (iter 3/4 per-language
   deltas). Currently we report the aggregate and one AUC-per-language
   table; a bar plot would add 1 figure but be somewhat redundant.
