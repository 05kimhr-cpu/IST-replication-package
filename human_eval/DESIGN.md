# Diff-Description Accuracy Human Audit — Design

Target venue: Information and Software Technology (IST)
Paper: "Commit Messages Are Not Diff Summaries: A Construct-Validity Critique of Reference-Based CMG Evaluation"
Created: 2026-06-16

## 1. Why this study exists

The paper's central claim is a *construct* claim: reference-match metrics measure
"reference proximity," whereas the signed BART-MNLI diagnostic is an automatic
proxy for whether a commit message is grounded in the code diff. The human audit
does not treat NLI as ground truth. It asks a narrower question: using the same
shown diff scope as the NLI premise, does a human judge the message as an
accurate description of the important change?

A reviewer can therefore dismiss the whole result as an **"NLI artifact"**: the paper
validates NLI against NLI-derived labels (circular). The existing spot-check
(`Results/runs/iter08_calibration/spotcheck/`) does *not* close this gap because
(a) the rater was an LLM (Claude Opus 4.7), not a human, and (b) it validates only the
RQ1 perturbation syn/antonym labels — **not** the diff→message diagnostic that carries
RQ2/RQ3.

This study supplies a single-rater human audit for the diff-description accuracy
construct and exposes why NLI and human judgments agree or diverge.

## 2. Constraint and honest positioning

Only **one independent human rater** is available. That is below the empirical-SE norm
(≥2 raters + inter-rater κ). We therefore position this as a **human–automatic agreement
check (triangulation anchor)**, *not* a definitive human gold standard, and we disclose
the single-rater limitation in Threats to Validity. The single-rater weakness is partially
offset by:

- **Machine-side triangulation already in the paper**: BART-MNLI + DeBERTa-v3 reproduce the
  same ordering (`iter14`).
- **Intra-rater test-retest reliability** (re-rate 15 items blind) substitutes weakly for
  inter-rater reliability and shows the rater's labels are internally stable.
- **Blind, independent rater** (not an author; never sees NLI scores, pass/fail, or whether
  the message is gold vs generated).

Claim language stays calibrated: "consistent with" / "triangulates", never "validates".

### Hard requirement
The rater **must not be an author** and **must be blind** to all machine outputs. A
non-blind or author-self rating is worse than no study (cf. the κ=1.0 LLM spot-check that
*invited* suspicion). If the only available human is an author, this study cannot run as
designed — escalate before proceeding.

## 3. Unit of annotation

A single **(shown diff, commit-message)** pair. The shown diff is capped to the
same 1,500-character premise used by the NLI diagnostic. The rater answers:
*using only this shown diff, does the message accurately describe the important
change?*

Both **gold** (human-written reference) and **generated** messages are annotated.
This lets the analysis compare human supported-rates and reason distributions for
gold vs generated messages under the same blind, NLI-matched diff input. The
result may confirm or challenge the automatic NLI ordering; either outcome is
reported as a calibration result, not forced into a validation claim.

## 4. Sampling frame and strata

Population: the 1,600 diffs × {gold, 3 generated} = long-form (diff, message) pairs built
from `Results/runs/iter13_scaled_triad/results.csv`, joined on `(language, diff_id)` to:
- diff text — `data/raw/<lang>.jsonl` (capped at 1,500 chars for readability, matching the
  paper's complete-raw-diff subset in RQ4);
- `has_intent` — `Results/runs/iter16_intent_classifier/classified.csv`.

Stratify on three binary axes → **8 cells**, **10 items/cell = 80 main**:

| axis | values | source |
|------|--------|--------|
| message source | gold / generated | results.csv |
| NLI decision | pass / fail at τ=+0.56 (signed_diff→msg ≥ τ) | results.csv |
| intent marker | intent / plain | iter16 `has_intent` |

Rationale: sampling on (source × NLI pass/fail) guarantees the human sees cases where NLI
says supported *and* where it says not-supported, on *both* gold and generated messages —
the cells where agreement actually matters. Intent-marker is crossed in because intent-tagged
commits are the paper's mechanistic explanation for the inversion (P2/P7).

**Lexical overlap** (reviewer's high/low-overlap axis) is recorded as a continuous covariate
(containment of message content-tokens in diff tokens) and reported via a **median split
post hoc**, rather than as a sampling cell, to keep the cell count at 8. Generated-message
**model** is a recorded covariate, not a sampling axis (generated pairs are drawn across all
three models).

If any cell is under-populated (e.g. gold × pass × intent is rare because diff→gold passes
only ~19%), the sampler takes `min(10, available)`, logs the shortfall, and the analysis
reports realized cell sizes. No sampling with replacement.

## 5. Rubric (3-way label + reason code)

Locked decision: rater assigns one primary label and one reason code.

- **supported** — the message accurately describes the important change shown in
  the diff without adding unsupported claims.
- **neutral** — the message is not clearly false, but the shown diff is not
  enough to call it accurate. Reasons include too abstract, missing key detail,
  near miss, external context, insufficient shown diff, or vague wording.
- **contradicted** — the message conflicts with the shown diff through opposite
  direction, wrong entity, wrong action, or a different change.

Primary analysis: `supported-rate = mean(label == supported)`, reported for
gold and generated messages. Secondary analysis reports label distribution and
reason distribution by source and by NLI cell. Human vs NLI agreement remains a
diagnostic, not the sole success criterion.

Full rater-facing instructions: `README_RATER.md` + `RUBRIC.md`.

## 6. Analysis outputs (`analyze_human_eval.py`)

1. **Primary supported-rate** for gold vs generated messages.
2. **Label distribution** (`supported`, `neutral`, `contradicted`) overall and by source.
3. **Reason distribution** overall and by source, especially neutral reasons.
4. **Human vs NLI agreement**: binary supported vs NLI pass + Cohen's κ.
5. **Per-cell agreement and supported-rate** for the 8 source/NLI/intent cells.
6. **Intra-rater reliability**: test-retest κ on the 15 re-rated items.

All written to `out/agreement.md` + machine-readable CSVs in `out/`.

## 7. Reporting rules (decided before rating)

- Supported-rate is the primary human outcome. Report gold and generated values
  as observed; do not force the result to match the NLI ordering.
- Human–NLI binary agreement / κ is reported as a calibration diagnostic. Low κ
  means NLI should be framed as an automatic proxy with failure modes.
- Reason distributions are used to explain whether differences come from
  abstraction, missing coverage, near-miss semantics, external context, or true
  contradiction.
- Intra-rater κ ≥ 0.6 is desirable; lower values must be disclosed as a
  single-rater stability limitation.

Threats-to-validity text is rewritten from "we deliberately omit human annotation" to
"a single independent blind rater triangulates the diff-grounding diagnostic; full
multi-rater study is future work," citing the realized numbers.

## 8. Reproducibility

```bash
cd IST_human_eval
python sample_human_eval.py        # → out/rater.csv (blind), out/key.csv (hidden), out/retest.csv (blind)
#  ... human fills rater_label and rater_reason in out/rater.csv and out/retest.csv ...
python analyze_human_eval.py       # → out/agreement.md + out/*.csv
```

Sampler is deterministic (`--seed 42`). `key.csv` is the hidden ground-truth map and must
**not** be shown to the rater before rating.
