# Rater instructions - diff-description accuracy audit

Please read `RUBRIC.md` once before starting.

## Who should rate

- A programmer who reads English fluently and can read code diffs.
- Not an author of the paper.

## What you judge

Each row contains one shown code diff and one commit message.

Question:

> Using only the shown diff, does the message accurately describe the important
> change?

The shown diff is capped to the same 1,500-character diff premise used by the
automatic NLI diagnostic. Do not use external repository knowledge.

## What you fill in

Open `out/rater.csv` in a spreadsheet editor. For each row, fill:

- `rater_label`
- `rater_reason`
- optionally `rater_notes`

Use lowercase values exactly as written.

### `rater_label`

- `supported`
- `neutral`
- `contradicted`

### `rater_reason`

Allowed reasons by label:

```text
supported:
  exact_core
  partial_but_sufficient
  high_level_but_correct

neutral:
  too_abstract
  missing_key_detail
  near_miss
  external_context
  diff_insufficient
  ambiguous_vague
  other_neutral

contradicted:
  opposite_direction
  wrong_entity
  wrong_action
  different_change
  other_contradicted
```

## Re-rating pass

After finishing `out/rater.csv`, take a short break, then rate
`out/retest.csv` the same way. It contains a blind subset in a different order.
Do not look back at your earlier answers.

## Files not to open

Do not open `out/key.csv` before rating. It contains hidden source and NLI
metadata and would invalidate the blind rating.

## About the diff format

The diffs come from MCMD and are tokenized:

- punctuation has spaces around it, e.g. `sw . cpp`, `+ =`, `: :`
- `mmm a / file` means `--- a/file`
- `ppp b / file` means `+++ b/file`
- lines starting with `-` are removed
- lines starting with `+` are added

The odd spacing is a dataset artifact, not part of your judgment.

## Notes

- Judge description accuracy, not writing style.
- A message can be supported even if it omits minor details.
- A message should be neutral if it is plausible but too abstract, incomplete,
  near-miss, externally contextual, or not verifiable from the shown diff.
- A message should be contradicted only when it conflicts with the shown diff.
- The main file has 80 items, followed by 15 re-rating items.
