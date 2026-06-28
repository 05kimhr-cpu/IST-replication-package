# Rubric - diff-description accuracy audit

You are given a **code diff** and one **commit message**. Judge only the
diff text shown in the spreadsheet. The shown diff is capped to the same
1,500-character diff premise used by the NLI diagnostic.

The question is:

> Using only this diff, does the message accurately describe the important
> change?

Write one value in `rater_label` and one value in `rater_reason`.

## Primary Labels

- **supported** - The message accurately describes the change shown in the
  diff. It does not need to mention every line, but it must capture the core
  change without adding unsupported claims.

- **neutral** - The message is not clearly false, but the shown diff is not
  enough to call it an accurate description. This includes vague messages,
  missing key details, near-miss wording, external context, or cases where the
  capped diff does not show enough evidence.

- **contradicted** - The message conflicts with the shown diff. It describes
  the opposite direction, the wrong entity, the wrong action, or a different
  change.

## Reason Codes

Use exactly one reason code that matches the primary label.

### supported

- **exact_core** - The message captures the central change accurately.
- **partial_but_sufficient** - The message omits minor details, but it still
  describes the important change well enough.
- **high_level_but_correct** - The message is broad, but the broad description
  is accurate for the shown diff.

### neutral

- **too_abstract** - The message is too generic to verify well, such as
  "improve", "update", "cleanup", or "refactor" without a clear target.
- **missing_key_detail** - The message describes a real part of the diff but
  leaves out another important part of the change.
- **near_miss** - The message is close in meaning but not quite the same as the
  shown change.
- **external_context** - The message relies on a bug cause, issue, intent,
  runtime behavior, or rationale that the diff does not show.
- **diff_insufficient** - The shown 1,500-character diff is not enough to judge;
  the missing evidence may be outside the shown diff.
- **ambiguous_vague** - The message is unclear enough that a precise judgment is
  not possible.
- **other_neutral** - Neutral for a reason not covered above; explain briefly in
  `rater_notes`.

### contradicted

- **opposite_direction** - The diff adds/enables/increases something while the
  message says remove/disable/decrease, or the reverse.
- **wrong_entity** - The message names the wrong file, function, variable,
  module, API, or feature.
- **wrong_action** - The message names the right area but the wrong operation.
- **different_change** - The message describes a change that is simply not the
  shown change.
- **other_contradicted** - Contradicted for a reason not covered above; explain
  briefly in `rater_notes`.

## Decision Steps

1. Read the diff and identify the important changed behavior, API, file, or
   code structure.
2. Read the message.
3. Ask whether the message accurately describes the important change using only
   the shown diff.
4. Choose `supported`, `neutral`, or `contradicted`.
5. Choose the most specific matching `rater_reason`.

## Examples

| shown diff gist | message | label | reason |
|---|---|---|---|
| adds a null check in `parse()` | "Fix NPE in parse()" | supported | exact_core |
| adds null check and changes error handling | "Add null check" | neutral | missing_key_detail |
| renames helper and moves code | "Refactor implementation" | neutral | too_abstract |
| changes request decoding but message says parsing | "Fix parser behavior" | neutral | near_miss |
| adds null check but message names a ticket-only cause | "Fix OAuth crash from malformed token" | neutral | external_context |
| shown diff starts before the relevant added code | "Add validation for user IDs" | neutral | diff_insufficient |
| changes `enabled = false` to `true` | "Disable feature flag" | contradicted | opposite_direction |

## Rules

- Do not judge grammar, style, or whether the message is pleasant to read.
- Do not reward surface word overlap by itself.
- Do not use the full repository or external knowledge.
- Do not infer gold/generated source. The source is intentionally hidden.
