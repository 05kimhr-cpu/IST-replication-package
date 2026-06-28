"""Iter 16 — Author-intent marker classification (CPU only).

Hypothesis for paper §7 (construct mismatch):
Commits whose *gold* message carries non-diff intent markers
(ticket IDs, revert/hotfix/release tags, co-author/signed-off trailers)
should have a LOWER diff→gold NLI pass-rate than plain commits —
because the intent marker encodes something that is not literally in
the diff. If the hypothesis holds, this is the smoking-gun evidence
for §7's construct-mismatch claim: the gold is "diff-weak" exactly
where it is doing intent-carrying work.

Inputs:
  - runs/iter13_scaled_triad/results.csv  (gold + NLI cols)

Outputs:
  - runs/iter16_intent_classifier/classified.csv   (per-row tags)
  - runs/iter16_intent_classifier/summary.md       (per-class stats)
"""
from __future__ import annotations

import csv
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ITER13_CSV = ROOT / "runs" / "iter13_scaled_triad" / "results.csv"
OUT_DIR = ROOT / "runs" / "iter16_intent_classifier"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TAU = 0.56

PATTERNS: dict[str, re.Pattern] = {
    # MCMD is tokenized with spaces around punctuation, so allow \s* inside compound tags.
    # JIRA-style (PROJ-123), GitHub refs (#123, GH-123), gitlab !123, CryEngine "CE - 10899"
    "ticket":   re.compile(r"(?<![A-Za-z0-9])(?:[A-Z][A-Z0-9]{1,9}\s*-\s*\d+|#\s*\d+|GH\s*-\s*\d+|!\s*\d+)(?![A-Za-z0-9])"),
    # Revert commit
    "revert":   re.compile(r"\brevert(?:s|ed|ing)?\b", re.IGNORECASE),
    # Hotfix / bugfix / CVE / security
    "hotfix":   re.compile(r"\b(?:hotfix|bugfix|patch|cve\s*-\s*\d+|security\s+(?:fix|patch))\b", re.IGNORECASE),
    # Release / version bump
    "release":  re.compile(r"\b(?:release|bump(?:ed)?|version\s+v?\d|v\d+\s*\.\s*\d+\s*\.\s*\d+|changelog)\b", re.IGNORECASE),
    # Merge commit (natural or PR)
    "merge":    re.compile(r"\bmerge\s+(?:pull\s+request|branch|remote|and\s+generalize|from)\b", re.IGNORECASE),
    # Co-authored / Signed-off trailers (author coordination) — still include though MCMD strips
    "trailer":  re.compile(r"(?:Co-authored-by|Signed-off-by|Reviewed-by|Acked-by)\s*:", re.IGNORECASE),
    # Bracketed area/module tag (e.g., "[ renderer ]", "[ Type checker ]", "[ NFC ]")
    "bracket":  re.compile(r"\[\s*[A-Za-z][A-Za-z0-9_\-/ ]*\s*\]"),
    # CryEngine-style "! B ( Audio )", "! XB ( Renderer )", "! CT ( PhysX )"
    "ce_flag":  re.compile(r"^\s*!\s*[A-Z]{1,3}\s*\("),
    # "NFC" = No Functional Change (common in LLVM/CE commits)
    "nfc":      re.compile(r"\bNFC\b|\[\s*NFC\s*\]"),
    # Branch / workflow prefix (e.g., "master - next :", "main :")
    "branch":   re.compile(r"^(?:master|main|dev(?:elop)?|feature|release|next)(?:\s*-\s*\w+)*\s*:\s"),
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    halfw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, centre - halfw), min(1.0, centre + halfw)


def classify(msg: str) -> list[str]:
    tags = [name for name, pat in PATTERNS.items() if pat.search(msg)]
    return tags


def two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Return (z, two-sided p) for pooled-proportion z-test between group1 vs group2."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1, p2 = k1 / n1, k2 / n2
    p = (k1 + k2) / (n1 + n2)
    se = (p * (1 - p) * (1 / n1 + 1 / n2)) ** 0.5
    if se == 0:
        return 0.0, 1.0
    z = (p1 - p2) / se
    # two-sided normal p via erf
    from math import erf, sqrt
    p_two = 2 * (1 - 0.5 * (1 + erf(abs(z) / sqrt(2))))
    return z, p_two


def main() -> None:
    rows: list[dict] = []
    with ITER13_CSV.open() as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    print(f"loaded {len(rows)} iter13 rows", flush=True)

    for r in rows:
        r["intent_tags"] = ",".join(classify(r["gold"])) or "plain"
        r["has_intent"] = "1" if r["intent_tags"] != "plain" else "0"

    # per-tag stats (rows can carry >1 tag; we count a row once per tag present)
    print("\n## Per-tag occurrence and diff→gold pass-rate\n", flush=True)
    lines_tag = [
        "| tag      |    n   | diff→gold pass ≥ +0.56 (95% CI)     | mean signed  |",
        "|----------|--------|--------------------------------------|--------------|",
    ]
    tag_counts: Counter = Counter()
    for r in rows:
        for t in r["intent_tags"].split(",") if r["intent_tags"] != "plain" else []:
            tag_counts[t] += 1
    for tag in PATTERNS:
        matching = [r for r in rows if tag in r["intent_tags"].split(",")]
        vs = [float(r["nli_signed_diff→gold"]) for r in matching]
        above = sum(1 for v in vs if v >= TAU)
        n = len(matching)
        if n == 0:
            lines_tag.append(f"| {tag:<8} |   0   | 0/0 (—)                              | —            |")
            continue
        lo, hi = wilson_ci(above, n)
        mu = statistics.mean(vs)
        lines_tag.append(f"| {tag:<8} | {n:>6} | {above}/{n} ({100*above/n:.1f}%) [{100*lo:.1f}–{100*hi:.1f}] | {mu:+.3f}       |")

    # has_intent vs plain
    intent_rows = [r for r in rows if r["has_intent"] == "1"]
    plain_rows = [r for r in rows if r["has_intent"] == "0"]
    vs_i = [float(r["nli_signed_diff→gold"]) for r in intent_rows]
    vs_p = [float(r["nli_signed_diff→gold"]) for r in plain_rows]
    k_i = sum(1 for v in vs_i if v >= TAU)
    k_p = sum(1 for v in vs_p if v >= TAU)
    lo_i, hi_i = wilson_ci(k_i, len(vs_i))
    lo_p, hi_p = wilson_ci(k_p, len(vs_p))
    z, p_two = two_proportion_z(k_i, len(vs_i), k_p, len(vs_p))

    # per-model gold→gen pass on intent vs plain (for §5 discussion)
    model_slugs = ["codellama-7b", "qwen2.5-coder-7b", "deepseek-6.7b"]
    per_model = []
    for ms in model_slugs:
        col = f"nli_signed_gold→gen_{ms}"
        vi = [float(r[col]) for r in intent_rows]
        vp = [float(r[col]) for r in plain_rows]
        ki = sum(1 for v in vi if v >= TAU)
        kp = sum(1 for v in vp if v >= TAU)
        _, pv = two_proportion_z(ki, len(vi), kp, len(vp))
        per_model.append((ms, ki, len(vi), kp, len(vp), pv))

    # Write classified.csv
    with (OUT_DIR / "classified.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["diff_id", "language", "has_intent", "intent_tags",
                                           "nli_signed_diff→gold",
                                           "nli_signed_gold→gen_codellama-7b",
                                           "nli_signed_gold→gen_qwen2.5-coder-7b",
                                           "nli_signed_gold→gen_deepseek-6.7b"],
                           extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Build summary
    lines = [
        "# Iter 16 — Author-intent marker classification",
        "",
        "Inputs: iter13 N=1600 rows, gold message text.",
        "Classes are NOT mutually exclusive (a commit can be both 'ticket' and 'hotfix').",
        "`has_intent = 1` iff ≥1 tag matches.",
        "",
        f"Total rows: {len(rows)}",
        f"Rows with ≥1 intent tag: {len(intent_rows)} ({100*len(intent_rows)/len(rows):.1f}%)",
        f"Plain rows: {len(plain_rows)} ({100*len(plain_rows)/len(rows):.1f}%)",
        "",
        "## Tag frequency (overall)",
        "",
        "| tag      | count | share of N=1600 |",
        "|----------|-------|-----------------|",
    ]
    for tag in PATTERNS:
        c = tag_counts[tag]
        lines.append(f"| {tag:<8} | {c:>5} | {100*c/len(rows):.1f}%          |")

    lines += ["", "## Tagged vs plain: diff→gold pass-rate (τ = +0.56)", "",
              "| group       |    n   | pass (95% CI)                        | mean signed  |",
              "|-------------|--------|--------------------------------------|--------------|",
              f"| has_intent  | {len(intent_rows):>6} | {k_i}/{len(intent_rows)} ({100*k_i/len(intent_rows):.1f}%) [{100*lo_i:.1f}–{100*hi_i:.1f}] | {statistics.mean(vs_i):+.3f}       |",
              f"| plain       | {len(plain_rows):>6} | {k_p}/{len(plain_rows)} ({100*k_p/len(plain_rows):.1f}%) [{100*lo_p:.1f}–{100*hi_p:.1f}] | {statistics.mean(vs_p):+.3f}       |",
              "",
              f"Two-proportion z = {z:+.3f}, p = {p_two:.3g} (two-sided)",
              "",
              "## Per-tag diff→gold pass-rate",
              ""]
    lines += lines_tag

    lines += ["", "## Does intent-tagging affect gold→gen pass-rate? (per model)", "",
              "| model            | intent pass-rate         | plain pass-rate          | p (z-test) |",
              "|------------------|--------------------------|--------------------------|------------|"]
    for ms, ki, ni, kp, np_, pv in per_model:
        lines.append(f"| {ms:<16} | {ki}/{ni} ({100*ki/ni:.1f}%)          | {kp}/{np_} ({100*kp/np_:.1f}%)          | {pv:.3g}   |")

    lines += ["", "## Interpretation",
              "",
              "If `has_intent` group has materially *lower* diff→gold pass-rate than",
              "`plain` (and z-test is significant), that supports §7: the gold is diff-weak",
              "precisely because it encodes intent markers that are not in the diff.",
              ""]

    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.md").write_text(summary)
    print(summary)
    print("DONE")


if __name__ == "__main__":
    main()
