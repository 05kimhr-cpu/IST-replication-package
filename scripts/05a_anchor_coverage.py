"""Iter 5a — how often do our anchor verbs appear in the corpus?

This quantifies the practical relevance of the meaning-direction blindness
finding. If a large fraction of commits use one of our anchor verbs, then
the class of errors we probe is plausibly common in real CMG output too.

Outputs: runs/iter05_anchor_coverage/summary.md
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from cmg_ist.io import load_samples, clean_msg  # noqa: E402
from cmg_ist.perturbation_pairs import TRIPLETS  # noqa: E402

LANGUAGES = ["cpp", "cs", "go", "java", "js", "php", "py", "rust"]
OUT_DIR = ROOT / "runs" / "iter05_anchor_coverage"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def tokens_lower(msg: str) -> set[str]:
    return {t.lower().strip(".,!?;:()[]") for t in msg.split()}


def main() -> None:
    lines: list[str] = ["# Iter 5a — anchor verb coverage in the corpus", ""]
    lines.append("We say a gold message is **covered** if any of our 52 triplet")
    lines.append("anchor forms (add/adds/..., remove/..., enable/..., etc.) appears")
    lines.append("as a whole word in the gold message (case-insensitive).")
    lines.append("")
    lines.append("| language | samples | covered | coverage % |")
    lines.append("|----------|---------|---------|------------|")

    total = 0
    total_covered = 0
    per_anchor_counts: dict[str, int] = {}

    for lang in LANGUAGES:
        samples = load_samples(lang, limit=None)
        covered = 0
        for s in samples:
            toks = tokens_lower(clean_msg(s["msg"]))
            hits = toks & TRIPLETS.keys()
            if hits:
                covered += 1
                for h in hits:
                    per_anchor_counts[h] = per_anchor_counts.get(h, 0) + 1
        total += len(samples)
        total_covered += covered
        lines.append(f"| {lang} | {len(samples)} | {covered} | {100*covered/len(samples):.1f}% |")

    lines.append(f"| **total** | {total} | {total_covered} | {100*total_covered/total:.1f}% |")
    lines.append("")
    lines.append("## Anchor frequency (pooled)")
    lines.append("")
    for anchor, cnt in sorted(per_anchor_counts.items(), key=lambda x: -x[1]):
        lines.append(f"- {anchor}: {cnt}")

    text = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
