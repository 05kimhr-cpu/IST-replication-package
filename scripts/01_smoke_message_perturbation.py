"""Iteration 1 smoke experiment.

Question probed: given a gold commit message, how much do BLEU-4 and ROUGE-L
scores move when we perturb the candidate at three levels of meaning-change?

  - trivial:       whitespace, trailing period
  - paraphrase:    synonym swap on action verbs
  - meaning_change: antonym swap on action verbs

Reference = original gold. Candidate = perturbed(gold).

Iter 1 scope:
  - 1 language (py)
  - 50 samples
  - 2 metrics (BLEU-4, ROUGE-L)
  - 4 perturbations (2 trivial, 1 paraphrase, 1 meaning-change)

Output: runs/iter01_smoke/results.csv and runs/iter01_smoke/summary.md
"""
from __future__ import annotations

import csv
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from cmg_ist.io import load_samples, clean_msg  # noqa: E402
from cmg_ist.perturbations import REGISTRY, apply  # noqa: E402
from cmg_ist.metrics import score_all  # noqa: E402

LANGUAGE = "py"
N_SAMPLES = 50
OUT_DIR = ROOT / "runs" / "iter01_smoke"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    samples = load_samples(LANGUAGE, limit=N_SAMPLES)

    rows: list[dict] = []
    for s in samples:
        gold = clean_msg(s["msg"])
        for pert in REGISTRY:
            cand, applied = apply(gold, pert.name)
            for sc in score_all(cand, gold):
                rows.append({
                    "diff_id": s["diff_id"],
                    "language": LANGUAGE,
                    "perturbation": pert.name,
                    "kind": pert.kind,
                    "applied": int(applied),
                    "metric": sc.metric,
                    "score": sc.score,
                })

    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # summary
    by_key: dict[tuple[str, str, str], list[float]] = {}
    applicability: dict[str, tuple[int, int]] = {}  # pert -> (applied, total)
    for r in rows:
        by_key.setdefault((r["perturbation"], r["kind"], r["metric"]), []).append(r["score"])
    for pert in REGISTRY:
        applied_count = sum(1 for r in rows if r["perturbation"] == pert.name and r["applied"])
        total_count = sum(1 for r in rows if r["perturbation"] == pert.name) // 2  # 2 metrics
        applicability[pert.name] = (applied_count // 2, total_count)

    lines: list[str] = []
    lines.append(f"# Iteration 1 smoke results")
    lines.append(f"- language: {LANGUAGE}")
    lines.append(f"- samples: {N_SAMPLES}")
    lines.append("")
    lines.append("## Applicability (how often the perturbation produced a change)")
    for name, (a, t) in applicability.items():
        lines.append(f"- {name}: {a}/{t}")
    lines.append("")
    lines.append("## Mean score vs identity (1.0) by (kind, perturbation, metric)")
    lines.append("")
    lines.append("| kind | perturbation | metric | mean | median | min | n |")
    lines.append("|------|--------------|--------|------|--------|-----|---|")
    for (pert, kind, metric), scores in sorted(by_key.items(), key=lambda x: (x[0][1], x[0][0], x[0][2])):
        applicable_scores = [
            r["score"] for r in rows
            if r["perturbation"] == pert and r["metric"] == metric and r["applied"]
        ]
        if not applicable_scores:
            lines.append(f"| {kind} | {pert} | {metric} | n/a | n/a | n/a | 0 |")
            continue
        m = statistics.mean(applicable_scores)
        md = statistics.median(applicable_scores)
        mn = min(applicable_scores)
        n = len(applicable_scores)
        lines.append(f"| {kind} | {pert} | {metric} | {m:.4f} | {md:.4f} | {mn:.4f} | {n} |")

    summary_path = OUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")

    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
