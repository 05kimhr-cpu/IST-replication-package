"""Iteration 4 — add BERTScore to the paired perturbation matrix.

Reuses the paired set from iter 3 (8 languages × 52 anchor triplets).
Because BERTScore is expensive per call, we batch all candidates and
references once and score in a single forward pass.

Outputs:
  runs/iter04_bertscore/results.csv      — rows extended with bertscore_syn/ant/delta
  runs/iter04_bertscore/summary.md       — per-language and pooled summary
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
from cmg_ist.perturbation_pairs import TRIPLETS, apply_pair  # noqa: E402
from cmg_ist.metrics import bertscore_batch  # noqa: E402

LANGUAGES = ["cpp", "cs", "go", "java", "js", "php", "py", "rust"]
OUT_DIR = ROOT / "runs" / "iter04_bertscore"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_paired_rows() -> list[dict]:
    """Same protocol as iter 3 but we only keep the syn/ant texts; we score
    later in a single BERTScore batch to amortize model load."""
    rows: list[dict] = []
    for lang in LANGUAGES:
        samples = load_samples(lang, limit=None)
        for s in samples:
            gold = clean_msg(s["msg"])
            gold_tokens_lower = {
                t.lower().strip(".,!?;:()[]") for t in gold.split()
            }
            for anchor in TRIPLETS:
                if anchor not in gold_tokens_lower:
                    continue
                res = apply_pair(gold, anchor)
                if res is None:
                    continue
                syn_cand, ant_cand, info = res
                rows.append({
                    "diff_id": s["diff_id"],
                    "language": lang,
                    "anchor": anchor,
                    "synonym": info.synonym,
                    "antonym": info.antonym,
                    "gold": gold,
                    "syn_cand": syn_cand,
                    "ant_cand": ant_cand,
                    "gold_word_count": len(gold.split()),
                })
    return rows


def main() -> None:
    rows = collect_paired_rows()
    print(f"collected {len(rows)} paired rows")

    # Build flat lists: score 2N candidates against 2N references in one batch
    cands: list[str] = []
    refs: list[str] = []
    for r in rows:
        cands.append(r["syn_cand"])
        refs.append(r["gold"])
        cands.append(r["ant_cand"])
        refs.append(r["gold"])

    print(f"running BERTScore on {len(cands)} pairs (single batched pass)...")
    f1s = bertscore_batch(cands, refs)
    assert len(f1s) == 2 * len(rows)

    for i, r in enumerate(rows):
        r["bertscore_syn"] = f1s[2 * i]
        r["bertscore_ant"] = f1s[2 * i + 1]
        r["bertscore_delta"] = r["bertscore_syn"] - r["bertscore_ant"]

    # CSV (only essential cols; full cand text stays out of CSV)
    fieldnames = [
        "diff_id", "language", "anchor", "synonym", "antonym",
        "gold_word_count",
        "bertscore_syn", "bertscore_ant", "bertscore_delta",
    ]
    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    # Summaries
    def desc(rs: list[dict], label: str) -> list[str]:
        xs = [r["bertscore_delta"] for r in rs]
        if not xs:
            return [f"- {label}: (empty)"]
        mean = statistics.mean(xs)
        med = statistics.median(xs)
        stdev = statistics.stdev(xs) if len(xs) > 1 else 0.0
        absmean = statistics.mean(abs(x) for x in xs)
        near_zero = sum(1 for x in xs if abs(x) < 0.01)
        exact_zero = sum(1 for x in xs if x == 0.0)
        positive = sum(1 for x in xs if x > 0)
        negative = sum(1 for x in xs if x < 0)
        return [
            f"### {label}",
            f"- n={len(xs)}",
            f"- mean={mean:+.4f} median={med:+.4f} stdev={stdev:.4f}",
            f"- mean|x|={absmean:.4f}",
            f"- exact_zero={exact_zero}/{len(xs)}  near_zero(|x|<0.01)={near_zero}/{len(xs)}",
            f"- sign: syn>ant: {positive}/{len(xs)}   syn<ant: {negative}/{len(xs)}",
            "",
        ]

    lines: list[str] = [
        "# Iteration 4 — BERTScore paired perturbation",
        "",
        f"- model: roberta-large, device: cuda",
        f"- total pairs: {len(rows)}",
        "",
    ]
    lines.extend(desc(rows, "Overall (all languages)"))
    lines.append("## Per language")
    lines.append("")
    for lang in LANGUAGES:
        sub = [r for r in rows if r["language"] == lang]
        lines.extend(desc(sub, lang))

    summary_path = OUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
