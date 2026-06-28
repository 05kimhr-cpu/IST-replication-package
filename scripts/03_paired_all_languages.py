"""Iteration 3 — paired perturbation across all 8 languages and 4 metrics.

Question: does the "overlap metrics give identical scores to syn/ant
substitutions" result from iter 2 generalize beyond BLEU+ROUGE+Python?

Metrics tested:
  - BLEU-4 (sacrebleu)   — word n-gram precision + BP
  - ROUGE-L              — longest common subsequence
  - CHRF++               — character n-gram overlap
  - METEOR               — word overlap with WordNet synonym matching

Expected:
  - BLEU, ROUGE-L: identical syn vs ant (mathematical, confirmed iter 2)
  - CHRF: depends on character overlap between syn and ant substitutes.
    Distinct scores only if syn/ant have systematically different char
    overlap with the anchor — they don't, by construction, so likely
    near-identical too.
  - METEOR: can discriminate IF anchor-syn is in WordNet's synonym set
    AND anchor-ant is not. For our CMG verb pairs, WordNet coverage is
    the open question.

Runs across 8 languages × all samples. Outputs per-language summary plus
a combined view.
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
from cmg_ist.metrics import bleu_sentence, rougeL, chrf, meteor  # noqa: E402

LANGUAGES = ["cpp", "cs", "go", "java", "js", "php", "py", "rust"]
METRICS = [
    ("bleu", bleu_sentence),
    ("rougeL", rougeL),
    ("chrf", chrf),
    ("meteor", meteor),
]
OUT_DIR = ROOT / "runs" / "iter03_paired_all"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_one_language(lang: str) -> list[dict]:
    samples = load_samples(lang, limit=None)
    rows: list[dict] = []
    for s in samples:
        gold = clean_msg(s["msg"])
        gold_tokens_lower = set(t.lower().strip(".,!?;:()[]") for t in gold.split())
        for anchor in TRIPLETS:
            if anchor not in gold_tokens_lower:
                continue
            res = apply_pair(gold, anchor)
            if res is None:
                continue
            syn_cand, ant_cand, info = res
            row = {
                "diff_id": s["diff_id"],
                "language": lang,
                "anchor": anchor,
                "synonym": info.synonym,
                "antonym": info.antonym,
                "gold_word_count": len(gold.split()),
            }
            for metric_name, fn in METRICS:
                row[f"{metric_name}_syn"] = fn(syn_cand, gold)
                row[f"{metric_name}_ant"] = fn(ant_cand, gold)
                row[f"{metric_name}_delta"] = row[f"{metric_name}_syn"] - row[f"{metric_name}_ant"]
            rows.append(row)
    return rows


def summarize(rows: list[dict], label: str) -> list[str]:
    lines = [f"### {label}", f"- rows: {len(rows)}"]
    for metric_name, _ in METRICS:
        deltas = [r[f"{metric_name}_delta"] for r in rows]
        if not deltas:
            lines.append(f"- {metric_name}: (no rows)")
            continue
        mean = statistics.mean(deltas)
        med = statistics.median(deltas)
        stdev = statistics.stdev(deltas) if len(deltas) > 1 else 0.0
        absmean = statistics.mean(abs(x) for x in deltas)
        near_zero = sum(1 for x in deltas if abs(x) < 0.01)
        exact_zero = sum(1 for x in deltas if x == 0.0)
        nonzero_abs = [abs(x) for x in deltas if x != 0]
        nonzero_mean = statistics.mean(nonzero_abs) if nonzero_abs else 0.0
        lines.append(
            f"- **{metric_name}**: mean={mean:+.4f} median={med:+.4f} stdev={stdev:.4f} "
            f"mean|x|={absmean:.4f} exact_zero={exact_zero}/{len(deltas)} "
            f"near_zero(|x|<0.01)={near_zero}/{len(deltas)} "
            f"mean|x|_if_nonzero={nonzero_mean:.4f}"
        )
    return lines


def main() -> None:
    all_rows: list[dict] = []
    per_lang_lines: list[str] = ["## Per-language breakdown", ""]

    for lang in LANGUAGES:
        rows = run_one_language(lang)
        all_rows.extend(rows)
        per_lang_lines.extend(summarize(rows, lang))
        per_lang_lines.append("")

    # Write combined CSV
    csv_path = OUT_DIR / "results.csv"
    if all_rows:
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)

    # Summary
    lines: list[str] = ["# Iteration 3 — paired perturbation, 8 langs × 4 metrics", ""]
    lines.append(f"Total paired rows: {len(all_rows)}")
    lines.append("")
    lines.extend(summarize(all_rows, "Overall (all languages pooled)"))
    lines.append("")
    lines.extend(per_lang_lines)

    summary_path = OUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
