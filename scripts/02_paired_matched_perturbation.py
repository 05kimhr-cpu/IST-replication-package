"""Iteration 2 — paired matched perturbation analysis.

Paired design: for each sample and each anchor word present in the gold
message, produce BOTH a synonym-substituted and an antonym-substituted
candidate. Score both against the gold. Compare per sample.

Mathematical prediction: for word-level overlap metrics (BLEU, ROUGE-L),
substituting word W at fixed positions in the candidate by synonym S or
antonym A — given len_words(S) == len_words(A) == 1 — breaks the same set
of n-grams against the reference. Scores should be numerically identical
up to tokenization edge cases.

A non-trivial gap between BLEU(syn) and BLEU(ant), *per sample*, would be
a surprise. A tight distribution around zero delta would confirm that
overlap metrics are provably blind to meaning direction in 1-token edits.

Scope: Python, 500 samples (all), triplet vocabulary of 48 anchors.
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
from cmg_ist.metrics import bleu_sentence, rougeL  # noqa: E402

LANGUAGE = "py"
OUT_DIR = ROOT / "runs" / "iter02_paired"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def main() -> None:
    samples = load_samples(LANGUAGE, limit=None)

    rows: list[dict] = []
    anchors_seen: dict[str, int] = {}  # anchor -> num samples hit

    for s in samples:
        gold = clean_msg(s["msg"])
        gold_lower_tokens = set(t.lower().strip(".,!?;:()[]") for t in gold.split())
        for anchor in TRIPLETS:
            if anchor not in gold_lower_tokens:
                continue
            res = apply_pair(gold, anchor)
            if res is None:
                continue
            syn_cand, ant_cand, info = res
            anchors_seen[anchor] = anchors_seen.get(anchor, 0) + 1

            bleu_syn = bleu_sentence(syn_cand, gold)
            bleu_ant = bleu_sentence(ant_cand, gold)
            rouge_syn = rougeL(syn_cand, gold)
            rouge_ant = rougeL(ant_cand, gold)

            rows.append({
                "diff_id": s["diff_id"],
                "language": LANGUAGE,
                "anchor": anchor,
                "synonym": info.synonym,
                "antonym": info.antonym,
                "anchor_hits": info.anchor_hit_count,
                "gold_word_count": len(gold.split()),
                "bleu_syn": bleu_syn,
                "bleu_ant": bleu_ant,
                "rougeL_syn": rouge_syn,
                "rougeL_ant": rouge_ant,
                "bleu_delta": bleu_syn - bleu_ant,
                "rougeL_delta": rouge_syn - rouge_ant,
            })

    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Analyze deltas
    bleu_deltas = [r["bleu_delta"] for r in rows]
    rouge_deltas = [r["rougeL_delta"] for r in rows]

    def describe(name: str, xs: list[float]) -> str:
        if not xs:
            return f"{name}: (empty)"
        n = len(xs)
        mean = statistics.mean(xs)
        median = statistics.median(xs)
        stdev = statistics.stdev(xs) if n > 1 else 0.0
        absmean = statistics.mean(abs(x) for x in xs)
        n_exact_zero = sum(1 for x in xs if x == 0.0)
        return (
            f"{name}: n={n} mean={mean:+.4f} median={median:+.4f} "
            f"stdev={stdev:.4f} mean|x|={absmean:.4f} zero={n_exact_zero}/{n}"
        )

    lines: list[str] = []
    lines.append("# Iteration 2 paired matched perturbation")
    lines.append(f"- language: {LANGUAGE}")
    lines.append(f"- samples total: {len(samples)}")
    lines.append(f"- triplets tried: {len(TRIPLETS)}")
    lines.append(f"- paired rows produced (anchor hit): {len(rows)}")
    lines.append(f"- unique anchors hit: {len(anchors_seen)}")
    lines.append("")
    lines.append("## Per-sample delta (syn_score - ant_score)")
    lines.append("")
    lines.append(describe("BLEU  delta", bleu_deltas))
    lines.append(describe("ROUGE delta", rouge_deltas))
    lines.append("")
    lines.append("## Anchor frequency")
    for anchor, cnt in sorted(anchors_seen.items(), key=lambda x: -x[1]):
        lines.append(f"- {anchor}: {cnt}")
    lines.append("")

    # Effect-size style summary: what fraction of rows have |delta| < 0.01?
    near_zero_bleu = sum(1 for x in bleu_deltas if abs(x) < 0.01)
    near_zero_rouge = sum(1 for x in rouge_deltas if abs(x) < 0.01)
    lines.append("## Fraction of near-identical rows (|delta| < 0.01)")
    lines.append(f"- BLEU : {near_zero_bleu}/{len(bleu_deltas)}")
    lines.append(f"- ROUGE: {near_zero_rouge}/{len(rouge_deltas)}")

    summary_path = OUT_DIR / "summary.md"
    summary_path.write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {summary_path}")
    print()
    print("\n".join(lines))


if __name__ == "__main__":
    main()
