#!/usr/bin/env python
"""RQ1 CSV audit for metric-blindness numbers.

CPU-only. Recomputes the per-family delta statistics used in Paper.md from the
released result CSVs so the manuscript can be checked against source artefacts.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def summarize(values: list[float]) -> dict[str, float]:
    n = len(values)
    return {
        "n": float(n),
        "mean_abs_delta": statistics.mean(abs(v) for v in values),
        "exact_zero_pct": 100.0 * sum(v == 0.0 for v in values) / n,
        "abs_lt_0.01_pct": 100.0 * sum(abs(v) < 0.01 for v in values) / n,
        "syn_pref_pct": 100.0 * sum(v > 0.0 for v in values) / n,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("Results/runs"))
    parser.add_argument("--output", type=Path, default=Path("Results/runs/iter18_rq1_csv_audit"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    verb = read_csv(args.runs / "iter03_paired_all" / "results.csv")
    verb_bert = read_csv(args.runs / "iter04_bertscore" / "results.csv")
    verb_nli = read_csv(args.runs / "iter05b_nli" / "results.csv")
    noun = read_csv(args.runs / "iter07_noun" / "results.csv")

    records: list[dict[str, str]] = []

    def add(metric: str, family: str, values: list[float]) -> None:
        stats = summarize(values)
        records.append(
            {
                "metric": metric,
                "family": family,
                "n": str(int(stats["n"])),
                "mean_abs_delta": f"{stats['mean_abs_delta']:.6f}",
                "exact_zero_pct": f"{stats['exact_zero_pct']:.1f}",
                "abs_lt_001_pct": f"{stats['abs_lt_0.01_pct']:.1f}",
                "syn_pref_pct": f"{stats['syn_pref_pct']:.1f}",
            }
        )

    for metric in ["bleu", "rougeL", "chrf", "meteor"]:
        add(metric, "verb", [float(r[f"{metric}_delta"]) for r in verb])
        add(metric, "noun", [float(r[f"{metric}_delta"]) for r in noun])
    add("bertscore", "verb", [float(r["bertscore_delta"]) for r in verb_bert])
    add("bertscore", "noun", [float(r["bertscore_delta"]) for r in noun])
    add("nli_signed", "verb", [float(r["signed_delta"]) for r in verb_nli])
    add("nli_signed", "noun", [float(r["signed_delta"]) for r in noun])

    csv_path = args.output / "rq1_delta_stats.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "metric",
                "family",
                "n",
                "mean_abs_delta",
                "exact_zero_pct",
                "abs_lt_001_pct",
                "syn_pref_pct",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    lines = [
        "# Iter 18 - RQ1 CSV audit",
        "",
        "These values are recomputed directly from the released RQ1 result CSVs.",
        "",
        "| metric | family | n | mean abs delta | exact zero % | abs(delta) < 0.01 % | syn-pref % |",
        "|--------|--------|---:|-------------:|-------------:|-----------------:|-----------:|",
    ]
    for r in records:
        lines.append(
            "| {metric} | {family} | {n} | {mean_abs_delta} | {exact_zero_pct} | "
            "{abs_lt_001_pct} | {syn_pref_pct} |".format(**r)
        )
    (args.output / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {args.output / 'summary.md'}")


if __name__ == "__main__":
    main()
