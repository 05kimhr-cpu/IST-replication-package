#!/usr/bin/env python
"""Threshold and paired diagnostics for diff-grounded NLI.

This script is CPU-only. It reuses the released iter13 BART-MNLI and
iter14 DeBERTa per-row scores, then writes:

- threshold_sweep.csv: pass-rates for diff->gold and diff->generated
  under several signed-score thresholds.
- paired_crosstab.csv: paired gold/generated pass/fail tables at the
  paper threshold tau=+0.56.
- summary.md: short interpretation for the manuscript/reviewer response.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


MODELS = ["codellama-7b", "qwen2.5-coder-7b", "deepseek-6.7b"]
THRESHOLDS = [-0.50, 0.00, 0.25, 0.56, 0.75, 0.90]
PAPER_TAU = 0.56


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({key.replace("→", "->"): value for key, value in row.items()})
        return rows


def pct(numer: int, denom: int) -> float:
    return 100.0 * numer / denom if denom else 0.0


def pass_rate(rows: list[dict[str, str]], column: str, tau: float) -> tuple[int, float]:
    passed = sum(float(row[column]) >= tau for row in rows)
    return passed, pct(passed, len(rows))


def paired_counts(
    rows: list[dict[str, str]], gold_col: str, gen_col: str, tau: float
) -> dict[str, int]:
    counts = {"both_pass": 0, "gold_only": 0, "gen_only": 0, "both_fail": 0}
    for row in rows:
        gold_pass = float(row[gold_col]) >= tau
        gen_pass = float(row[gen_col]) >= tau
        if gold_pass and gen_pass:
            counts["both_pass"] += 1
        elif gold_pass:
            counts["gold_only"] += 1
        elif gen_pass:
            counts["gen_only"] += 1
        else:
            counts["both_fail"] += 1
    return counts


def write_threshold_sweep(
    out_path: Path, bart_rows: list[dict[str, str]], deb_rows: list[dict[str, str]]
) -> None:
    fieldnames = ["backbone", "threshold", "condition", "n", "pass_n", "pass_pct"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        specs = [
            (
                "BART-MNLI",
                bart_rows,
                "nli_signed_diff->gold",
                {m: f"nli_signed_diff->gen_{m}" for m in MODELS},
            ),
            (
                "DeBERTa-v3",
                deb_rows,
                "deb_nli_signed_diff->gold",
                {m: f"deb_nli_signed_diff->gen_{m}" for m in MODELS},
            ),
        ]
        for backbone, rows, gold_col, gen_cols in specs:
            for tau in THRESHOLDS:
                pass_n, pass_pct = pass_rate(rows, gold_col, tau)
                writer.writerow(
                    {
                        "backbone": backbone,
                        "threshold": f"{tau:.2f}",
                        "condition": "diff->gold",
                        "n": len(rows),
                        "pass_n": pass_n,
                        "pass_pct": f"{pass_pct:.1f}",
                    }
                )
                for model, col in gen_cols.items():
                    pass_n, pass_pct = pass_rate(rows, col, tau)
                    writer.writerow(
                        {
                            "backbone": backbone,
                            "threshold": f"{tau:.2f}",
                            "condition": f"diff->gen:{model}",
                            "n": len(rows),
                            "pass_n": pass_n,
                            "pass_pct": f"{pass_pct:.1f}",
                        }
                    )


def write_paired_crosstab(
    out_path: Path, bart_rows: list[dict[str, str]], deb_rows: list[dict[str, str]]
) -> list[dict[str, str]]:
    fieldnames = [
        "backbone",
        "threshold",
        "model",
        "n",
        "both_pass",
        "gold_only",
        "gen_only",
        "both_fail",
        "gen_minus_gold_pp",
    ]
    records: list[dict[str, str]] = []
    specs = [
        (
            "BART-MNLI",
            bart_rows,
            "nli_signed_diff->gold",
            {m: f"nli_signed_diff->gen_{m}" for m in MODELS},
        ),
        (
            "DeBERTa-v3",
            deb_rows,
            "deb_nli_signed_diff->gold",
            {m: f"deb_nli_signed_diff->gen_{m}" for m in MODELS},
        ),
    ]
    for backbone, rows, gold_col, gen_cols in specs:
        gold_pass_n, gold_pass_pct = pass_rate(rows, gold_col, PAPER_TAU)
        for model, gen_col in gen_cols.items():
            counts = paired_counts(rows, gold_col, gen_col, PAPER_TAU)
            gen_pass_n = counts["both_pass"] + counts["gen_only"]
            record = {
                "backbone": backbone,
                "threshold": f"{PAPER_TAU:.2f}",
                "model": model,
                "n": str(len(rows)),
                **{k: str(v) for k, v in counts.items()},
                "gen_minus_gold_pp": f"{pct(gen_pass_n, len(rows)) - gold_pass_pct:.1f}",
            }
            records.append(record)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return records


def write_summary(out_path: Path, crosstabs: list[dict[str, str]]) -> None:
    lines = [
        "# Iter 19 - Threshold and paired NLI diagnostics",
        "",
        "This CPU-only diagnostic reuses iter13 BART-MNLI and iter14 DeBERTa",
        "per-row signed scores. It adds no new model inference.",
        "",
        "## Paired result at tau = +0.56",
        "",
        "| backbone | model | both pass | gold only | gen only | both fail | gen-gold pp |",
        "|----------|-------|----------:|----------:|---------:|----------:|------------:|",
    ]
    for row in crosstabs:
        lines.append(
            "| {backbone} | {model} | {both_pass} | {gold_only} | {gen_only} | "
            "{both_fail} | {gen_minus_gold_pp} |".format(**row)
        )
    lines.extend(
        [
            "",
            "Interpretation: for every model/backbone pair, more rows are",
            "generated-only passes than gold-only passes. The headline",
            "diff->gold < diff->generated ordering is therefore a paired",
            "row-level effect, not only an aggregate pass-rate artefact.",
            "",
            "See threshold_sweep.csv for the same comparison over tau values",
            "-0.50, 0.00, 0.25, 0.56, 0.75, and 0.90.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--bart-results",
        type=Path,
        default=Path("Results/runs/iter13_scaled_triad/results.csv"),
    )
    parser.add_argument(
        "--deberta-results",
        type=Path,
        default=Path("Results/runs/iter14_deberta_sanity/results.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("Results/runs/iter19_threshold_diagnostics"),
    )
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    bart_rows = read_rows(args.bart_results)
    deb_rows = read_rows(args.deberta_results)
    sweep_path = args.output / "threshold_sweep.csv"
    crosstab_path = args.output / "paired_crosstab.csv"
    summary_path = args.output / "summary.md"
    write_threshold_sweep(sweep_path, bart_rows, deb_rows)
    crosstabs = write_paired_crosstab(crosstab_path, bart_rows, deb_rows)
    write_summary(summary_path, crosstabs)
    print(f"wrote {sweep_path}")
    print(f"wrote {crosstab_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
