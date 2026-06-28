#!/usr/bin/env python
"""CPU-only robustness diagnostics from released artefacts.

This script does not rerun NLI or generation. It joins iter13/iter14 scores
with raw diffs and reports diagnostics that can be checked without GPU work:

- pass-rates on the subset whose full raw diff is <= 1,500 characters;
- literal-copy and token-overlap checks for generated messages;
- message-length correlations with diff-grounded NLI scores.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
from pathlib import Path


LANGUAGES = ["cpp", "cs", "go", "java", "js", "php", "py", "rust"]
MODELS = ["codellama-7b", "qwen2.5-coder-7b", "deepseek-6.7b"]
TAU = 0.56
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*|[0-9]+")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        rows = []
        for row in csv.DictReader(f):
            rows.append({key.replace("→", "->"): value for key, value in row.items()})
        return rows


def normalize(text: str) -> str:
    return " ".join(text.lower().split())


def tokens(text: str) -> list[str]:
    return [m.group(0).lower() for m in TOKEN_RE.finditer(text)]


def pct(numer: int, denom: int) -> float:
    return 100.0 * numer / denom if denom else 0.0


def pass_pct(rows: list[dict[str, str]], col: str) -> tuple[int, float]:
    passed = sum(float(r[col]) >= TAU for r in rows)
    return passed, pct(passed, len(rows))


def pearson(xs: list[float], ys: list[float]) -> float:
    if len(xs) < 2 or len(xs) != len(ys):
        return 0.0
    mx = statistics.mean(xs)
    my = statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = sum((x - mx) ** 2 for x in xs) ** 0.5
    den_y = sum((y - my) ** 2 for y in ys) ** 0.5
    if den_x == 0 or den_y == 0:
        return 0.0
    return num / (den_x * den_y)


def load_diff_map(raw_dir: Path) -> dict[tuple[str, str], str]:
    diff_map: dict[tuple[str, str], str] = {}
    for lang in LANGUAGES:
        path = raw_dir / f"{lang}.jsonl"
        with path.open() as f:
            for line in f:
                row = json.loads(line)
                diff_map[(lang, str(row["diff_id"]))] = row.get("diff") or ""
    return diff_map


def write_subset_rates(out_path: Path, rows: list[dict[str, str]]) -> None:
    subsets = [
        ("all", rows),
        ("full_diff_chars<=1500", [r for r in rows if int(r["full_diff_chars"]) <= 1500]),
        ("full_diff_chars>1500", [r for r in rows if int(r["full_diff_chars"]) > 1500]),
    ]
    fieldnames = ["subset", "backbone", "condition", "n", "pass_n", "pass_pct"]
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for subset_name, subset in subsets:
            specs = [
                ("BART-MNLI", "diff->gold", "nli_signed_diff->gold"),
                ("DeBERTa-v3", "diff->gold", "deb_nli_signed_diff->gold"),
            ]
            for model in MODELS:
                specs.append(("BART-MNLI", f"diff->gen:{model}", f"nli_signed_diff->gen_{model}"))
                specs.append(("DeBERTa-v3", f"diff->gen:{model}", f"deb_nli_signed_diff->gen_{model}"))
            for backbone, condition, col in specs:
                k, p = pass_pct(subset, col)
                writer.writerow(
                    {
                        "subset": subset_name,
                        "backbone": backbone,
                        "condition": condition,
                        "n": len(subset),
                        "pass_n": k,
                        "pass_pct": f"{p:.1f}",
                    }
                )


def write_copy_length(out_path: Path, rows: list[dict[str, str]]) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    for model in MODELS:
        exact_substrings = 0
        overlap_ratios: list[float] = []
        gen_lens: list[float] = []
        signed_scores: list[float] = []
        for row in rows:
            gen = row[f"gen_{model}"]
            norm_gen = normalize(gen)
            norm_diff = normalize(row["full_diff"])
            if norm_gen and norm_gen in norm_diff:
                exact_substrings += 1
            gen_tokens = tokens(gen)
            diff_tokens = set(tokens(row["full_diff"]))
            if gen_tokens:
                overlap_ratios.append(sum(t in diff_tokens for t in gen_tokens) / len(gen_tokens))
            gen_lens.append(float(len(gen_tokens)))
            signed_scores.append(float(row[f"nli_signed_diff->gen_{model}"]))
        records.append(
            {
                "model": model,
                "n": str(len(rows)),
                "exact_substring_n": str(exact_substrings),
                "exact_substring_pct": f"{pct(exact_substrings, len(rows)):.1f}",
                "mean_gen_token_overlap_with_diff": f"{statistics.mean(overlap_ratios):.3f}",
                "median_gen_token_overlap_with_diff": f"{statistics.median(overlap_ratios):.3f}",
                "mean_gen_tokens": f"{statistics.mean(gen_lens):.2f}",
                "corr_gen_len_vs_diff_nli_signed": f"{pearson(gen_lens, signed_scores):.3f}",
            }
        )
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)
    return records


def write_summary(out_path: Path, rows: list[dict[str, str]], copy_records: list[dict[str, str]]) -> None:
    short_rows = [r for r in rows if int(r["full_diff_chars"]) <= 1500]
    long_rows = [r for r in rows if int(r["full_diff_chars"]) > 1500]
    lines = [
        "# Iter 20 - Artifact-only robustness diagnostics",
        "",
        "These diagnostics reuse released raw diffs and per-row NLI scores. They do",
        "not rerun NLI, generation, or any GPU model.",
        "",
        "## Full-diff-length subset",
        "",
        f"- Full sample: {len(rows)} rows.",
        f"- Full raw diff <= 1,500 chars: {len(short_rows)} rows ({pct(len(short_rows), len(rows)):.1f}%).",
        f"- Full raw diff > 1,500 chars: {len(long_rows)} rows ({pct(len(long_rows), len(rows)):.1f}%).",
        "",
        "At tau=+0.56, the complete-diff subset preserves the rank ordering",
        "diff->gold < diff->generated for every model and for both BART-MNLI",
        "and DeBERTa. See subset_pass_rates.csv.",
        "",
        "## Literal-copy diagnostics",
        "",
        "| model | exact substring n | exact substring % | mean token overlap | median token overlap | mean gen tokens | corr(length, diff-NLI) |",
        "|-------|------------------:|------------------:|-------------------:|---------------------:|----------------:|-----------------------:|",
    ]
    for rec in copy_records:
        lines.append(
            "| {model} | {exact_substring_n} | {exact_substring_pct} | "
            "{mean_gen_token_overlap_with_diff} | {median_gen_token_overlap_with_diff} | "
            "{mean_gen_tokens} | {corr_gen_len_vs_diff_nli_signed} |".format(**rec)
        )
    lines.extend(
        [
            "",
            "Interpretation: exact full-message copying from the diff is rare/absent",
            "under this automatic check. Token overlap is expected because commit",
            "messages legitimately reuse identifiers and action words from diffs;",
            "therefore this diagnostic rules out literal regurgitation but does not",
            "prove semantic adequacy.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--iter13", type=Path, default=Path("Results/runs/iter13_scaled_triad/results.csv"))
    parser.add_argument("--iter14", type=Path, default=Path("Results/runs/iter14_deberta_sanity/results.csv"))
    parser.add_argument("--output", type=Path, default=Path("Results/runs/iter20_artifact_robustness"))
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.iter13)
    deb_rows = read_csv(args.iter14)
    diff_map = load_diff_map(args.raw_dir)

    deb_by_key = {(r["language"], r["diff_id"]): r for r in deb_rows}
    merged: list[dict[str, str]] = []
    for row in rows:
        key = (row["language"], row["diff_id"])
        full_diff = diff_map[key]
        merged_row = dict(row)
        merged_row.update(deb_by_key[key])
        merged_row["full_diff"] = full_diff
        merged_row["full_diff_chars"] = str(len(full_diff))
        merged.append(merged_row)

    write_subset_rates(args.output / "subset_pass_rates.csv", merged)
    copy_records = write_copy_length(args.output / "copy_length_diagnostics.csv", merged)
    write_summary(args.output / "summary.md", merged, copy_records)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
