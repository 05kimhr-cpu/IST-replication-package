"""Iter 5b — NLI-based discriminative probe on the paired set.

Reuses the paired rows from iter 3 (gold + syn_cand + ant_cand).
Runs facebook/bart-large-mnli on (premise=gold, hypothesis=candidate)
for both syn and ant candidates. We report:

  - entail_syn,  entail_ant     : P(entail | premise=gold, hypothesis=cand)
  - contra_syn,  contra_ant     : P(contradict | ...)
  - signed_syn  = entail - contradict  (in [-1, 1], "truthiness" of cand)
  - signed_ant  = entail - contradict

If NLI is a good discriminative probe, we expect signed_syn > signed_ant
on the vast majority of pairs, with a large average delta.
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
from cmg_ist.nli import batch_nli  # noqa: E402

LANGUAGES = ["cpp", "cs", "go", "java", "js", "php", "py", "rust"]
OUT_DIR = ROOT / "runs" / "iter05b_nli"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_paired_rows() -> list[dict]:
    rows: list[dict] = []
    for lang in LANGUAGES:
        for s in load_samples(lang, limit=None):
            gold = clean_msg(s["msg"])
            toks = {t.lower().strip(".,!?;:()[]") for t in gold.split()}
            for anchor in TRIPLETS:
                if anchor not in toks:
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
                })
    return rows


def main() -> None:
    rows = collect_paired_rows()
    print(f"{len(rows)} paired rows")

    # Interleaved batch: [syn_cand_0, ant_cand_0, syn_cand_1, ant_cand_1, ...]
    prem: list[str] = []
    hyp: list[str] = []
    for r in rows:
        prem.append(r["gold"]); hyp.append(r["syn_cand"])
        prem.append(r["gold"]); hyp.append(r["ant_cand"])
    print(f"running NLI on {len(prem)} (premise, hypothesis) pairs ...")
    results = batch_nli(prem, hyp, batch_size=64)

    for i, r in enumerate(rows):
        s = results[2 * i]
        a = results[2 * i + 1]
        r["entail_syn"] = s.entailment
        r["entail_ant"] = a.entailment
        r["contra_syn"] = s.contradiction
        r["contra_ant"] = a.contradiction
        r["signed_syn"] = s.entailment - s.contradiction
        r["signed_ant"] = a.entailment - a.contradiction
        r["entail_delta"] = s.entailment - a.entailment
        r["signed_delta"] = r["signed_syn"] - r["signed_ant"]

    fieldnames = [
        "diff_id", "language", "anchor", "synonym", "antonym",
        "entail_syn", "entail_ant", "contra_syn", "contra_ant",
        "signed_syn", "signed_ant", "entail_delta", "signed_delta",
    ]
    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def desc(rs: list[dict], label: str) -> list[str]:
        if not rs:
            return [f"### {label}", "- (empty)", ""]
        n = len(rs)
        ent_d = [r["entail_delta"] for r in rs]
        sgn_d = [r["signed_delta"] for r in rs]
        syn_wins = sum(1 for r in rs if r["entail_delta"] > 0)
        strong = sum(1 for r in rs if r["entail_delta"] > 0.5)
        decisive = sum(1 for r in rs if r["signed_delta"] > 1.0)  # max 2
        avg_syn_ent = statistics.mean(r["entail_syn"] for r in rs)
        avg_ant_ent = statistics.mean(r["entail_ant"] for r in rs)
        avg_syn_sig = statistics.mean(r["signed_syn"] for r in rs)
        avg_ant_sig = statistics.mean(r["signed_ant"] for r in rs)
        return [
            f"### {label}",
            f"- n={n}",
            f"- mean entail_syn={avg_syn_ent:.3f} mean entail_ant={avg_ant_ent:.3f}",
            f"- mean signed_syn={avg_syn_sig:+.3f} mean signed_ant={avg_ant_sig:+.3f}",
            f"- entail_delta: mean={statistics.mean(ent_d):+.3f} median={statistics.median(ent_d):+.3f}",
            f"- signed_delta: mean={statistics.mean(sgn_d):+.3f} median={statistics.median(sgn_d):+.3f}",
            f"- syn>ant (entail): {syn_wins}/{n} ({100*syn_wins/n:.1f}%)",
            f"- strong delta>0.5: {strong}/{n} ({100*strong/n:.1f}%)",
            f"- decisive signed_delta>1.0: {decisive}/{n} ({100*decisive/n:.1f}%)",
            "",
        ]

    lines: list[str] = [
        "# Iter 5b — NLI discriminative probe (facebook/bart-large-mnli)", "",
        f"Paired rows: {len(rows)}", "",
    ]
    lines.extend(desc(rows, "Overall (all languages)"))
    lines.append("## Per-language")
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
