"""Iter 7 — paired target-noun perturbation across 8 languages.

Mirror of iter 3/4/5b but swapping nouns. Runs all 5 standard metrics +
the NLI probe in one pass for efficiency. Reports per-kind breakdown so
cleaner triplet families (directional: import/export) are visible
separately from fuzzier ones (close_entity: function/variable).

Outputs:
  runs/iter07_noun/results.csv
  runs/iter07_noun/summary.md
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
from cmg_ist.perturbation_noun_pairs import TRIPLETS_NOUN, KINDS, apply_noun_pair  # noqa: E402
from cmg_ist.metrics import bleu_sentence, rougeL, chrf, meteor, bertscore_batch  # noqa: E402
from cmg_ist.nli import batch_nli  # noqa: E402

LANGUAGES = ["cpp", "cs", "go", "java", "js", "php", "py", "rust"]
OUT_DIR = ROOT / "runs" / "iter07_noun"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_rows() -> list[dict]:
    rows: list[dict] = []
    for lang in LANGUAGES:
        for s in load_samples(lang, limit=None):
            gold = clean_msg(s["msg"])
            toks = {t.lower().strip(".,!?;:()[]") for t in gold.split()}
            for anchor in TRIPLETS_NOUN:
                if anchor not in toks:
                    continue
                res = apply_noun_pair(gold, anchor)
                if res is None:
                    continue
                syn_c, dis_c, info = res
                rows.append({
                    "diff_id": s["diff_id"],
                    "language": lang,
                    "anchor": anchor,
                    "synonym": info.synonym,
                    "disjoint": info.disjoint,
                    "kind": info.kind,
                    "gold": gold,
                    "syn_cand": syn_c,
                    "dis_cand": dis_c,
                })
    return rows


def main() -> None:
    rows = collect_rows()
    print(f"collected {len(rows)} paired rows")

    # Per-sample text metrics (cheap)
    for r in rows:
        r["bleu_syn"] = bleu_sentence(r["syn_cand"], r["gold"])
        r["bleu_dis"] = bleu_sentence(r["dis_cand"], r["gold"])
        r["rougeL_syn"] = rougeL(r["syn_cand"], r["gold"])
        r["rougeL_dis"] = rougeL(r["dis_cand"], r["gold"])
        r["chrf_syn"] = chrf(r["syn_cand"], r["gold"])
        r["chrf_dis"] = chrf(r["dis_cand"], r["gold"])
        r["meteor_syn"] = meteor(r["syn_cand"], r["gold"])
        r["meteor_dis"] = meteor(r["dis_cand"], r["gold"])

    # Batched BERTScore
    cands = []
    refs = []
    for r in rows:
        cands.append(r["syn_cand"]); refs.append(r["gold"])
        cands.append(r["dis_cand"]); refs.append(r["gold"])
    print(f"running BERTScore on {len(cands)} pairs ...")
    bert_f1s = bertscore_batch(cands, refs)
    for i, r in enumerate(rows):
        r["bertscore_syn"] = bert_f1s[2 * i]
        r["bertscore_dis"] = bert_f1s[2 * i + 1]

    # Batched NLI
    prem = []
    hyp = []
    for r in rows:
        prem.append(r["gold"]); hyp.append(r["syn_cand"])
        prem.append(r["gold"]); hyp.append(r["dis_cand"])
    print(f"running NLI on {len(prem)} pairs ...")
    nli_res = batch_nli(prem, hyp)
    for i, r in enumerate(rows):
        s = nli_res[2 * i]
        d = nli_res[2 * i + 1]
        r["entail_syn"] = s.entailment
        r["entail_dis"] = d.entailment
        r["signed_syn"] = s.entailment - s.contradiction
        r["signed_dis"] = d.entailment - d.contradiction

    # Deltas
    metrics = ["bleu", "rougeL", "chrf", "meteor", "bertscore", "entail", "signed"]
    for r in rows:
        for m in metrics:
            r[f"{m}_delta"] = r[f"{m}_syn"] - r[f"{m}_dis" if m in ("bleu", "rougeL", "chrf", "meteor", "bertscore") else f"{m}_dis"]

    # Write CSV
    fieldnames = (
        ["diff_id", "language", "anchor", "synonym", "disjoint", "kind"]
        + [f"{m}_{suf}" for m in ["bleu", "rougeL", "chrf", "meteor", "bertscore", "entail", "signed"]
                         for suf in ("syn", "dis", "delta")]
    )
    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    def desc(rs: list[dict], label: str) -> list[str]:
        if not rs:
            return [f"### {label}", "- (empty)", ""]
        n = len(rs)
        out = [f"### {label} (n={n})", ""]
        out.append("| metric | mean | mean\\|x\\| | |x|<0.01 | syn>dis | strong>0.5 |")
        out.append("|--------|------|----------|---------|---------|-----------|")
        for m in metrics:
            xs = [r[f"{m}_delta"] for r in rs]
            mean = statistics.mean(xs)
            absmean = statistics.mean(abs(x) for x in xs)
            near = sum(1 for x in xs if abs(x) < 0.01)
            synwins = sum(1 for x in xs if x > 0)
            strong = sum(1 for x in xs if x > 0.5)
            out.append(
                f"| {m} | {mean:+.4f} | {absmean:.4f} | {near}/{n} "
                f"({100*near/n:.0f}%) | {synwins}/{n} ({100*synwins/n:.0f}%) | "
                f"{strong}/{n} ({100*strong/n:.0f}%) |"
            )
        out.append("")
        return out

    lines: list[str] = [
        "# Iter 7 — paired target-noun perturbation",
        "",
        f"- 8 languages, {len(rows)} paired rows, {len(TRIPLETS_NOUN)} anchor nouns",
        "- syn = near-synonym (meaning preserved)",
        "- dis = reference-disjoint (commit claim changed)",
        "",
    ]
    lines.extend(desc(rows, "Overall"))
    lines.append("## By triplet kind")
    lines.append("")
    for kind in sorted({k for k in KINDS.values()}):
        sub = [r for r in rows if r["kind"] == kind]
        lines.extend(desc(sub, kind))
    lines.append("## By language")
    lines.append("")
    for lang in LANGUAGES:
        sub = [r for r in rows if r["language"] == lang]
        lines.extend(desc(sub, lang))
    lines.append("## By anchor (top 10 by frequency)")
    lines.append("")
    anchor_counts = {}
    for r in rows:
        anchor_counts[r["anchor"]] = anchor_counts.get(r["anchor"], 0) + 1
    top = sorted(anchor_counts.items(), key=lambda x: -x[1])[:10]
    for anchor, _ in top:
        sub = [r for r in rows if r["anchor"] == anchor]
        lines.extend(desc(sub, anchor))

    (OUT_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    print(f"wrote {csv_path}")
    print(f"wrote {OUT_DIR / 'summary.md'}")
    print()
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
