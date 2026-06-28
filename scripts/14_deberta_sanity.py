"""Iter 14 — NLI model sensitivity (DeBERTa-v3-large-MNLI).

Reuse iter 13's 1600 samples (CodeLlama + Qwen + DeepSeek generations)
and re-score every NLI probe with `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli`
(state-of-the-art MNLI accuracy). If the construct-validity pattern
(diff→gold < diff→gen; gold→gen ~10%) holds under a different NLI
backbone, reviewers cannot blame the finding on the BART-MNLI choice.

Artefacts:
  - runs/iter14_deberta_sanity/summary.md
  - runs/iter14_deberta_sanity/results.csv
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

ITER13_CSV = ROOT / "runs" / "iter13_scaled_triad" / "results.csv"
MODEL_SLUGS = ["codellama-7b", "qwen2.5-coder-7b", "deepseek-6.7b"]
DEBERTA_MODEL = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
OUT_DIR = ROOT / "runs" / "iter14_deberta_sanity"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    halfw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, centre - halfw), min(1.0, centre + halfw)


def batch_nli_deberta(tok, mdl, prem: list[str], hyp: list[str], batch_size: int = 32):
    """DeBERTa MNLI label order: entailment=0, neutral=1, contradiction=2.
    Returns list of (entail, contra) pairs."""
    out = []
    for i in range(0, len(prem), batch_size):
        p = prem[i:i + batch_size]
        h = hyp[i:i + batch_size]
        inp = tok(p, h, return_tensors="pt", truncation=True, padding=True, max_length=512).to(mdl.device)
        with torch.no_grad():
            logits = mdl(**inp).logits
        probs = torch.softmax(logits, dim=-1).cpu().tolist()
        for q in probs:
            out.append((q[0], q[2]))  # entail, contra
    return out


def main() -> None:
    print(f"loading {DEBERTA_MODEL} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(DEBERTA_MODEL)
    mdl = AutoModelForSequenceClassification.from_pretrained(DEBERTA_MODEL, torch_dtype=torch.float16).to("cuda").eval()
    print("  loaded", flush=True)

    # Load iter 13 rows
    rows: list[dict] = []
    with ITER13_CSV.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            rows.append(r)
    print(f"loaded {len(rows)} iter13 rows", flush=True)

    # Reload diffs — they were excluded from iter13 CSV. Read directly from raw.
    from cmg_ist.io import load_samples, clean_msg  # noqa: E402
    diff_map: dict[str, str] = {}
    needed = {(r["language"], r["diff_id"]) for r in rows}
    for lang in sorted({r["language"] for r in rows}):
        for s in load_samples(lang, limit=None):
            k = (lang, str(s["diff_id"]))
            if k in needed:
                diff_map[f"{lang}:{s['diff_id']}"] = (s.get("diff") or "")[:1500]
    print(f"mapped {len(diff_map)} diffs", flush=True)

    golds = [r["gold"] for r in rows]
    diffs = [diff_map[f"{r['language']}:{r['diff_id']}"] for r in rows]

    # diff→gold (DeBERTa)
    print("[DeBERTa] diff→gold ...", flush=True)
    res = batch_nli_deberta(tok, mdl, diffs, golds)
    for r, (e, c) in zip(rows, res):
        r["deb_nli_signed_diff→gold"] = e - c

    for ms in MODEL_SLUGS:
        gens = [r[f"gen_{ms}"] for r in rows]

        print(f"[DeBERTa] gold→gen ({ms}) ...", flush=True)
        res = batch_nli_deberta(tok, mdl, golds, gens)
        for r, (e, c) in zip(rows, res):
            r[f"deb_nli_signed_gold→gen_{ms}"] = e - c

        print(f"[DeBERTa] diff→gen ({ms}) ...", flush=True)
        res = batch_nli_deberta(tok, mdl, diffs, gens)
        for r, (e, c) in zip(rows, res):
            r[f"deb_nli_signed_diff→gen_{ms}"] = e - c

    # Write per-row CSV (only DeBERTa columns + key cols)
    cols = ["diff_id", "language",
            "deb_nli_signed_diff→gold"]
    for ms in MODEL_SLUGS:
        cols += [f"deb_nli_signed_gold→gen_{ms}", f"deb_nli_signed_diff→gen_{ms}"]
    with (OUT_DIR / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_DIR/'results.csv'}", flush=True)

    # Summary with τ=+0.56 threshold (same as BART calibration for cross-comparability)
    lines = [
        f"# Iter 14 — NLI backbone sensitivity: {DEBERTA_MODEL}",
        "",
        "τ = +0.56 (paper's BART-calibrated threshold reused as-is for cross-comparison)",
        f"N = {len(rows)}",
        "",
        "## Gold → Generated (DeBERTa signed)",
        "",
        "| model            | pass ≥ +0.56 (95% CI)     | mean signed |",
        "|------------------|---------------------------|-------------|",
    ]
    for ms in MODEL_SLUGS:
        vs = [float(r[f"deb_nli_signed_gold→gen_{ms}"]) for r in rows]
        above = sum(1 for v in vs if v >= 0.56)
        lo, hi = wilson_ci(above, len(vs))
        lines.append(f"| {ms:<16} | {above}/{len(vs)} ({100*above/len(vs):.1f}%)  [{100*lo:.1f}–{100*hi:.1f}] | {statistics.mean(vs):+.3f} |")

    lines += ["", "## Diff → X (DeBERTa signed)", "",
              "| probe                  | pass ≥ +0.56 (95% CI)     | mean signed |",
              "|------------------------|---------------------------|-------------|"]
    vs = [float(r["deb_nli_signed_diff→gold"]) for r in rows]
    above = sum(1 for v in vs if v >= 0.56)
    lo, hi = wilson_ci(above, len(vs))
    lines.append(f"| diff → gold            | {above}/{len(vs)} ({100*above/len(vs):.1f}%)  [{100*lo:.1f}–{100*hi:.1f}] | {statistics.mean(vs):+.3f} |")
    for ms in MODEL_SLUGS:
        vs = [float(r[f"deb_nli_signed_diff→gen_{ms}"]) for r in rows]
        above = sum(1 for v in vs if v >= 0.56)
        lo, hi = wilson_ci(above, len(vs))
        lines.append(f"| diff → gen ({ms:<14}) | {above}/{len(vs)} ({100*above/len(vs):.1f}%)  [{100*lo:.1f}–{100*hi:.1f}] | {statistics.mean(vs):+.3f} |")

    summary = "\n".join(lines) + "\n"
    (OUT_DIR / "summary.md").write_text(summary)
    print(summary)
    print("DONE")


if __name__ == "__main__":
    main()
