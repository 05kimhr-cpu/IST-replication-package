"""Iter 13 — Scaled triad + diff-grounded NLI (submission scale).

Replicates iter 11's three-model + diff-grounded design at N=200 per
language (=1600 samples total) rather than the N=10-per-lang pilot of
iter 10/11. Purpose: tighten per-language CIs for the construct-
validity claims in v4 of the paper.

Artefacts:
  - runs/iter13_scaled_triad/generations_{model}.jsonl
  - runs/iter13_scaled_triad/results.csv
  - runs/iter13_scaled_triad/summary.md
"""
from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT / "src"))

from cmg_ist.io import load_samples, clean_msg  # noqa: E402
from cmg_ist.metrics import bleu_sentence, rougeL, chrf, meteor, bertscore_batch  # noqa: E402
from cmg_ist.nli import batch_nli  # noqa: E402

LANGUAGES = ["cpp", "cs", "go", "java", "js", "php", "py", "rust"]
PER_LANG = 200
MIN_GOLD_TOKS = 8
MAX_DIFF_CHARS = 6000
# Model weights can be loaded either from a local directory (set
# CMG_IST_MODELS_DIR) or directly from HuggingFace by leaving the env
# variable unset — in that case the HuggingFace model IDs below are
# used.
import os  # noqa: E402

_LOCAL = os.environ.get("CMG_IST_MODELS_DIR")
MODELS = {
    "codellama-7b":      (f"{_LOCAL}/CodeLlama/7B-Instruct" if _LOCAL
                          else "codellama/CodeLlama-7b-Instruct-hf"),
    "qwen2.5-coder-7b":  (f"{_LOCAL}/Qwen/2.5/7B-Instruct" if _LOCAL
                          else "Qwen/Qwen2.5-Coder-7B-Instruct"),
    "deepseek-6.7b":     (f"{_LOCAL}/DeepSeek/6.7B-Instruct" if _LOCAL
                          else "deepseek-ai/deepseek-coder-6.7b-instruct"),
}
MAX_NEW_TOKENS = 40

OUT_DIR = ROOT / "runs" / "iter13_scaled_triad"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def select_samples() -> list[dict]:
    rows = []
    for lang in LANGUAGES:
        picked = 0
        for s in load_samples(lang, limit=None):
            gold = clean_msg(s["msg"])
            if len(gold.split()) < MIN_GOLD_TOKS:
                continue
            diff = (s.get("diff") or "")
            if not diff.strip():
                continue
            rows.append({
                "diff_id": s["diff_id"],
                "language": lang,
                "gold": gold,
                "diff": diff[:MAX_DIFF_CHARS],
            })
            picked += 1
            if picked >= PER_LANG:
                break
        print(f"  [{lang}] picked {picked}", flush=True)
    return rows


def build_prompt(model_slug: str, diff: str) -> str:
    instr = (
        "Given the following diff, write a single-line git commit message "
        "(under 20 words). Output only the commit message — no explanation."
    )
    if model_slug.startswith("qwen"):
        return f"<|im_start|>user\n{instr}\n\nDiff:\n{diff}\n<|im_end|>\n<|im_start|>assistant\n"
    if model_slug.startswith("deepseek"):
        return f"{instr}\n\nDiff:\n{diff}\n### Response:\n"
    # codellama-instruct
    return f"[INST] {instr}\n\nDiff:\n{diff}\n[/INST]"


def generate_for_model(model_slug: str, path: str, rows: list[dict]) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[{model_slug}] loading {path} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        path, torch_dtype=torch.float16, device_map={"": 0}, trust_remote_code=True
    ).eval()
    print(f"[{model_slug}]   loaded in {time.time() - t0:.1f}s", flush=True)

    key_gen = f"gen_{model_slug}"
    for i, r in enumerate(rows):
        prompt = build_prompt(model_slug, r["diff"])
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(
                **inp,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        decoded = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        first = next((ln.strip() for ln in decoded.splitlines() if ln.strip()), "")
        r[key_gen] = first
        if (i + 1) % 100 == 0 or (i + 1) == len(rows):
            print(f"[{model_slug}]   {i + 1}/{len(rows)}  [{r['language']}] {first[:60]}", flush=True)

    out_path = OUT_DIR / f"generations_{model_slug}.jsonl"
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps({
                "diff_id": r["diff_id"], "language": r["language"],
                "gold": r["gold"], "generated": r[key_gen],
            }) + "\n")
    print(f"[{model_slug}] wrote {out_path}", flush=True)

    del mdl
    torch.cuda.empty_cache()


def score_model(rows: list[dict], model_slug: str) -> None:
    key_gen = f"gen_{model_slug}"
    cands = [r[key_gen] for r in rows]
    refs = [r["gold"] for r in rows]
    diffs = [r["diff"][:1500] for r in rows]

    for r in rows:
        g = r[key_gen]
        r[f"bleu_{model_slug}"] = bleu_sentence(g, r["gold"])
        r[f"rougeL_{model_slug}"] = rougeL(g, r["gold"])
        r[f"chrf_{model_slug}"] = chrf(g, r["gold"])
        r[f"meteor_{model_slug}"] = meteor(g, r["gold"])

    print(f"[{model_slug}] bertscore ...", flush=True)
    bs = bertscore_batch(cands, refs, model_type="roberta-large", device="cuda")
    for r, v in zip(rows, bs):
        r[f"bertscore_{model_slug}"] = v

    print(f"[{model_slug}] NLI gold→gen ...", flush=True)
    nli = batch_nli(refs, cands, batch_size=32)
    for r, n in zip(rows, nli):
        r[f"nli_signed_gold→gen_{model_slug}"] = n.entailment - n.contradiction
        r[f"nli_entail_gold→gen_{model_slug}"] = n.entailment

    print(f"[{model_slug}] NLI diff→gen ...", flush=True)
    nli_d = batch_nli(diffs, cands, batch_size=16)
    for r, n in zip(rows, nli_d):
        r[f"nli_signed_diff→gen_{model_slug}"] = n.entailment - n.contradiction
        r[f"nli_entail_diff→gen_{model_slug}"] = n.entailment


def score_gold_diff(rows: list[dict]) -> None:
    refs = [r["gold"] for r in rows]
    diffs = [r["diff"][:1500] for r in rows]
    print(f"[ref ] NLI diff→gold ...", flush=True)
    nli = batch_nli(diffs, refs, batch_size=16)
    for r, n in zip(rows, nli):
        r["nli_signed_diff→gold"] = n.entailment - n.contradiction
        r["nli_entail_diff→gold"] = n.entailment


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    halfw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, centre - halfw), min(1.0, centre + halfw)


def summarize(rows: list[dict]) -> str:
    model_slugs = list(MODELS.keys())
    lines = [
        "# Iter 13 — Scaled triad + diff-grounded NLI",
        "",
        f"Samples: {len(rows)} (target {PER_LANG} × {len(LANGUAGES)} = {PER_LANG * len(LANGUAGES)})",
        "",
        "## Gold ↔ Generated: NLI pass-rate (τ = +0.56)",
        "",
        "| model              |   n | pass ≥ +0.56 (95% CI)          | mean signed | mean BERTScore |",
        "|--------------------|-----|--------------------------------|-------------|----------------|",
    ]
    for ms in model_slugs:
        k_sig = f"nli_signed_gold→gen_{ms}"
        k_bs = f"bertscore_{ms}"
        vs = [r[k_sig] for r in rows if k_sig in r]
        bs = [r[k_bs] for r in rows if k_bs in r]
        if not vs:
            continue
        above = sum(1 for v in vs if v >= 0.56)
        lo, hi = wilson_ci(above, len(vs))
        lines.append(
            f"| {ms:<18} | {len(vs):>3} | {above}/{len(vs)} ({100*above/len(vs):.1f}%)  [{100*lo:.1f}–{100*hi:.1f}] | "
            f"{statistics.mean(vs):+.3f} | {statistics.mean(bs):+.3f} |"
        )

    lines.append("")
    lines.append("## Diff ↔ X : source-of-truth entailment")
    lines.append("")
    lines.append("| probe                      |   n | pass ≥ +0.56 (95% CI)          | mean signed | mean entail |")
    lines.append("|----------------------------|-----|--------------------------------|-------------|-------------|")
    vs_gold = [r["nli_signed_diff→gold"] for r in rows if "nli_signed_diff→gold" in r]
    ent_gold = [r["nli_entail_diff→gold"] for r in rows if "nli_entail_diff→gold" in r]
    above = sum(1 for v in vs_gold if v >= 0.56)
    lo, hi = wilson_ci(above, len(vs_gold))
    lines.append(
        f"| diff → gold                | {len(vs_gold):>3} | {above}/{len(vs_gold)} ({100*above/len(vs_gold):.1f}%)  [{100*lo:.1f}–{100*hi:.1f}] | "
        f"{statistics.mean(vs_gold):+.3f} | {statistics.mean(ent_gold):+.3f} |"
    )
    for ms in model_slugs:
        k_sig = f"nli_signed_diff→gen_{ms}"
        k_ent = f"nli_entail_diff→gen_{ms}"
        vs = [r[k_sig] for r in rows if k_sig in r]
        ent = [r[k_ent] for r in rows if k_ent in r]
        if not vs:
            continue
        above = sum(1 for v in vs if v >= 0.56)
        lo, hi = wilson_ci(above, len(vs))
        lines.append(
            f"| diff → gen ({ms:<17}) | {len(vs):>3} | {above}/{len(vs)} ({100*above/len(vs):.1f}%)  [{100*lo:.1f}–{100*hi:.1f}] | "
            f"{statistics.mean(vs):+.3f} | {statistics.mean(ent):+.3f} |"
        )

    lines.append("")
    lines.append("## Per-language diff→gold pass-rates (τ = +0.56)")
    lines.append("")
    lines.append("| lang |  n  | pass (95% CI)            | mean signed |")
    lines.append("|------|-----|--------------------------|-------------|")
    for lang in LANGUAGES:
        vs = [r["nli_signed_diff→gold"] for r in rows if r["language"] == lang]
        if not vs:
            continue
        above = sum(1 for v in vs if v >= 0.56)
        lo, hi = wilson_ci(above, len(vs))
        lines.append(
            f"| {lang:<4} | {len(vs):>3} | {above}/{len(vs)} ({100*above/len(vs):.1f}%)  [{100*lo:.1f}–{100*hi:.1f}] | "
            f"{statistics.mean(vs):+.3f} |"
        )

    return "\n".join(lines) + "\n"


def main() -> None:
    t_start = time.time()
    rows = select_samples()
    print(f"selected {len(rows)} samples (target {PER_LANG * len(LANGUAGES)})", flush=True)

    for slug, path in MODELS.items():
        generate_for_model(slug, path, rows)

    for slug in MODELS:
        score_model(rows, slug)
    score_gold_diff(rows)

    fixed = ["diff_id", "language", "gold",
             "gen_codellama-7b", "gen_qwen2.5-coder-7b", "gen_deepseek-6.7b"]
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    others = sorted(k for k in all_keys if k not in fixed and k != "diff")
    fieldnames = fixed + others
    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}", flush=True)

    summary = summarize(rows)
    (OUT_DIR / "summary.md").write_text(summary)
    print(summary)
    print(f"total elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
