"""Iter 11 — Multi-model real-generation pilot.

Extend iter 10 (CodeLlama-7B-Instruct) to Qwen2.5-Coder-7B-Instruct and
DeepSeek-Coder-6.7B-Instruct on the SAME 80 diffs. Purpose: decide
whether the low NLI pass-rate in iter 10 is model-specific (capacity)
or reference-specific (construct mismatch between gold and generated).

If all three models land at similar NLI pass-rates (~7%), the NLI
probe's calibration does not transfer from paired protocol to
free-form generation — a reference-based measurement problem.

Artefacts:
  - runs/iter11_real_model_triad/generations_{model}.jsonl
  - runs/iter11_real_model_triad/results.csv       (one row per sample×model)
  - runs/iter11_real_model_triad/summary.md
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

from cmg_ist.metrics import bleu_sentence, rougeL, chrf, meteor, bertscore_batch  # noqa: E402
from cmg_ist.nli import batch_nli  # noqa: E402

import os  # noqa: E402

_LOCAL = os.environ.get("CMG_IST_MODELS_DIR")
MODELS = {
    "qwen2.5-coder-7b":  (f"{_LOCAL}/Qwen/2.5/7B-Instruct" if _LOCAL
                          else "Qwen/Qwen2.5-Coder-7B-Instruct"),
    "deepseek-6.7b":     (f"{_LOCAL}/DeepSeek/6.7B-Instruct" if _LOCAL
                          else "deepseek-ai/deepseek-coder-6.7b-instruct"),
}
MAX_NEW_TOKENS = 40
ITER10_GEN = ROOT / "runs" / "iter10_real_model_pilot" / "generations.jsonl"
OUT_DIR = ROOT / "runs" / "iter11_real_model_triad"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_iter10_samples() -> list[dict]:
    """Reuse iter10 selection (diff_id, language, gold, generated[codellama])."""
    rows = []
    for line in ITER10_GEN.open():
        r = json.loads(line)
        rows.append({
            "diff_id": r["diff_id"],
            "language": r["language"],
            "gold": r["gold"],
            "gen_codellama-7b": r["generated"],
        })
    return rows


def load_diffs_for_samples(rows: list[dict]) -> dict[str, str]:
    """Re-read raw MCMD for each (lang, diff_id) to fetch the diff text."""
    from cmg_ist.io import load_samples
    needed = {(r["language"], r["diff_id"]) for r in rows}
    diff_map: dict[str, str] = {}
    for lang in sorted({r["language"] for r in rows}):
        # iter 10 never selected past the per-lang cutoff of 10 after filter;
        # the filter keeps only gold >= 8 toks so we scan the whole raw file.
        for s in load_samples(lang, limit=None):
            k = (lang, s["diff_id"])
            if k in needed:
                diff_map[f"{lang}:{s['diff_id']}"] = (s.get("diff") or "")[:6000]
    return diff_map


def build_prompt(model_slug: str, diff: str) -> str:
    instruction = (
        "Given the following diff, write a single-line git commit message "
        "(under 20 words). Output only the commit message — no explanation."
    )
    if model_slug.startswith("qwen"):
        return (
            "<|im_start|>user\n"
            f"{instruction}\n\nDiff:\n{diff}\n<|im_end|>\n"
            "<|im_start|>assistant\n"
        )
    if model_slug.startswith("deepseek"):
        return (
            f"{instruction}\n\nDiff:\n{diff}\n"
            "### Response:\n"
        )
    # fallback: plain instruct
    return f"{instruction}\n\nDiff:\n{diff}\n\nCommit message:"


def generate_for_model(model_slug: str, path: str, rows: list[dict], diff_map: dict[str, str]) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"[{model_slug}] loading {path} in fp16 ...", flush=True)
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
        diff = diff_map[f"{r['language']}:{r['diff_id']}"]
        prompt = build_prompt(model_slug, diff)
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
        if (i + 1) % 20 == 0 or (i + 1) == len(rows):
            print(f"[{model_slug}]   gen {i + 1}/{len(rows)}: [{r['language']}] {first[:80]}", flush=True)

    # Persist per-model generations
    out_path = OUT_DIR / f"generations_{model_slug}.jsonl"
    with out_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps({
                "diff_id": r["diff_id"],
                "language": r["language"],
                "gold": r["gold"],
                "generated": r[key_gen],
            }) + "\n")
    print(f"[{model_slug}] wrote {out_path}", flush=True)

    del mdl
    torch.cuda.empty_cache()


def score_block(rows: list[dict], model_slug: str, diff_map: dict[str, str]) -> list[dict]:
    """One scored row per sample, tagged with model_slug."""
    key_gen = f"gen_{model_slug}"
    cands = [r[key_gen] for r in rows]
    refs = [r["gold"] for r in rows]
    diffs = [diff_map[f"{r['language']}:{r['diff_id']}"] for r in rows]

    # Text metrics
    for r in rows:
        g = r[key_gen]
        r[f"bleu_{model_slug}"] = bleu_sentence(g, r["gold"])
        r[f"rougeL_{model_slug}"] = rougeL(g, r["gold"])
        r[f"chrf_{model_slug}"] = chrf(g, r["gold"])
        r[f"meteor_{model_slug}"] = meteor(g, r["gold"])

    print(f"[{model_slug}] bertscore (gold ↔ gen) ...", flush=True)
    bs = bertscore_batch(cands, refs, model_type="roberta-large", device="cuda")
    for r, v in zip(rows, bs):
        r[f"bertscore_{model_slug}"] = v

    print(f"[{model_slug}] NLI (gold → gen) ...", flush=True)
    nli = batch_nli(refs, cands, batch_size=32)
    for r, n in zip(rows, nli):
        r[f"nli_signed_gold→gen_{model_slug}"] = n.entailment - n.contradiction
        r[f"nli_entail_gold→gen_{model_slug}"] = n.entailment

    # iter 12 portion: diff ↔ gen NLI (truncate diff for NLI tokenizer)
    print(f"[{model_slug}] NLI (diff → gen) ...", flush=True)
    diff_prem = [d[:1500] for d in diffs]  # bart-mnli max 1024 tokens
    nli_d = batch_nli(diff_prem, cands, batch_size=16)
    for r, n in zip(rows, nli_d):
        r[f"nli_signed_diff→gen_{model_slug}"] = n.entailment - n.contradiction
        r[f"nli_entail_diff→gen_{model_slug}"] = n.entailment
    return rows


def score_gold_diff_once(rows: list[dict], diff_map: dict[str, str]) -> None:
    """Diff → gold NLI (iter 12 kernel): does the gold message itself
    follow from the diff? If not, reference-based eval is construct-broken
    regardless of model quality."""
    refs = [r["gold"] for r in rows]
    diffs = [diff_map[f"{r['language']}:{r['diff_id']}"][:1500] for r in rows]
    print(f"[ref ] NLI (diff → gold) ...", flush=True)
    nli = batch_nli(diffs, refs, batch_size=16)
    for r, n in zip(rows, nli):
        r["nli_signed_diff→gold"] = n.entailment - n.contradiction
        r["nli_entail_diff→gold"] = n.entailment


def iter10_score_block(rows: list[dict], diff_map: dict[str, str]) -> None:
    """Fill in CodeLlama's iter10 metrics into the unified csv rows."""
    model_slug = "codellama-7b"
    cands = [r[f"gen_{model_slug}"] for r in rows]
    refs = [r["gold"] for r in rows]
    diffs = [diff_map[f"{r['language']}:{r['diff_id']}"][:1500] for r in rows]

    for r in rows:
        g = r[f"gen_{model_slug}"]
        r[f"bleu_{model_slug}"] = bleu_sentence(g, r["gold"])
        r[f"rougeL_{model_slug}"] = rougeL(g, r["gold"])
        r[f"chrf_{model_slug}"] = chrf(g, r["gold"])
        r[f"meteor_{model_slug}"] = meteor(g, r["gold"])

    print(f"[{model_slug}] bertscore (gold ↔ gen) ...", flush=True)
    bs = bertscore_batch(cands, refs, model_type="roberta-large", device="cuda")
    for r, v in zip(rows, bs):
        r[f"bertscore_{model_slug}"] = v

    print(f"[{model_slug}] NLI (gold → gen) ...", flush=True)
    nli = batch_nli(refs, cands, batch_size=32)
    for r, n in zip(rows, nli):
        r[f"nli_signed_gold→gen_{model_slug}"] = n.entailment - n.contradiction
        r[f"nli_entail_gold→gen_{model_slug}"] = n.entailment

    print(f"[{model_slug}] NLI (diff → gen) ...", flush=True)
    nli_d = batch_nli(diffs, cands, batch_size=16)
    for r, n in zip(rows, nli_d):
        r[f"nli_signed_diff→gen_{model_slug}"] = n.entailment - n.contradiction
        r[f"nli_entail_diff→gen_{model_slug}"] = n.entailment


def summarize(rows: list[dict]) -> str:
    model_slugs = ["codellama-7b", "qwen2.5-coder-7b", "deepseek-6.7b"]
    lines = [
        "# Iter 11 — Multi-model + diff-grounded NLI",
        "",
        f"Samples: {len(rows)} (reused iter10 selection)",
        "",
        "## Gold ↔ Generated: NLI pass-rates at paper's τ = +0.56",
        "",
        "| model              | n | signed ≥ +0.56 | signed ≥ +0.88 | mean signed | mean BERTScore |",
        "|--------------------|---|----------------|----------------|-------------|----------------|",
    ]
    for ms in model_slugs:
        key_sig = f"nli_signed_gold→gen_{ms}"
        key_bs = f"bertscore_{ms}"
        vs = [r[key_sig] for r in rows if key_sig in r]
        bs = [r[key_bs] for r in rows if key_bs in r]
        if not vs:
            continue
        above = sum(1 for v in vs if v >= 0.56)
        cons = sum(1 for v in vs if v >= 0.88)
        lines.append(
            f"| {ms:<18} | {len(vs):>3} | {above:>3}/{len(vs)} ({100*above/len(vs):.1f}%) | "
            f"{cons:>3}/{len(vs)} ({100*cons/len(vs):.1f}%) | "
            f"{statistics.mean(vs):+.3f} | {statistics.mean(bs):+.3f} |"
        )

    lines.append("")
    lines.append("## Diff ↔ X: does the source-of-truth entail each reference or candidate?")
    lines.append("")
    lines.append("| probe                    | n | signed ≥ +0.56 | mean signed | mean entail |")
    lines.append("|--------------------------|---|----------------|-------------|-------------|")
    # gold pillar
    vs_gold = [r["nli_signed_diff→gold"] for r in rows if "nli_signed_diff→gold" in r]
    ent_gold = [r["nli_entail_diff→gold"] for r in rows if "nli_entail_diff→gold" in r]
    above = sum(1 for v in vs_gold if v >= 0.56)
    lines.append(
        f"| diff → gold              | {len(vs_gold):>3} | {above}/{len(vs_gold)} ({100*above/len(vs_gold):.1f}%) | "
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
        lines.append(
            f"| diff → gen ({ms:<17}) | {len(vs):>3} | {above}/{len(vs)} ({100*above/len(vs):.1f}%) | "
            f"{statistics.mean(vs):+.3f} | {statistics.mean(ent):+.3f} |"
        )
    lines.append("")
    lines.append("## Interpretation cheatsheet")
    lines.append("")
    lines.append("- If `diff→gold` also fails (signed near 0 or <<0.56): **reference itself is not diff-entailing** → construct-validity critique is empirical. Gold = author *intent*, not diff *content*.")
    lines.append("- If `diff→gold` passes but `diff→gen` fails across all models: **models underfit content**, gold is fine, reference framework salvageable.")
    lines.append("- If `diff→gold` and `diff→gen` both pass: NLI works on both — iter10's failure is a **calibration-shift** issue between paired and free-form regimes.")
    return "\n".join(lines) + "\n"


def main() -> None:
    t_start = time.time()
    rows = load_iter10_samples()
    print(f"loaded {len(rows)} iter10 samples (with CodeLlama generations)", flush=True)
    diff_map = load_diffs_for_samples(rows)
    print(f"mapped diffs for {len(diff_map)} (lang, diff_id) pairs", flush=True)

    for slug, path in MODELS.items():
        generate_for_model(slug, path, rows, diff_map)

    # Score all three (reuse iter10 codellama generations; compute everything fresh
    # for a single consistent metric implementation).
    iter10_score_block(rows, diff_map)
    for slug in MODELS:
        score_block(rows, slug, diff_map)
    score_gold_diff_once(rows, diff_map)

    # Flat CSV: one row per sample, wide columns per model
    all_keys = set()
    for r in rows:
        all_keys.update(r.keys())
    fixed = ["diff_id", "language", "gold",
             "gen_codellama-7b", "gen_qwen2.5-coder-7b", "gen_deepseek-6.7b"]
    others = sorted(k for k in all_keys if k not in fixed)
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
