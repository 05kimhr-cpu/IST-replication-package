"""Iter 10 — Real-model realism pilot.

Generate commit messages from CodeLlama-7B-Instruct on a curated pilot
slice of MCMD (8 langs × 10 commits = 80), score (gold, generated) with
the full metric stack + NLI, and compare to the iter 5b/iter 8 paired
baseline.

Question: does the NLI probe's discrimination carry over from
synthetic paired perturbations (iter 1–9) to real generated outputs?
This directly informs whether the paper's NLI recommendation
(signed >= +0.56 threshold) needs re-framing for real CMG workflows.

Artefacts:
  - runs/iter10_real_model_pilot/generations.jsonl
  - runs/iter10_real_model_pilot/results.csv
  - runs/iter10_real_model_pilot/summary.md
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
PER_LANG = 10
MIN_GOLD_TOKS = 8
MAX_DIFF_CHARS = 6000  # truncate huge diffs to fit context
import os  # noqa: E402

_LOCAL = os.environ.get("CMG_IST_MODELS_DIR")
MODEL_PATH = (f"{_LOCAL}/CodeLlama/7B-Instruct" if _LOCAL
              else "codellama/CodeLlama-7b-Instruct-hf")
MAX_NEW_TOKENS = 40

OUT_DIR = ROOT / "runs" / "iter10_real_model_pilot"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def select_pilot_samples() -> list[dict]:
    rows: list[dict] = []
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
    return rows


def build_prompt(diff: str) -> str:
    return (
        "[INST] You are an assistant that writes git commit messages. "
        "Given the following diff, write a single-line commit message "
        "(under 20 words). Do not add explanation; output only the commit message.\n\n"
        f"Diff:\n{diff}\n[/INST]"
    )


def generate_messages(rows: list[dict]) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"loading {MODEL_PATH} in fp16 ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map={"": 0}
    ).eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    for i, r in enumerate(rows):
        prompt = build_prompt(r["diff"])
        inp = tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(mdl.device)
        with torch.no_grad():
            out = mdl.generate(
                **inp,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=1.0,
                pad_token_id=tok.pad_token_id,
            )
        generated = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
        # Keep first non-empty line; real models occasionally emit multi-line.
        first_line = next((ln.strip() for ln in generated.splitlines() if ln.strip()), "")
        r["generated"] = first_line
        if (i + 1) % 10 == 0 or (i + 1) == len(rows):
            print(f"  gen {i + 1}/{len(rows)}: [{r['language']}] {first_line[:80]}", flush=True)

    # Free GPU before metric stack
    del mdl
    torch.cuda.empty_cache()


def score_metrics(rows: list[dict]) -> None:
    cands = [r["generated"] for r in rows]
    refs = [r["gold"] for r in rows]

    print("scoring bleu / rougeL / chrf / meteor ...", flush=True)
    for r in rows:
        r["bleu"] = bleu_sentence(r["generated"], r["gold"])
        r["rougeL"] = rougeL(r["generated"], r["gold"])
        r["chrf"] = chrf(r["generated"], r["gold"])
        r["meteor"] = meteor(r["generated"], r["gold"])

    print("scoring bertscore (batched) ...", flush=True)
    bs = bertscore_batch(cands, refs, model_type="roberta-large", device="cuda")
    for r, v in zip(rows, bs):
        r["bertscore_f1"] = v

    print("scoring NLI (gold -> generated) ...", flush=True)
    nli = batch_nli(refs, cands, batch_size=32)
    for r, n in zip(rows, nli):
        r["nli_entail"] = n.entailment
        r["nli_contra"] = n.contradiction
        r["nli_neutral"] = n.neutral
        r["nli_signed"] = n.entailment - n.contradiction


def summarize(rows: list[dict]) -> str:
    def agg(rs: list[dict], label: str) -> list[str]:
        if not rs:
            return [f"### {label}", "- (empty)", ""]
        keys = ["bleu", "rougeL", "chrf", "meteor", "bertscore_f1", "nli_signed", "nli_entail"]
        out = [f"### {label}  (n={len(rs)})"]
        for k in keys:
            vals = [r[k] for r in rs]
            out.append(f"- {k:14s}  mean={statistics.mean(vals):+.3f}  median={statistics.median(vals):+.3f}")
        # Threshold at paper's calibrated operating point (signed >= +0.56)
        above = sum(1 for r in rs if r["nli_signed"] >= 0.56)
        conservative = sum(1 for r in rs if r["nli_signed"] >= 0.88)
        out.append(f"- NLI signed >= +0.56 (paper balanced op point):   {above}/{len(rs)}  ({100*above/len(rs):.1f}%)")
        out.append(f"- NLI signed >= +0.88 (paper conservative op pt):  {conservative}/{len(rs)}  ({100*conservative/len(rs):.1f}%)")
        out.append("")
        return out

    lines = [
        "# Iter 10 — Real-model realism pilot (CodeLlama-7B-Instruct on MCMD)",
        "",
        f"Model: {MODEL_PATH}",
        f"Samples: {len(rows)} ({PER_LANG} per language × {len(LANGUAGES)} langs)",
        f"Gold token floor: {MIN_GOLD_TOKS}",
        "",
    ]
    lines += agg(rows, "Overall")
    lines.append("## Per-language")
    lines.append("")
    for lang in LANGUAGES:
        lines += agg([r for r in rows if r["language"] == lang], lang)
    return "\n".join(lines) + "\n"


def main() -> None:
    t_start = time.time()
    rows = select_pilot_samples()
    print(f"selected {len(rows)} pilot samples (target {PER_LANG * len(LANGUAGES)})")
    if len(rows) == 0:
        print("no samples — abort"); return

    generate_messages(rows)

    gen_path = OUT_DIR / "generations.jsonl"
    with gen_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps({k: r[k] for k in ("diff_id", "language", "gold", "generated")}) + "\n")
    print(f"wrote {gen_path}", flush=True)

    score_metrics(rows)

    fieldnames = [
        "diff_id", "language", "gold", "generated",
        "bleu", "rougeL", "chrf", "meteor", "bertscore_f1",
        "nli_entail", "nli_contra", "nli_neutral", "nli_signed",
    ]
    csv_path = OUT_DIR / "results.csv"
    with csv_path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {csv_path}", flush=True)

    summary_path = OUT_DIR / "summary.md"
    summary_path.write_text(summarize(rows))
    print(f"wrote {summary_path}", flush=True)
    print()
    print(summary_path.read_text())
    print(f"total elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
