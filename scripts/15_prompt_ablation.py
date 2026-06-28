"""Iter 15 — Prompt-sensitivity ablation.

Does the iter 10/11/13 finding (low gold→gen pass, mid diff→gen pass)
depend on the prompt we gave the model? Re-generate the same 1600
iter 13 diffs with a *diff-oriented* prompt and a *intent-oriented*
prompt using CodeLlama-7B-Instruct, then re-score both probes
(gold→gen, diff→gen) with BART-MNLI.

If the diff-oriented prompt does NOT materially raise diff→gen
pass-rates, the paired-NLI non-transfer is prompt-invariant (reviewer
cannot blame prompt engineering). If it does raise, paper must
caveat.

We only re-generate CodeLlama (one model suffices for prompt-sensitivity;
all three behaved identically in iter 13).

Artefacts:
  - runs/iter15_prompt_ablation/generations_{prompt_tag}.jsonl
  - runs/iter15_prompt_ablation/summary.md
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
from cmg_ist.nli import batch_nli  # noqa: E402

import os  # noqa: E402

_LOCAL = os.environ.get("CMG_IST_MODELS_DIR")
MODEL_PATH = (f"{_LOCAL}/CodeLlama/7B-Instruct" if _LOCAL
              else "codellama/CodeLlama-7b-Instruct-hf")
ITER13_CSV = ROOT / "runs" / "iter13_scaled_triad" / "results.csv"
OUT_DIR = ROOT / "runs" / "iter15_prompt_ablation"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MAX_NEW_TOKENS = 40
MAX_DIFF_CHARS = 6000

PROMPTS = {
    "intent":   # "paraphrase the author's intent"
        "[INST] Write a single-line git commit message that captures the "
        "author's intent for the following diff. Be concise (under 20 words). "
        "Output only the commit message.\n\nDiff:\n{diff}\n[/INST]",
    "content":  # "describe exactly what the diff does"
        "[INST] Describe, in a single line under 20 words, ONLY what the "
        "following diff changes at the code level. Do not guess motivation "
        "or intent. Output only the description.\n\nDiff:\n{diff}\n[/INST]",
    "baseline":  # same as iter 13
        "[INST] Given the following diff, write a single-line git commit "
        "message (under 20 words). Output only the commit message — no "
        "explanation.\n\nDiff:\n{diff}\n[/INST]",
}


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    halfw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, centre - halfw), min(1.0, centre + halfw)


def load_samples_from_iter13() -> list[dict]:
    rows: list[dict] = []
    with ITER13_CSV.open() as fh:
        for r in csv.DictReader(fh):
            rows.append({"diff_id": r["diff_id"], "language": r["language"], "gold": r["gold"]})
    # map diffs from raw
    needed = {(r["language"], r["diff_id"]) for r in rows}
    diff_map: dict[str, str] = {}
    for lang in sorted({r["language"] for r in rows}):
        for s in load_samples(lang, limit=None):
            k = (lang, str(s["diff_id"]))
            if k in needed:
                diff_map[f"{lang}:{s['diff_id']}"] = (s.get("diff") or "")[:MAX_DIFF_CHARS]
    for r in rows:
        r["diff"] = diff_map[f"{r['language']}:{r['diff_id']}"]
    return rows


def generate_all(rows: list[dict]) -> None:
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"loading {MODEL_PATH} ...", flush=True)
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, device_map={"": 0}, trust_remote_code=True
    ).eval()
    print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

    for ptag, template in PROMPTS.items():
        print(f"[prompt={ptag}] generating ...", flush=True)
        key = f"gen_{ptag}"
        for i, r in enumerate(rows):
            prompt = template.format(diff=r["diff"])
            inp = tok(prompt, return_tensors="pt", truncation=True, max_length=3072).to(mdl.device)
            with torch.no_grad():
                out = mdl.generate(**inp, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, pad_token_id=tok.pad_token_id)
            decoded = tok.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=True)
            first = next((ln.strip() for ln in decoded.splitlines() if ln.strip()), "")
            r[key] = first
            if (i + 1) % 200 == 0:
                print(f"  [{ptag}] {i + 1}/{len(rows)}", flush=True)

        with (OUT_DIR / f"generations_{ptag}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps({
                    "diff_id": r["diff_id"], "language": r["language"],
                    "gold": r["gold"], "generated": r[key],
                }) + "\n")
        print(f"[prompt={ptag}] wrote generations", flush=True)

    del mdl
    torch.cuda.empty_cache()


def score_all(rows: list[dict]) -> None:
    refs = [r["gold"] for r in rows]
    diffs = [r["diff"][:1500] for r in rows]
    for ptag in PROMPTS:
        key = f"gen_{ptag}"
        gens = [r[key] for r in rows]
        print(f"[{ptag}] gold→gen NLI ...", flush=True)
        nli = batch_nli(refs, gens, batch_size=32)
        for r, n in zip(rows, nli):
            r[f"nli_signed_gold→gen_{ptag}"] = n.entailment - n.contradiction
        print(f"[{ptag}] diff→gen NLI ...", flush=True)
        nli_d = batch_nli(diffs, gens, batch_size=16)
        for r, n in zip(rows, nli_d):
            r[f"nli_signed_diff→gen_{ptag}"] = n.entailment - n.contradiction


def summarize(rows: list[dict]) -> str:
    lines = ["# Iter 15 — Prompt-sensitivity ablation (CodeLlama-7B, N = 1600)", "",
             "Threshold τ = +0.56 (paper's BART calibration).", ""]
    lines += ["## Gold → Generated", "",
              "| prompt   | pass ≥ +0.56 (95% CI)     | mean signed |",
              "|----------|---------------------------|-------------|"]
    for ptag in PROMPTS:
        vs = [r[f"nli_signed_gold→gen_{ptag}"] for r in rows]
        above = sum(1 for v in vs if v >= 0.56)
        lo, hi = wilson_ci(above, len(vs))
        lines.append(f"| {ptag:<8} | {above}/{len(vs)} ({100*above/len(vs):.1f}%)  [{100*lo:.1f}–{100*hi:.1f}] | {statistics.mean(vs):+.3f} |")
    lines += ["", "## Diff → Generated", "",
              "| prompt   | pass ≥ +0.56 (95% CI)     | mean signed |",
              "|----------|---------------------------|-------------|"]
    for ptag in PROMPTS:
        vs = [r[f"nli_signed_diff→gen_{ptag}"] for r in rows]
        above = sum(1 for v in vs if v >= 0.56)
        lo, hi = wilson_ci(above, len(vs))
        lines.append(f"| {ptag:<8} | {above}/{len(vs)} ({100*above/len(vs):.1f}%)  [{100*lo:.1f}–{100*hi:.1f}] | {statistics.mean(vs):+.3f} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    t_start = time.time()
    rows = load_samples_from_iter13()
    print(f"loaded {len(rows)} iter13 samples", flush=True)
    generate_all(rows)
    score_all(rows)

    cols = ["diff_id", "language"]
    for ptag in PROMPTS:
        cols += [f"gen_{ptag}", f"nli_signed_gold→gen_{ptag}", f"nli_signed_diff→gen_{ptag}"]
    with (OUT_DIR / "results.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {OUT_DIR/'results.csv'}", flush=True)

    summary = summarize(rows)
    (OUT_DIR / "summary.md").write_text(summary)
    print(summary)
    print(f"total elapsed: {time.time() - t_start:.1f}s")


if __name__ == "__main__":
    main()
