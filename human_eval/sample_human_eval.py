#!/usr/bin/env python3
"""Stratified sampler for the diff-description accuracy human audit.

Builds long-form (diff, message) pairs from the scaled-triad results, joins the
raw diff text and the intent-marker flag, stratifies into 8 cells
(source x NLI pass/fail x intent), samples 10 per cell, and emits:

  out/rater.csv   BLIND task file (diff + message only; rater fills label/reason)
  out/key.csv     hidden ground-truth map (source/model/scores/cell) -- DO NOT show rater
  out/retest.csv  BLIND re-rate subset for intra-rater reliability

Stdlib only. Deterministic given --seed.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict, Counter

MODELS = ["codellama-7b", "qwen2.5-coder-7b", "deepseek-6.7b"]
TAU = 0.56
LANGS = ["cpp", "cs", "go", "java", "js", "php", "py", "rust"]

# resolve paths relative to this script's parent (the "Untitled Folder" workspace)
HERE = os.path.dirname(os.path.abspath(__file__))
WORK = os.path.dirname(HERE)


def p(*parts):
    return os.path.join(WORK, *parts)


def clamp(x, lo=0.0, hi=1.0):
    return max(lo, min(hi, x))


def readable_diff(d):
    """MCMD stores diffs tokenized with literal <nl> markers. Restore line
    structure (so +/- diff markers are visible per line) without touching the
    intra-line spacing -- information is unchanged, only newlines restored."""
    return d.replace(" <nl> ", "\n").replace("<nl>", "\n")


def tokenize(text):
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


def containment(message, diff):
    """Fraction of message content-tokens that also appear in the diff token set."""
    mtok = set(tokenize(message))
    if not mtok:
        return 0.0
    dtok = set(tokenize(diff))
    return sum(1 for t in mtok if t in dtok) / len(mtok)


def argmax_class(entail, signed):
    contra = clamp(entail - signed)
    neutral = clamp(1.0 - entail - contra)
    triple = {"entailment": entail, "neutral": neutral, "contradiction": contra}
    return max(triple, key=triple.get)


def load_diffs(data_dir, cap):
    """(language, diff_id) -> diff text (capped)."""
    diffs = {}
    for lang in LANGS:
        fp = os.path.join(data_dir, f"{lang}.jsonl")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                o = json.loads(line)
                did = str(o["diff_id"]).strip()
                d = o["diff"]
                truncated = len(d) > cap
                diffs[(lang, did)] = (d[:cap], truncated)
    return diffs


def load_intent(intent_csv):
    m = {}
    with open(intent_csv) as f:
        for row in csv.DictReader(f):
            key = (row["language"].strip(), str(row["diff_id"]).strip())
            m[key] = int(row["has_intent"])
    return m


def fcol(row, name):
    v = row.get(name, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def build_pairs(results_csv, diffs, intent):
    pairs = []
    missing_diff = 0
    with open(results_csv) as f:
        for row in csv.DictReader(f):
            lang = row["language"].strip()
            did = str(row["diff_id"]).strip()
            key = (lang, did)
            if key not in diffs:
                missing_diff += 1
                continue
            diff_text, truncated = diffs[key]
            has_intent = intent.get(key, 0)

            def add(source, model, message, signed, entail):
                if message is None or signed is None or entail is None:
                    return
                message = message.strip()
                if not message:
                    return
                pairs.append({
                    "language": lang,
                    "diff_id": did,
                    "source": source,
                    "model": model,
                    "message": message,
                    "diff": diff_text,
                    "diff_truncated": int(truncated),
                    "signed": signed,
                    "entail": entail,
                    "nli_pass": int(signed >= TAU),
                    "nli_argmax": argmax_class(entail, signed),
                    "has_intent": has_intent,
                    "lexical_overlap": round(containment(message, diff_text), 4),
                })

            add("gold", "", row.get("gold"),
                fcol(row, "nli_signed_diff→gold"), fcol(row, "nli_entail_diff→gold"))
            for mdl in MODELS:
                add("gen", mdl, row.get(f"gen_{mdl}"),
                    fcol(row, f"nli_signed_diff→gen_{mdl}"),
                    fcol(row, f"nli_entail_diff→gen_{mdl}"))
    return pairs, missing_diff


def cell_of(pr):
    s = "gold" if pr["source"] == "gold" else "gen"
    d = "pass" if pr["nli_pass"] else "fail"
    i = "intent" if pr["has_intent"] else "plain"
    return f"{s}|{d}|{i}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=p("Results", "runs", "iter13_scaled_triad", "results.csv"))
    ap.add_argument("--intent", default=p("Results", "runs", "iter16_intent_classifier", "classified.csv"))
    ap.add_argument("--data-dir", default=p("IST_paper_core_2026-06-04.repair_staging", "data", "raw"))
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--retest", type=int, default=15)
    # Match the BART-MNLI diff premise used in iter13_scaled_triad.py.
    ap.add_argument("--diff-cap", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    for label, path in [("results", args.results), ("intent", args.intent)]:
        if not os.path.exists(path):
            sys.exit(f"ERROR: {label} file not found: {path}")
    if not os.path.isdir(args.data_dir):
        sys.exit(f"ERROR: data dir not found: {args.data_dir}")

    rng = random.Random(args.seed)
    diffs = load_diffs(args.data_dir, args.diff_cap)
    intent = load_intent(args.intent)
    pairs, missing_diff = build_pairs(args.results, diffs, intent)

    print(f"built {len(pairs)} (diff,message) pairs  (diffs loaded={len(diffs)}, "
          f"rows missing diff={missing_diff})")

    by_cell = defaultdict(list)
    for pr in pairs:
        by_cell[cell_of(pr)].append(pr)

    cells = [f"{s}|{d}|{i}" for s in ("gold", "gen") for d in ("pass", "fail")
             for i in ("intent", "plain")]
    print("\npopulation per cell:")
    for c in cells:
        print(f"  {c:22s} {len(by_cell[c]):5d}")

    sample = []
    for c in cells:
        pool = by_cell[c][:]
        rng.shuffle(pool)
        take = pool[:args.per_cell]
        if len(take) < args.per_cell:
            print(f"  WARNING: cell {c} short: {len(take)}/{args.per_cell}")
        for pr in take:
            pr["cell"] = c
        sample.extend(take)

    rng.shuffle(sample)
    for idx, pr in enumerate(sample, 1):
        pr["sample_id"] = f"S{idx:03d}"

    os.makedirs(args.out, exist_ok=True)

    # BLIND rater file
    with open(os.path.join(args.out, "rater.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "language", "diff", "message",
                    "rater_label", "rater_reason", "rater_notes"])
        for pr in sample:
            w.writerow([pr["sample_id"], pr["language"], readable_diff(pr["diff"]),
                        pr["message"], "", "", ""])

    # hidden ground-truth key
    keycols = ["sample_id", "language", "diff_id", "source", "model", "signed", "entail",
               "nli_pass", "nli_argmax", "has_intent", "lexical_overlap", "diff_truncated", "cell"]
    with open(os.path.join(args.out, "key.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keycols)
        w.writeheader()
        for pr in sample:
            w.writerow({k: pr[k] for k in keycols})

    # cell population vs sampled -> post-stratification weights for population estimates
    sampled_per_cell = Counter(pr["cell"] for pr in sample)
    with open(os.path.join(args.out, "cells.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "population", "sampled"])
        for c in cells:
            w.writerow([c, len(by_cell[c]), sampled_per_cell.get(c, 0)])

    # retest subset (same items, blind, shuffled, mapped in key)
    retest_n = min(args.retest, len(sample))
    retest = rng.sample(sample, retest_n)
    rng.shuffle(retest)
    with open(os.path.join(args.out, "retest.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["retest_id", "orig_sample_id", "language", "diff", "message",
                    "rater_label", "rater_reason", "rater_notes"])
        for j, pr in enumerate(retest, 1):
            w.writerow([f"R{j:02d}", pr["sample_id"], pr["language"], readable_diff(pr["diff"]),
                        pr["message"], "", "", ""])

    print(f"\nwrote {len(sample)} main + {retest_n} retest to {args.out}/")
    print("  rater.csv  (BLIND - give to rater)")
    print("  retest.csv (BLIND - give to rater AFTER rater.csv, separate session)")
    print("  key.csv    (HIDDEN ground truth - do NOT show rater)")
    print("\nsource mix:", dict(Counter(pr["source"] for pr in sample)))


if __name__ == "__main__":
    main()
