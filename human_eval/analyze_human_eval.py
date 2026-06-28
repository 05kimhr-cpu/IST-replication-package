#!/usr/bin/env python3
"""Analyze the completed diff-description accuracy human audit.

Reads out/rater.csv (filled) + out/key.csv (+ optional out/retest.csv filled)
and computes:
  - primary human supported-rate for gold vs generated messages
  - supported/neutral/contradicted label distribution
  - rater_reason distribution overall and by source
  - human binary_supported vs NLI_pass agreement + Cohen's kappa
  - 3-way human label vs NLI argmax class + kappa
  - per-cell supported-rate and NLI agreement
  - intra-rater test-retest kappa when retest.csv is filled

Writes out/agreement.md and machine-readable CSVs. Stdlib only.
"""
import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
LABELS3 = ["supported", "neutral", "contradicted"]
ARGMAX_TO_3 = {"entailment": "supported", "neutral": "neutral", "contradiction": "contradicted"}
REASONS_BY_LABEL = {
    "supported": {"exact_core", "partial_but_sufficient", "high_level_but_correct"},
    "neutral": {
        "too_abstract", "missing_key_detail", "near_miss", "external_context",
        "diff_insufficient", "ambiguous_vague", "other_neutral",
    },
    "contradicted": {
        "opposite_direction", "wrong_entity", "wrong_action",
        "different_change", "other_contradicted",
    },
}


def cohen_kappa(pairs, classes):
    """pairs: list of (a,b). Returns Cohen's kappa over the given class set."""
    n = len(pairs)
    if n == 0:
        return float("nan")
    idx = {c: i for i, c in enumerate(classes)}
    k = len(classes)
    conf = [[0] * k for _ in range(k)]
    for a, b in pairs:
        if a not in idx or b not in idx:
            return float("nan")
        conf[idx[a]][idx[b]] += 1
    po = sum(conf[i][i] for i in range(k)) / n
    ra = [sum(conf[i]) / n for i in range(k)]
    cb = [sum(conf[i][j] for i in range(k)) / n for j in range(k)]
    pe = sum(ra[i] * cb[i] for i in range(k))
    if abs(1 - pe) < 1e-12:
        return float("nan")
    return (po - pe) / (1 - pe)


def load_rater(path, id_col="sample_id"):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            lab = (row.get("rater_label") or "").strip().lower()
            reason = (row.get("rater_reason") or "").strip().lower()
            out[row[id_col]] = {
                "label": lab,
                "reason": reason,
                "notes": row.get("rater_notes", ""),
                "orig": row.get("orig_sample_id", ""),
            }
    return out


def load_key(path):
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            out[row["sample_id"]] = row
    return out


def fmt(x):
    return "nan" if x != x else f"{x:.3f}"


def pct(x):
    return "nan" if x != x else f"{x * 100:.1f}%"


def validate_ratings(rater):
    invalid = []
    rated = []
    pending = []
    for sid, row in rater.items():
        lab = row["label"]
        reason = row["reason"]
        if not lab:
            pending.append(sid)
            continue
        if lab not in LABELS3:
            invalid.append(f"{sid}: invalid rater_label={lab!r}")
            continue
        if reason not in REASONS_BY_LABEL[lab]:
            invalid.append(f"{sid}: reason {reason!r} not allowed for label {lab!r}")
            continue
        rated.append((sid, row))
    if invalid:
        msg = "\n".join(invalid[:30])
        more = "" if len(invalid) <= 30 else f"\n... {len(invalid) - 30} more"
        sys.exit(f"Invalid ratings:\n{msg}{more}")
    return rated, pending


def load_weights(out_dir):
    weight = {}
    cells_path = os.path.join(out_dir, "cells.csv")
    if os.path.exists(cells_path):
        with open(cells_path, newline="") as f:
            for c in csv.DictReader(f):
                sampled = int(c["sampled"])
                weight[c["cell"]] = (int(c["population"]) / sampled) if sampled else 0.0
    return weight


def rate(rows, src, field):
    sub = [x for x in rows if x["source"] == src]
    if not sub:
        return float("nan"), 0
    return sum(x[field] for x in sub) / len(sub), len(sub)


def weighted_rate(rows, src, field):
    sub = [x for x in rows if x["source"] == src]
    wsum = sum(x["w"] for x in sub)
    if not sub or wsum == 0:
        return float("nan")
    return sum(x[field] * x["w"] for x in sub) / wsum


def count_by(rows, *fields):
    c = Counter()
    for row in rows:
        c[tuple(row[f] for f in fields)] += 1
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "out"))
    args = ap.parse_args()
    o = args.out

    rater = load_rater(os.path.join(o, "rater.csv"))
    key = load_key(os.path.join(o, "key.csv"))
    rated, pending = validate_ratings(rater)
    if not rated:
        sys.exit("No rows rated yet. Fill rater_label and rater_reason in out/rater.csv, then re-run.")

    weight = load_weights(o)
    rows = []
    for sid, r in rated:
        k = key[sid]
        rows.append({
            "sample_id": sid,
            "human3": r["label"],
            "reason": r["reason"],
            "human_bin": 1 if r["label"] == "supported" else 0,
            "nli_pass": int(k["nli_pass"]),
            "nli3": ARGMAX_TO_3[k["nli_argmax"]],
            "source": k["source"],
            "cell": k["cell"],
            "overlap": float(k["lexical_overlap"]),
            "w": weight.get(k["cell"], 1.0),
        })

    n = len(rows)
    agree_bin = sum(1 for x in rows if x["human_bin"] == x["nli_pass"]) / n
    kappa_bin = cohen_kappa(
        [("S" if x["human_bin"] else "N", "S" if x["nli_pass"] else "N") for x in rows],
        ["S", "N"],
    )
    kappa3 = cohen_kappa([(x["human3"], x["nli3"]) for x in rows], LABELS3)

    conf3 = defaultdict(int)
    for x in rows:
        conf3[(x["human3"], x["nli3"])] += 1

    label_counts = count_by(rows, "source", "human3")
    reason_counts = count_by(rows, "source", "human3", "reason")
    overall_labels = Counter(x["human3"] for x in rows)
    overall_reasons = Counter((x["human3"], x["reason"]) for x in rows)

    h_gold_raw, n_gold = rate(rows, "gold", "human_bin")
    h_gen_raw, n_gen = rate(rows, "gen", "human_bin")
    nli_gold_raw, _ = rate(rows, "gold", "nli_pass")
    nli_gen_raw, _ = rate(rows, "gen", "nli_pass")
    h_gold_w = weighted_rate(rows, "gold", "human_bin")
    h_gen_w = weighted_rate(rows, "gen", "human_bin")
    nli_gold_w = weighted_rate(rows, "gold", "nli_pass")
    nli_gen_w = weighted_rate(rows, "gen", "nli_pass")

    cell_stats = defaultdict(lambda: {"n": 0, "supported": 0, "nli_agree": 0})
    for x in rows:
        st = cell_stats[x["cell"]]
        st["n"] += 1
        st["supported"] += x["human_bin"]
        st["nli_agree"] += int(x["human_bin"] == x["nli_pass"])

    intra_txt = "Not available (out/retest.csv absent or unrated)."
    kappa_re3 = kappa_rebin = float("nan")
    rt_path = os.path.join(o, "retest.csv")
    if os.path.exists(rt_path):
        rt = load_rater(rt_path, id_col="retest_id")
        valid_rt, _ = validate_ratings(rt)
        re_pairs3, re_pairsb = [], []
        for _, rr in valid_rt:
            orig = rr["orig"]
            if orig in rater and rater[orig]["label"] in LABELS3:
                a, b = rater[orig]["label"], rr["label"]
                re_pairs3.append((a, b))
                re_pairsb.append(("S" if a == "supported" else "N", "S" if b == "supported" else "N"))
        if re_pairs3:
            kappa_re3 = cohen_kappa(re_pairs3, LABELS3)
            kappa_rebin = cohen_kappa(re_pairsb, ["S", "N"])
            intra_txt = (
                f"{len(re_pairs3)} items re-rated. 3-way kappa={fmt(kappa_re3)}, "
                f"binary kappa={fmt(kappa_rebin)}."
            )

    L = []
    L.append("# Diff-description accuracy audit - analysis\n")
    L.append(f"- Rated: **{n}** / {len(rater)} main items"
             + (f" ({len(pending)} pending)" if pending else "") + "\n")
    L.append("- Rater: single independent blind human. Treat as a calibration audit, "
             "not a multi-rater gold standard.\n")
    L.append("- Human input: the shown diff is capped to the same 1,500-character premise "
             "used by the BART-MNLI diff diagnostic.\n")

    L.append("\n## 1. Primary outcome: human supported-rate\n\n")
    L.append("| source | supported-rate weighted | supported-rate raw | n raw | NLI pass-rate weighted |\n")
    L.append("|---|--:|--:|--:|--:|\n")
    L.append(f"| diff->gold | {pct(h_gold_w)} | {pct(h_gold_raw)} | {n_gold} | {pct(nli_gold_w)} |\n")
    L.append(f"| diff->generated | {pct(h_gen_w)} | {pct(h_gen_raw)} | {n_gen} | {pct(nli_gen_w)} |\n")
    L.append("\nWeighted rates use `cells.csv` post-stratification. Raw rates are diagnostic because "
             "the sample is intentionally balanced across source/NLI/intent cells.\n")

    L.append("\n## 2. Label distribution\n\n")
    L.append("| source | supported | neutral | contradicted | n |\n|---|--:|--:|--:|--:|\n")
    for src in ["gold", "gen"]:
        total = sum(label_counts[(src, lab)] for lab in LABELS3)
        L.append(f"| {src} | " + " | ".join(str(label_counts[(src, lab)]) for lab in LABELS3)
                 + f" | {total} |\n")
    L.append("| overall | " + " | ".join(str(overall_labels[lab]) for lab in LABELS3)
             + f" | {n} |\n")

    L.append("\n## 3. Reason distribution\n\n")
    L.append("| source | label | reason | n |\n|---|---|---|--:|\n")
    for (src, lab, reason), cnt in sorted(reason_counts.items()):
        L.append(f"| {src} | {lab} | {reason} | {cnt} |\n")

    L.append("\n## 4. Human supported vs NLI pass @ tau=+0.56\n")
    L.append(f"- Raw binary agreement: **{agree_bin*100:.1f}%**\n")
    L.append(f"- Cohen's kappa (binary): **{fmt(kappa_bin)}**\n")
    L.append(f"- 3-class Cohen's kappa: **{fmt(kappa3)}**\n\n")
    L.append("| human \\ NLI | " + " | ".join(LABELS3) + " |\n")
    L.append("|---|" + "---|" * len(LABELS3) + "\n")
    for h in LABELS3:
        L.append(f"| {h} | " + " | ".join(str(conf3[(h, c)]) for c in LABELS3) + " |\n")

    L.append("\n## 5. Per-cell supported-rate and NLI agreement\n\n")
    L.append("| cell | n | human supported-rate | NLI agreement |\n|---|--:|--:|--:|\n")
    for c in sorted(cell_stats):
        st = cell_stats[c]
        L.append(f"| {c} | {st['n']} | {st['supported']/st['n']*100:.1f}% | "
                 f"{st['nli_agree']/st['n']*100:.1f}% |\n")

    L.append("\n## 6. Intra-rater test-retest reliability\n")
    L.append(f"- {intra_txt}\n")

    with open(os.path.join(o, "agreement.md"), "w") as f:
        f.write("".join(L))

    with open(os.path.join(o, "per_cell.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "n", "human_supported_rate", "nli_agreement_rate"])
        for c in sorted(cell_stats):
            st = cell_stats[c]
            w.writerow([c, st["n"], round(st["supported"] / st["n"], 4),
                        round(st["nli_agree"] / st["n"], 4)])

    with open(os.path.join(o, "label_distribution.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "label", "n"])
        for src in ["gold", "gen"]:
            for lab in LABELS3:
                w.writerow([src, lab, label_counts[(src, lab)]])
        for lab in LABELS3:
            w.writerow(["overall", lab, overall_labels[lab]])

    with open(os.path.join(o, "reason_distribution.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source", "label", "reason", "n"])
        for (src, lab, reason), cnt in sorted(reason_counts.items()):
            w.writerow([src, lab, reason, cnt])
        for (lab, reason), cnt in sorted(overall_reasons.items()):
            w.writerow(["overall", lab, reason, cnt])

    with open(os.path.join(o, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["metric", "value"])
        for kname, v in [
            ("n_rated", n),
            ("human_supported_gold_raw", round(h_gold_raw, 4) if h_gold_raw == h_gold_raw else "nan"),
            ("human_supported_gen_raw", round(h_gen_raw, 4) if h_gen_raw == h_gen_raw else "nan"),
            ("human_supported_gold_weighted", round(h_gold_w, 4) if h_gold_w == h_gold_w else "nan"),
            ("human_supported_gen_weighted", round(h_gen_w, 4) if h_gen_w == h_gen_w else "nan"),
            ("nli_pass_gold_weighted", round(nli_gold_w, 4) if nli_gold_w == nli_gold_w else "nan"),
            ("nli_pass_gen_weighted", round(nli_gen_w, 4) if nli_gen_w == nli_gen_w else "nan"),
            ("agreement_binary", round(agree_bin, 4)),
            ("kappa_binary", round(kappa_bin, 4) if kappa_bin == kappa_bin else "nan"),
            ("kappa_3way", round(kappa3, 4) if kappa3 == kappa3 else "nan"),
            ("kappa_intra_3way", round(kappa_re3, 4) if kappa_re3 == kappa_re3 else "nan"),
            ("kappa_intra_binary", round(kappa_rebin, 4) if kappa_rebin == kappa_rebin else "nan"),
        ]:
            w.writerow([kname, v])

    print(f"wrote {o}/agreement.md, summary.csv, per_cell.csv, "
          f"label_distribution.csv, reason_distribution.csv  (n_rated={n})")
    print(f"  human supported weighted: gold={fmt(h_gold_w)} gen={fmt(h_gen_w)}")
    print(f"  NLI agreement={agree_bin*100:.1f}%  kappa_bin={fmt(kappa_bin)}  kappa_3way={fmt(kappa3)}")


if __name__ == "__main__":
    main()
