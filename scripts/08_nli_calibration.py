"""Iter 8 — NLI probe calibration.

Treat the probe as a binary classifier:
  positive = meaning-preserving candidate (syn)
  negative = meaning-changing candidate (ant for verbs, dis for nouns)

Reuses iter 5b (verb) + iter 7 (noun) NLI scores — no re-inference needed.

Outputs:
  runs/iter08_calibration/operating_points.csv
  runs/iter08_calibration/per_kind.csv
  runs/iter08_calibration/per_language.csv
  runs/iter08_calibration/summary.md
  runs/iter08_calibration/roc.png
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "runs"
OUT = RUNS / "iter08_calibration"
OUT.mkdir(parents=True, exist_ok=True)


def load_candidates() -> list[dict]:
    """Flatten iter 5b + iter 7 CSVs into per-candidate rows."""
    rows: list[dict] = []

    verb_csv = RUNS / "iter05b_nli" / "results.csv"
    with verb_csv.open() as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "diff_id": r["diff_id"], "language": r["language"],
                "anchor": r["anchor"], "kind": "verb",
                "candidate_type": "syn", "label": 1,
                "entail": float(r["entail_syn"]),
                "signed": float(r["signed_syn"]),
                "partner": r["synonym"],
            })
            rows.append({
                "diff_id": r["diff_id"], "language": r["language"],
                "anchor": r["anchor"], "kind": "verb",
                "candidate_type": "ant", "label": 0,
                "entail": float(r["entail_ant"]),
                "signed": float(r["signed_ant"]),
                "partner": r["antonym"],
            })

    noun_csv = RUNS / "iter07_noun" / "results.csv"
    with noun_csv.open() as fh:
        for r in csv.DictReader(fh):
            rows.append({
                "diff_id": r["diff_id"], "language": r["language"],
                "anchor": r["anchor"], "kind": f"noun_{r['kind']}",
                "candidate_type": "syn", "label": 1,
                "entail": float(r["entail_syn"]),
                "signed": float(r["signed_syn"]),
                "partner": r["synonym"],
            })
            rows.append({
                "diff_id": r["diff_id"], "language": r["language"],
                "anchor": r["anchor"], "kind": f"noun_{r['kind']}",
                "candidate_type": "dis", "label": 0,
                "entail": float(r["entail_dis"]),
                "signed": float(r["signed_dis"]),
                "partner": r["disjoint"],
            })
    return rows


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Mann-Whitney U formulation of AUC — handles ties, no sklearn."""
    pos = [s for s, y in zip(scores, labels) if y == 1]
    neg = [s for s, y in zip(scores, labels) if y == 0]
    if not pos or not neg:
        return float("nan")
    # For each pos,neg pair: +1 if pos>neg, +0.5 if equal, +0 if pos<neg
    wins = 0.0
    for p in pos:
        for n in neg:
            if p > n: wins += 1.0
            elif p == n: wins += 0.5
    return wins / (len(pos) * len(neg))


def confusion_at(scores: list[float], labels: list[int], tau: float) -> dict:
    tp = fp = fn = tn = 0
    for s, y in zip(scores, labels):
        pred = 1 if s >= tau else 0
        if pred == 1 and y == 1: tp += 1
        elif pred == 1 and y == 0: fp += 1
        elif pred == 0 and y == 1: fn += 1
        else: tn += 1
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    tpr = rec
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    acc = (tp + tn) / n if n else 0.0
    return {
        "tau": tau, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": prec, "recall": rec, "f1": f1,
        "tpr": tpr, "fpr": fpr, "accuracy": acc,
    }


def sweep(scores: list[float], labels: list[int], grid: list[float]) -> list[dict]:
    return [confusion_at(scores, labels, t) for t in grid]


def best_f1(sweep_rows: list[dict]) -> dict:
    return max(sweep_rows, key=lambda r: r["f1"])


def threshold_at_fpr(sweep_rows: list[dict], fpr_max: float) -> dict | None:
    """Pick highest-TPR threshold whose FPR ≤ fpr_max."""
    eligible = [r for r in sweep_rows if r["fpr"] <= fpr_max]
    if not eligible:
        return None
    return max(eligible, key=lambda r: r["tpr"])


def write_op_csv(rows: list[dict], path: Path, extra_cols: list[str] = None) -> None:
    extra_cols = extra_cols or []
    cols = extra_cols + [
        "score_kind", "tau", "tp", "fp", "fn", "tn",
        "precision", "recall", "f1", "tpr", "fpr", "accuracy",
    ]
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main() -> None:
    rows = load_candidates()
    print(f"loaded {len(rows)} candidates (pos + neg) from iter 5b + iter 7")

    # Two score kinds
    score_kinds = {
        "entail": (
            [r["entail"] for r in rows],
            [r["label"] for r in rows],
            [round(x, 3) for x in [i / 100 for i in range(0, 101)]],
        ),
        "signed": (
            [r["signed"] for r in rows],
            [r["label"] for r in rows],
            [round(x, 3) for x in [-1 + i / 50 for i in range(0, 101)]],
        ),
    }

    # Overall operating points
    op_rows: list[dict] = []
    overall_auc: dict[str, float] = {}
    overall_best: dict[str, dict] = {}
    overall_fpr5: dict[str, dict | None] = {}

    for name, (scores, labels, grid) in score_kinds.items():
        sw = sweep(scores, labels, grid)
        for r in sw:
            r["score_kind"] = name
            op_rows.append(r)
        overall_auc[name] = roc_auc(scores, labels)
        overall_best[name] = best_f1(sw)
        overall_fpr5[name] = threshold_at_fpr(sw, 0.05)

    write_op_csv(op_rows, OUT / "operating_points.csv")

    # Per kind
    kinds = sorted({r["kind"] for r in rows})
    per_kind_rows: list[dict] = []
    per_kind_summary: list[dict] = []
    for k in kinds:
        sub = [r for r in rows if r["kind"] == k]
        n_pos = sum(1 for r in sub if r["label"] == 1)
        n_neg = sum(1 for r in sub if r["label"] == 0)
        for name, (_, _, grid) in score_kinds.items():
            scores = [r[name] for r in sub]
            labels = [r["label"] for r in sub]
            sw = sweep(scores, labels, grid)
            best = best_f1(sw)
            fpr5 = threshold_at_fpr(sw, 0.05)
            auc = roc_auc(scores, labels)
            per_kind_summary.append({
                "kind": k, "n_pos": n_pos, "n_neg": n_neg,
                "score_kind": name, "auc": auc,
                "best_f1_tau": best["tau"], "best_f1": best["f1"],
                "best_f1_precision": best["precision"], "best_f1_recall": best["recall"],
                "fpr5_tau": fpr5["tau"] if fpr5 else "",
                "fpr5_tpr": fpr5["tpr"] if fpr5 else "",
            })
            for r in sw:
                r2 = dict(r); r2["score_kind"] = name; r2["kind"] = k
                per_kind_rows.append(r2)

    write_op_csv(per_kind_rows, OUT / "per_kind_operating_points.csv", extra_cols=["kind"])

    with (OUT / "per_kind.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "kind", "n_pos", "n_neg", "score_kind", "auc",
            "best_f1_tau", "best_f1", "best_f1_precision", "best_f1_recall",
            "fpr5_tau", "fpr5_tpr",
        ])
        w.writeheader()
        w.writerows(per_kind_summary)

    # Per language
    langs = sorted({r["language"] for r in rows})
    per_lang_summary = []
    for lang in langs:
        sub = [r for r in rows if r["language"] == lang]
        n_pos = sum(1 for r in sub if r["label"] == 1)
        n_neg = sum(1 for r in sub if r["label"] == 0)
        for name in ("entail", "signed"):
            scores = [r[name] for r in sub]
            labels = [r["label"] for r in sub]
            auc = roc_auc(scores, labels)
            # Use overall recommended threshold to check generalization
            tau = overall_best[name]["tau"]
            conf = confusion_at(scores, labels, tau)
            per_lang_summary.append({
                "language": lang, "n_pos": n_pos, "n_neg": n_neg,
                "score_kind": name, "auc": auc,
                "tau_used": tau, "f1_at_tau": conf["f1"],
                "precision_at_tau": conf["precision"],
                "recall_at_tau": conf["recall"],
            })

    with (OUT / "per_language.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=[
            "language", "n_pos", "n_neg", "score_kind", "auc",
            "tau_used", "f1_at_tau", "precision_at_tau", "recall_at_tau",
        ])
        w.writeheader()
        w.writerows(per_lang_summary)

    # Summary markdown
    n_pos = sum(1 for r in rows if r["label"] == 1)
    n_neg = sum(1 for r in rows if r["label"] == 0)
    lines = [
        "# Iter 8 — NLI probe calibration",
        "",
        f"- Candidates: **{len(rows)}** ({n_pos} meaning-preserving + {n_neg} meaning-changing)",
        "- Sources: iter 5b (verb pairs, 1630×2) + iter 7 (noun pairs, 1277×2)",
        "- Task: binary classifier — flag candidate as 'faithful to gold'",
        "",
        "## Overall operating points",
        "",
        "| score   | AUC  | best-F1 τ | best F1 | prec  | recall | FPR≤0.05 τ | TPR@FPR0.05 |",
        "|---------|-----:|----------:|--------:|------:|-------:|-----------:|------------:|",
    ]
    for name in ("entail", "signed"):
        b = overall_best[name]
        f = overall_fpr5[name]
        lines.append(
            f"| {name:<7} | {overall_auc[name]:.4f} | {b['tau']:+.3f} | {b['f1']:.4f} | "
            f"{b['precision']:.4f} | {b['recall']:.4f} | "
            f"{f['tau']:+.3f} | {f['tpr']:.4f} |" if f else
            f"| {name:<7} | {overall_auc[name]:.4f} | {b['tau']:+.3f} | {b['f1']:.4f} | "
            f"{b['precision']:.4f} | {b['recall']:.4f} | —     |   —    |"
        )
    lines.append("")

    lines.append("## Per-kind AUC and best operating point (signed score)")
    lines.append("")
    lines.append("| kind | n+ | n- | AUC | best-F1 τ | best F1 | prec | recall |")
    lines.append("|------|---:|---:|----:|----------:|--------:|-----:|-------:|")
    for s in per_kind_summary:
        if s["score_kind"] != "signed": continue
        lines.append(
            f"| {s['kind']} | {s['n_pos']} | {s['n_neg']} | {s['auc']:.4f} | "
            f"{s['best_f1_tau']:+.3f} | {s['best_f1']:.4f} | "
            f"{s['best_f1_precision']:.4f} | {s['best_f1_recall']:.4f} |"
        )
    lines.append("")

    lines.append("## Per-language F1 at overall best-F1 threshold (signed score)")
    lines.append("")
    tau_overall = overall_best["signed"]["tau"]
    lines.append(f"Threshold used: τ = {tau_overall:+.3f}")
    lines.append("")
    lines.append("| language | n+ | n- | AUC | F1@τ | prec | recall |")
    lines.append("|----------|---:|---:|----:|-----:|-----:|-------:|")
    for s in per_lang_summary:
        if s["score_kind"] != "signed": continue
        lines.append(
            f"| {s['language']} | {s['n_pos']} | {s['n_neg']} | {s['auc']:.4f} | "
            f"{s['f1_at_tau']:.4f} | {s['precision_at_tau']:.4f} | {s['recall_at_tau']:.4f} |"
        )
    lines.append("")

    (OUT / "summary.md").write_text("\n".join(lines) + "\n")

    # ROC-like figure (text approximation if no matplotlib)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(11, 5))
        for ax, name in zip(axes, ("entail", "signed")):
            sw = [r for r in op_rows if r["score_kind"] == name]
            sw_sorted = sorted(sw, key=lambda r: r["fpr"])
            xs = [r["fpr"] for r in sw_sorted]
            ys = [r["tpr"] for r in sw_sorted]
            ax.plot(xs, ys, "-", lw=2, color="#1f77b4")
            ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.5)
            b = overall_best[name]
            ax.scatter([b["fpr"]], [b["tpr"]], color="red", zorder=5,
                       label=f"best F1={b['f1']:.3f}\nτ={b['tau']:+.2f}")
            ax.set_xlabel("FPR (false-faithful rate)")
            ax.set_ylabel("TPR (true-faithful rate)")
            ax.set_title(f"ROC — NLI {name} score\nAUC={overall_auc[name]:.4f}")
            ax.legend(loc="lower right", fontsize=9)
            ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
            ax.grid(alpha=0.3)
        fig.suptitle("NLI probe: meaning-preserving vs meaning-changing", fontsize=12)
        fig.tight_layout()
        fig.savefig(OUT / "roc.png", dpi=140)
        print(f"wrote {OUT / 'roc.png'}")
    except Exception as e:
        print(f"(matplotlib skipped: {e})")

    print(f"wrote {OUT / 'operating_points.csv'}")
    print(f"wrote {OUT / 'per_kind.csv'}")
    print(f"wrote {OUT / 'per_language.csv'}")
    print(f"wrote {OUT / 'summary.md'}")
    print()
    print("\n".join(lines[:40]))


if __name__ == "__main__":
    main()
