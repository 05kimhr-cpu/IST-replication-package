"""Iter 9 — paper plots.

Three figures assembled from existing CSV outputs (no new experiments):
  fig1_delta_panel.png   — 2x3 grid: delta distributions per metric
  fig2_sign_bars.png     — syn > ant/dis fraction, one bar per metric
  fig3_nli_per_kind.png  — NLI AUC + F1 per perturbation kind
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
RUNS = ROOT / "runs"
OUT = RUNS / "iter09_plots"
OUT.mkdir(parents=True, exist_ok=True)


def load_verb_deltas() -> dict[str, list[float]]:
    """Merge iter 3 (bleu/rougeL/chrf/meteor) + iter 4 (bertscore) + iter 5b
    (entail/signed). All keyed by (diff_id, language, anchor)."""
    key = lambda r: (r["diff_id"], r["language"], r["anchor"])
    out: dict[tuple, dict] = {}
    with (RUNS / "iter03_paired_all" / "results.csv").open() as fh:
        for r in csv.DictReader(fh):
            out[key(r)] = {
                "bleu_delta": float(r["bleu_delta"]),
                "rougeL_delta": float(r["rougeL_delta"]),
                "chrf_delta": float(r["chrf_delta"]),
                "meteor_delta": float(r["meteor_delta"]),
            }
    with (RUNS / "iter04_bertscore" / "results.csv").open() as fh:
        for r in csv.DictReader(fh):
            k = key(r)
            if k in out:
                out[k]["bertscore_delta"] = float(r["bertscore_delta"])
    with (RUNS / "iter05b_nli" / "results.csv").open() as fh:
        for r in csv.DictReader(fh):
            k = key(r)
            if k in out:
                out[k]["signed_delta"] = float(r["signed_delta"])
                out[k]["entail_delta"] = float(r["entail_delta"])

    metrics = ["bleu_delta", "rougeL_delta", "chrf_delta", "meteor_delta",
               "bertscore_delta", "entail_delta", "signed_delta"]
    bundled = {m: [] for m in metrics}
    for row in out.values():
        if not all(m in row for m in metrics):
            continue
        for m in metrics:
            bundled[m].append(row[m])
    return bundled


def load_noun_deltas() -> dict[str, list[float]]:
    metrics = ["bleu_delta", "rougeL_delta", "chrf_delta", "meteor_delta",
               "bertscore_delta", "entail_delta", "signed_delta"]
    bundled = {m: [] for m in metrics}
    with (RUNS / "iter07_noun" / "results.csv").open() as fh:
        for r in csv.DictReader(fh):
            for m in metrics:
                bundled[m].append(float(r[m]))
    return bundled


def fig1_delta_panel(verbs: dict, nouns: dict) -> None:
    order = [
        ("bleu_delta", "BLEU-4", (-0.3, 0.3)),
        ("rougeL_delta", "ROUGE-L", (-0.3, 0.3)),
        ("chrf_delta", "CHRF++", (-0.2, 0.2)),
        ("meteor_delta", "METEOR", (-0.3, 0.3)),
        ("bertscore_delta", "BERTScore F1", (-0.1, 0.1)),
        ("signed_delta", "NLI signed (entail−contra)", (-2.0, 2.0)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5))
    axes = axes.flatten()
    for ax, (key, title, xlim) in zip(axes, order):
        v = np.array(verbs[key]); n = np.array(nouns[key])
        bins = 40
        ax.hist(v, bins=bins, range=xlim, alpha=0.55, color="#1f77b4",
                label=f"verbs (n={len(v)})")
        ax.hist(n, bins=bins, range=xlim, alpha=0.55, color="#ff7f0e",
                label=f"nouns (n={len(n)})")
        ax.axvline(0, color="black", lw=0.8, alpha=0.5)
        ax.set_title(title, fontsize=11)
        ax.set_xlim(*xlim)
        ax.set_xlabel(f"delta (syn − ant/dis)")
        ax.set_ylabel("pairs")
        ax.legend(fontsize=8, loc="upper left")
        ax.grid(alpha=0.25)
    fig.suptitle(
        "Per-pair score deltas (syn candidate − meaning-changing candidate)\n"
        "Standard metrics cluster at zero; NLI signed delta separates the two classes.",
        fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    path = OUT / "fig1_delta_panel.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def fig2_sign_bars(verbs: dict, nouns: dict) -> None:
    metrics = [
        ("bleu_delta", "BLEU-4"),
        ("rougeL_delta", "ROUGE-L"),
        ("chrf_delta", "CHRF++"),
        ("meteor_delta", "METEOR"),
        ("bertscore_delta", "BERTScore"),
        ("entail_delta", "NLI entail"),
        ("signed_delta", "NLI signed"),
    ]

    def frac_win(xs: list[float]) -> float:
        xs = np.array(xs)
        if len(xs) == 0: return float("nan")
        return float((xs > 0).mean())

    verb_fracs = [frac_win(verbs[m]) for m, _ in metrics]
    noun_fracs = [frac_win(nouns[m]) for m, _ in metrics]

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(metrics))
    w = 0.38
    ax.bar(x - w/2, verb_fracs, width=w, label="verbs", color="#1f77b4")
    ax.bar(x + w/2, noun_fracs, width=w, label="nouns", color="#ff7f0e")
    ax.axhline(0.5, color="red", ls="--", lw=1, alpha=0.7, label="chance")
    ax.set_xticks(x)
    ax.set_xticklabels([t for _, t in metrics], rotation=25, ha="right")
    ax.set_ylabel("fraction of pairs with syn > meaning-changing")
    ax.set_title("Sign agreement: does each metric prefer the meaning-preserving candidate?")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.25, axis="y")
    for i, (vf, nf) in enumerate(zip(verb_fracs, noun_fracs)):
        ax.text(i - w/2, vf + 0.01, f"{vf:.2f}", ha="center", fontsize=8)
        ax.text(i + w/2, nf + 0.01, f"{nf:.2f}", ha="center", fontsize=8)
    ax.legend(loc="upper left")
    fig.tight_layout()
    path = OUT / "fig2_sign_bars.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def fig3_nli_per_kind() -> None:
    with (RUNS / "iter08_calibration" / "per_kind.csv").open() as fh:
        rows = [r for r in csv.DictReader(fh) if r["score_kind"] == "signed"]
    order = ["verb", "noun_directional", "noun_infrastructure",
             "noun_close_entity", "noun_diagnostic"]
    rows = sorted(rows, key=lambda r: order.index(r["kind"]) if r["kind"] in order else 999)

    kinds = [r["kind"] for r in rows]
    aucs = [float(r["auc"]) for r in rows]
    f1s = [float(r["best_f1"]) for r in rows]
    ns = [int(r["n_pos"]) for r in rows]

    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    x = np.arange(len(kinds))
    w = 0.4
    ax.bar(x - w/2, aucs, width=w, label="AUC", color="#2ca02c")
    ax.bar(x + w/2, f1s, width=w, label="best F1", color="#d62728")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}\n(n={n})" for k, n in zip(kinds, ns)],
                       fontsize=9)
    ax.axhline(0.5, color="gray", ls="--", lw=1, alpha=0.5, label="chance")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("score")
    ax.set_title("NLI probe discrimination by perturbation kind")
    ax.grid(alpha=0.25, axis="y")
    for i, (a, f) in enumerate(zip(aucs, f1s)):
        ax.text(i - w/2, a + 0.01, f"{a:.3f}", ha="center", fontsize=8)
        ax.text(i + w/2, f + 0.01, f"{f:.3f}", ha="center", fontsize=8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = OUT / "fig3_nli_per_kind.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"wrote {path}")


def main() -> None:
    verbs = load_verb_deltas()
    nouns = load_noun_deltas()
    print(f"verbs: {len(verbs['bleu_delta'])} pairs; nouns: {len(nouns['bleu_delta'])} pairs")
    fig1_delta_panel(verbs, nouns)
    fig2_sign_bars(verbs, nouns)
    fig3_nli_per_kind()


if __name__ == "__main__":
    main()
