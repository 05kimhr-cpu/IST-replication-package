"""Iter 17 — Build paper figures fig4, fig5, fig6.

fig4: Gold → Generated NLI pass-rate across 3 models × 2 NLI backbones
       (submission-grade: demonstrates 7–10% gold→gen is not a BART artefact).
fig5: Diff → {gold, gen×3} pass-rate (§6 core visual).
fig6: Per-language diff → gold pass-rate (§6 appendix).

Sources:
  - runs/iter13_scaled_triad/results.csv  (BART signed scores)
  - runs/iter14_deberta_sanity/results.csv  (DeBERTa signed scores)

Outputs:
  - runs/figures/fig4_gold_gen_pass.pdf / .png
  - runs/figures/fig5_diff_x_pass.pdf / .png
  - runs/figures/fig6_diff_gold_per_lang.pdf / .png
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
I13 = ROOT / "runs" / "iter13_scaled_triad" / "results.csv"
I14 = ROOT / "runs" / "iter14_deberta_sanity" / "results.csv"
FIG = ROOT / "runs" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TAU = 0.56
MODELS = ["codellama-7b", "qwen2.5-coder-7b", "deepseek-6.7b"]
MODEL_LABELS = ["CodeLlama-7B", "Qwen2.5-Coder-7B", "DeepSeek-Coder-6.7B"]
LANGS = ["cpp", "cs", "go", "java", "js", "php", "py", "rust"]
LANG_LABELS = ["C++", "C#", "Go", "Java", "JS", "PHP", "Python", "Rust"]


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    halfw = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, centre - halfw), min(1.0, centre + halfw)


def load(path: Path) -> list[dict]:
    with path.open() as fh:
        return list(csv.DictReader(fh))


def pass_rate(rows: list[dict], col: str) -> tuple[float, float, float, int, int]:
    vs = [float(r[col]) for r in rows if r.get(col) not in (None, "")]
    n = len(vs)
    k = sum(1 for v in vs if v >= TAU)
    p = k / n if n else 0.0
    lo, hi = wilson(k, n)
    return p, lo, hi, k, n


def fig4() -> None:
    r13 = load(I13)
    r14 = load(I14)
    bart = [pass_rate(r13, f"nli_signed_gold→gen_{m}") for m in MODELS]
    deb = [pass_rate(r14, f"deb_nli_signed_gold→gen_{m}") for m in MODELS]

    x = np.arange(len(MODELS))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.0, 3.6))
    bart_p = [b[0] * 100 for b in bart]
    bart_err_lo = [(b[0] - b[1]) * 100 for b in bart]
    bart_err_hi = [(b[2] - b[0]) * 100 for b in bart]
    deb_p = [d[0] * 100 for d in deb]
    deb_err_lo = [(d[0] - d[1]) * 100 for d in deb]
    deb_err_hi = [(d[2] - d[0]) * 100 for d in deb]

    ax.bar(x - w / 2, bart_p, w, yerr=[bart_err_lo, bart_err_hi],
           label="BART-MNLI", color="#4C72B0", capsize=3, edgecolor="black", linewidth=0.5)
    ax.bar(x + w / 2, deb_p, w, yerr=[deb_err_lo, deb_err_hi],
           label="DeBERTa-v3-MNLI", color="#DD8452", capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(MODEL_LABELS, fontsize=9)
    ax.set_ylabel("Gold → Generated pass rate (%)", fontsize=10)
    ax.set_title(r"Gold → Generated NLI pass rate at $\tau = +0.56$ (N=1600)", fontsize=10)
    ax.axhline(50, ls=":", color="grey", lw=0.7)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 60)
    for xi, p, (lo, hi) in zip(x - w / 2, bart_p, [(b[1], b[2]) for b in bart]):
        ax.text(xi, p + 1, f"{p:.1f}%", ha="center", fontsize=8)
    for xi, p, (lo, hi) in zip(x + w / 2, deb_p, [(d[1], d[2]) for d in deb]):
        ax.text(xi, p + 1, f"{p:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIG / f"fig4_gold_gen_pass.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG/'fig4_gold_gen_pass.pdf'}")


def fig5() -> None:
    r13 = load(I13)
    r14 = load(I14)

    probes = [
        ("diff → gold", "nli_signed_diff→gold", "deb_nli_signed_diff→gold"),
        ("diff → CodeLlama", f"nli_signed_diff→gen_codellama-7b", f"deb_nli_signed_diff→gen_codellama-7b"),
        ("diff → Qwen", f"nli_signed_diff→gen_qwen2.5-coder-7b", f"deb_nli_signed_diff→gen_qwen2.5-coder-7b"),
        ("diff → DeepSeek", f"nli_signed_diff→gen_deepseek-6.7b", f"deb_nli_signed_diff→gen_deepseek-6.7b"),
    ]

    bart = [pass_rate(r13, p[1]) for p in probes]
    deb = [pass_rate(r14, p[2]) for p in probes]

    x = np.arange(len(probes))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    bart_p = [b[0] * 100 for b in bart]
    bart_err_lo = [(b[0] - b[1]) * 100 for b in bart]
    bart_err_hi = [(b[2] - b[0]) * 100 for b in bart]
    deb_p = [d[0] * 100 for d in deb]
    deb_err_lo = [(d[0] - d[1]) * 100 for d in deb]
    deb_err_hi = [(d[2] - d[0]) * 100 for d in deb]

    colors = ["#C44E52", "#4C72B0", "#4C72B0", "#4C72B0"]
    colors2 = ["#8B2E35", "#2E4F78", "#2E4F78", "#2E4F78"]
    ax.bar(x - w / 2, bart_p, w, yerr=[bart_err_lo, bart_err_hi],
           color=colors, capsize=3, edgecolor="black", linewidth=0.5, label="BART-MNLI")
    ax.bar(x + w / 2, deb_p, w, yerr=[deb_err_lo, deb_err_hi],
           color=colors2, capsize=3, edgecolor="black", linewidth=0.5, label="DeBERTa-v3-MNLI")
    ax.set_xticks(x)
    ax.set_xticklabels([p[0] for p in probes], fontsize=9)
    ax.set_ylabel(r"Diff → X pass rate (%) at $\tau=+0.56$", fontsize=10)
    ax.set_title("Source-of-truth entailment: generations > gold (N=1600)", fontsize=10)
    ax.axvspan(-0.5, 0.5, alpha=0.08, color="red")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 70)
    for xi, p in zip(x - w / 2, bart_p):
        ax.text(xi, p + 1, f"{p:.1f}%", ha="center", fontsize=8)
    for xi, p in zip(x + w / 2, deb_p):
        ax.text(xi, p + 1, f"{p:.1f}%", ha="center", fontsize=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIG / f"fig5_diff_x_pass.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG/'fig5_diff_x_pass.pdf'}")


def fig6() -> None:
    r13 = load(I13)
    per = {}
    for lang in LANGS:
        rs = [r for r in r13 if r["language"] == lang]
        per[lang] = pass_rate(rs, "nli_signed_diff→gold")

    x = np.arange(len(LANGS))
    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    ps = [per[l][0] * 100 for l in LANGS]
    err_lo = [(per[l][0] - per[l][1]) * 100 for l in LANGS]
    err_hi = [(per[l][2] - per[l][0]) * 100 for l in LANGS]
    bars = ax.bar(x, ps, yerr=[err_lo, err_hi], color="#C44E52",
                  capsize=3, edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(LANG_LABELS, fontsize=9)
    ax.set_ylabel(r"diff → gold pass rate (%) at $\tau=+0.56$", fontsize=10)
    ax.set_title("Diff-faithfulness of gold messages varies by language (N=200/lang)", fontsize=10)
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, 40)
    for xi, p in zip(x, ps):
        ax.text(xi, p + 1, f"{p:.0f}%", ha="center", fontsize=8)
    plt.tight_layout()
    for ext in ("pdf", "png"):
        plt.savefig(FIG / f"fig6_diff_gold_per_lang.{ext}", dpi=200, bbox_inches="tight")
    plt.close()
    print(f"wrote {FIG/'fig6_diff_gold_per_lang.pdf'}")


if __name__ == "__main__":
    fig4()
    fig5()
    fig6()
    print("ALL FIGURES DONE")
