"""
Generate all 4 figures (B&W) and 4 tables (.tex) for the IST paper.
Run with: python generate_figures.py
No GPU required — pure matplotlib + LaTeX text output.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np
from pathlib import Path

OUT = Path(__file__).parent
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ─────────────────────────────────────────────────────────────────────────────
# Fig. 1  Evidence-funnel pipeline overview  (§3)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig1():
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")

    def box(cx, cy, w, h, text, style="round,pad=0.15", lw=1.2, hatch=None):
        fc = "white"
        rect = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle=style, linewidth=lw,
            edgecolor="black", facecolor=fc, hatch=hatch, zorder=3)
        ax.add_patch(rect)
        ax.text(cx, cy, text, ha="center", va="center",
                fontsize=8.5, wrap=True, zorder=4,
                multialignment="center")

    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.1),
                    zorder=2)

    # ── Column labels ─────────────────────────────────────────────────────
    for x, label in [(1.3, "Data"), (4.15, "Evaluation stage"), (7.8, "Construct")]:
        ax.text(x, 5.7, label, ha="center", va="center",
                fontsize=9, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc="0.88", ec="black", lw=0.7))

    # ── Data boxes ────────────────────────────────────────────────────────
    box(1.3, 4.5, 2.1, 0.75, "MCMD corpus\n4,000 gold messages\n8 languages")
    box(1.3, 3.0, 2.1, 0.75, "MCMD corpus\n1,600 diffs\n8 languages")
    box(1.3, 1.5, 2.1, 0.75, "3 × 7B code LLMs\n(1,600 generations)")

    # ── Stage boxes ────────────────────────────────────────────────────────
    box(4.15, 4.5, 2.6, 0.85,
        "RQ1 · Controlled perturbation\n2,907 paired comparisons\n5 metrics + NLI probe",
        hatch="///")
    box(4.15, 3.0, 2.6, 0.85,
        "RQ2 · NLI calibration transfer\ngold → generated pass-rate\n(τ = +0.56)",
        hatch="...")
    box(4.15, 1.5, 2.6, 0.85,
        "RQ3 · Diff-grounded eval\ndiff → gold vs diff → generated\n(BART-MNLI + DeBERTa)",
        hatch="xxx")

    box(4.15, 0.35, 2.6, 0.55,
        "RQ4 · Robustness: language / backbone / prompt / intent-marker")

    # ── Construct boxes ────────────────────────────────────────────────────
    box(7.8, 4.5, 2.0, 0.75, "Metric blindness\nto meaning flips")
    box(7.8, 3.0, 2.0, 0.75, "NLI threshold\nnon-transferable")
    box(7.8, 1.5, 2.0, 0.75, "Gold = intent plane\nGen = content plane")

    # ── Arrows ────────────────────────────────────────────────────────────
    # Data → Stage
    arrow(2.35, 4.5,  2.84, 4.5)
    arrow(2.35, 3.0,  2.84, 3.0)
    arrow(2.35, 1.5,  2.84, 1.5)
    # Stage → Construct
    arrow(5.44, 4.5,  6.8,  4.5)
    arrow(5.44, 3.0,  6.8,  3.0)
    arrow(5.44, 1.5,  6.8,  1.5)
    # Stage 3 also feeds RQ4
    arrow(4.15, 1.07, 4.15, 0.625)

    fig.savefig(OUT / "fig1_evidence_funnel.pdf")
    fig.savefig(OUT / "fig1_evidence_funnel.png")
    plt.close(fig)
    print("fig1 done")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 2  Metric sign-preference bars  (§4.1, RQ1)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig2():
    metrics = ["BLEU-4", "ROUGE-L", "CHRF++", "METEOR", "BERTScore", "NLI probe"]
    verb    = [0.00,     0.00,      0.54,     0.52,     0.506,        0.999]
    noun    = [0.00,     0.00,      0.90,     0.01,     0.738,        0.940]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.6, 3.0))

    bars_v = ax.bar(x - width / 2, verb, width, label="Verb pairs",
                    color="white", edgecolor="black", linewidth=0.9, hatch="///")
    bars_n = ax.bar(x + width / 2, noun, width, label="Noun pairs",
                    color="0.72", edgecolor="black", linewidth=0.9)

    # 50 % chance line
    ax.axhline(0.50, color="black", linestyle="--", linewidth=0.8, zorder=0)
    ax.text(5.65, 0.51, "chance", fontsize=7.5, va="bottom")

    ax.set_ylabel("Fraction of pairs: synonym preferred")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=20, ha="right")
    ax.set_ylim(0, 1.09)
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.legend(loc="upper left", frameon=True)

    # Annotate NLI bars
    ax.text(x[-1] - width / 2, verb[-1] + 0.02, "99.9%",
            ha="center", va="bottom", fontsize=7.5)
    ax.text(x[-1] + width / 2, noun[-1] + 0.02, "94.0%",
            ha="center", va="bottom", fontsize=7.5)
    # Annotate BLEU/ROUGE
    for i in range(2):
        ax.text(x[i], 0.03, "100% ties",
                ha="center", va="bottom", fontsize=6.5, rotation=90)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(OUT / "fig2_metric_delta.pdf")
    fig.savefig(OUT / "fig2_metric_delta.png")
    plt.close(fig)
    print("fig2 done")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 3  diff → gold vs diff → generated pass-rate  (§4.3, RQ3)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig3():
    labels    = ["Gold\nref.", "Code-\nLlama", "Qwen2.5-\nCoder", "DeepSeek-\nCoder"]
    bart_vals = [18.8,  37.2,  38.2,  33.7]
    deb_vals  = [26.1,  55.4,  44.2,  47.3]
    ci_bart   = [(16.9, 20.8), (34.9, 39.5), (35.9, 40.5), (31.5, 36.0)]
    ci_deb    = [(24.0, 28.3), (53.0, 57.8), (41.8, 46.6), (44.9, 49.7)]

    def err(vals, cis):
        lo = [v - c[0] for v, c in zip(vals, cis)]
        hi = [c[1] - v for v, c in zip(vals, cis)]
        return [lo, hi]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(5.6, 3.2))

    ax.bar(x - width / 2, bart_vals, width,
           yerr=err(bart_vals, ci_bart), capsize=3,
           label="BART-MNLI", color="white", edgecolor="black",
           linewidth=0.9, hatch="///",
           error_kw=dict(elinewidth=0.9, ecolor="black"))
    ax.bar(x + width / 2, deb_vals, width,
           yerr=err(deb_vals, ci_deb), capsize=3,
           label="DeBERTa-v3", color="0.68", edgecolor="black",
           linewidth=0.9,
           error_kw=dict(elinewidth=0.9, ecolor="black"))

    ax.set_ylabel("Diff-grounded NLI pass-rate (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 68)
    ax.legend(loc="upper left", frameon=True)

    # Bracket showing gold < all generated for BART
    y_bracket = 43
    ax.annotate("", xy=(x[1] - width / 2, y_bracket),
                xytext=(x[0] - width / 2, y_bracket),
                arrowprops=dict(arrowstyle="<->", color="black", lw=0.9))
    ax.text((x[0] + x[1]) / 2 - width / 2, y_bracket + 1.0,
            "18.4 pp gap", ha="center", fontsize=7.5)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(OUT / "fig3_triad_barplot.pdf")
    fig.savefig(OUT / "fig3_triad_barplot.png")
    plt.close(fig)
    print("fig3 done")


# ─────────────────────────────────────────────────────────────────────────────
# Fig. 4  Intent-match / content-match construct plane  (§5)
# ─────────────────────────────────────────────────────────────────────────────
def make_fig4():
    fig, ax = plt.subplots(figsize=(4.5, 3.8))

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Diff-faithfulness (content-match)")
    ax.set_ylabel("Reference proximity (intent-match)")

    # Shaded regions
    # Gold region: high intent-match, low diff-faithfulness
    gold_poly = plt.Polygon(
        [(0.05, 0.55), (0.05, 0.95), (0.45, 0.95), (0.45, 0.55)],
        closed=True, fc="0.82", ec="black", lw=0.9, hatch="///", zorder=2)
    ax.add_patch(gold_poly)
    ax.text(0.25, 0.75, "Gold\nmessages", ha="center", va="center",
            fontsize=8.5, fontweight="bold")

    # Generated region: lower intent-match, higher diff-faithfulness
    gen_poly = plt.Polygon(
        [(0.55, 0.10), (0.55, 0.50), (0.95, 0.50), (0.95, 0.10)],
        closed=True, fc="0.93", ec="black", lw=0.9, hatch="...", zorder=2)
    ax.add_patch(gen_poly)
    ax.text(0.75, 0.30, "Generated\nmessages", ha="center", va="center",
            fontsize=8.5, fontweight="bold")

    # Ideal region (upper-right)
    ideal_poly = plt.Polygon(
        [(0.60, 0.60), (0.60, 0.98), (0.98, 0.98), (0.98, 0.60)],
        closed=True, fc="white", ec="black", lw=1.2, linestyle="--", zorder=2)
    ax.add_patch(ideal_poly)
    ax.text(0.79, 0.79, "Ideal\n(both high)", ha="center", va="center",
            fontsize=8, style="italic")

    # Construct mismatch arrow
    ax.annotate("", xy=(0.68, 0.68), xytext=(0.32, 0.32),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=1.2),
                zorder=5)
    ax.text(0.53, 0.46, "construct\ngap", ha="center", va="center",
            fontsize=7.5, rotation=45,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none"))

    # Annotations: what metrics measure
    ax.text(0.25, 0.02, "← BLEU / ROUGE measure here →", ha="center",
            fontsize=7, style="italic", color="0.35")
    ax.annotate("", xy=(0.00, 0.35), xytext=(0.00, 0.65),
                arrowprops=dict(arrowstyle="<->", color="0.45", lw=0.8))
    ax.text(-0.04, 0.50, "NLI\nprobe", ha="center", va="center",
            fontsize=7, color="0.35", rotation=90)

    ax.set_xticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_xticklabels(["Low", "", "", "", "High"])
    ax.set_yticks([0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["Low", "", "", "", "High"])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()

    fig.savefig(OUT / "fig4_construct_plane.pdf")
    fig.savefig(OUT / "fig4_construct_plane.png")
    plt.close(fig)
    print("fig4 done")


# ─────────────────────────────────────────────────────────────────────────────
# Table 1  RQ × dataset × metric/probe mapping  (§3)
# ─────────────────────────────────────────────────────────────────────────────
def make_tab1():
    tex = r"""% Table 1 — Research Design Overview
\begin{table}[t]
\centering
\caption{Research design: RQ, data, sample size, and evaluation signal.
         MCMD = Multi-language Commit Message Dataset.
         $^\dagger$Calibrated at $\tau$\,=\,+0.56 on the paired set;
         used comparatively (not as absolute faithfulness probability) in RQ3.}
\label{tab:design}
\footnotesize
\begin{tabular}{p{0.9cm}p{4.6cm}p{1.8cm}rp{3.6cm}}
\toprule
\textbf{RQ} & \textbf{Question (short)} & \textbf{Data} & \textbf{$N$} & \textbf{Evaluation signal} \\
\midrule
RQ1 & Do reference-based metrics distinguish meaning-preserving vs.\ meaning-changing edits? & MCMD gold & 2,907 pairs & BLEU-4, ROUGE-L, CHRF++, METEOR, BERTScore, NLI (AUC) \\[4pt]
RQ2 & Does paired NLI calibration transfer to real CMG outputs (gold as premise)? & 1,600 diffs; 3 LLMs & 1,600 $\times$ 3 & gold\,$\to$\,gen pass-rate at $\tau$\,=\,+0.56$^\dagger$ \\[4pt]
RQ3 & Are gold and generated messages equally supported by the diff? & 1,600 diffs; 3 LLMs & 1,600 $\times$ 4 & diff\,$\to$\,gold vs.\ diff\,$\to$\,gen pass-rate (BART-MNLI + DeBERTa) \\[4pt]
RQ4 & How robust and explainable is the construct mismatch? & 1,600 diffs & 1,600 & Per-language, NLI backbone, prompt variant, intent-marker z-test \\
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "tab1_design.tex").write_text(tex)
    print("tab1 done")


# ─────────────────────────────────────────────────────────────────────────────
# Table 2  Metric blindness (RQ1)
# ─────────────────────────────────────────────────────────────────────────────
def make_tab2():
    tex = r"""% Table 2 — Metric Blindness on Paired Perturbation Dataset
\begin{table}[t]
\centering
\caption{Metric response to controlled meaning-direction edits ($N$\,=\,2,907 pairs,
         8 languages). \emph{Zero $\Delta$}: fraction of pairs where both
         candidates receive identical scores. \emph{Syn.\ pref.}: fraction of
         non-tied pairs where the synonym candidate scores higher. AUC is
         one-vs-rest (synonym vs.\ antonym/disjoint).
         ``---'' denotes undefined (100\% ties).}
\label{tab:blindness}
\begin{tabular}{lrrrr}
\toprule
\textbf{Metric} & \textbf{Zero $\Delta$ (\%)} & \textbf{Mean $|\Delta|$} & \textbf{Syn.\ pref.\ (\%)} & \textbf{AUC} \\
\midrule
BLEU-4          & 100.0 & 0.000 & ---  & --- \\
ROUGE-L         & 100.0 & 0.000 & ---  & --- \\
CHRF++          &  78.0 & 0.006 & 54.2 & 0.54 \\
METEOR          & $>$99 & 0.008 & 52.1 & 0.52 \\
BERTScore       &  83.0 & 0.003 & 50.6 & 0.51 \\
\midrule
NLI probe (BART-MNLI) & \phantom{0}0.1 & 0.931 & 99.9 & \textbf{0.962} \\
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "tab2_metric_blindness.tex").write_text(tex)
    print("tab2 done")


# ─────────────────────────────────────────────────────────────────────────────
# Table 3  gold→gen + diff→X pass-rates  (§4.2–4.3, RQ2+RQ3)
# ─────────────────────────────────────────────────────────────────────────────
def make_tab3():
    tex = r"""% Table 3 — NLI Pass-Rates (RQ2 and RQ3)
\begin{table}[t]
\centering
\caption{NLI pass-rates under two premises ($N$\,=\,1,600, $\tau$\,=\,+0.56).
         \emph{Gold\,$\to$\,gen}: gold message as premise, generated as hypothesis
         (RQ2). \emph{Diff\,$\to$\,X}: diff as premise (RQ3);
         lower value for gold indicates that gold messages often encode
         intent beyond literal diff content.
         95\% Wilson confidence intervals in brackets.}
\label{tab:passrate}
\begin{tabular}{llrr}
\toprule
\textbf{Backbone} & \textbf{Condition} & \textbf{Pass (\%)} & \textbf{95\% CI} \\
\midrule
\multirow{4}{*}{BART-MNLI}
  & gold\,$\to$\,CodeLlama-7B       &  9.8 & [8.6,\,11.1] \\
  & gold\,$\to$\,Qwen2.5-Coder-7B   &  7.1 & [6.1,\,8.2]  \\
  & gold\,$\to$\,DeepSeek-Coder-6.7B&  9.8 & [8.6,\,11.1] \\
\cmidrule(lr){2-4}
  & diff\,$\to$\,Gold                & 18.8 & [16.9,\,20.8] \\
  & diff\,$\to$\,CodeLlama-7B        & 37.2 & [34.9,\,39.5] \\
  & diff\,$\to$\,Qwen2.5-Coder-7B   & 38.2 & [35.9,\,40.5] \\
  & diff\,$\to$\,DeepSeek-Coder-6.7B& 33.7 & [31.5,\,36.0] \\
\midrule
\multirow{4}{*}{DeBERTa-v3}
  & diff\,$\to$\,Gold                & 26.1 & [24.0,\,28.3] \\
  & diff\,$\to$\,CodeLlama-7B        & 55.4 & [53.0,\,57.8] \\
  & diff\,$\to$\,Qwen2.5-Coder-7B   & 44.2 & [41.8,\,46.6] \\
  & diff\,$\to$\,DeepSeek-Coder-6.7B& 47.3 & [44.9,\,49.7] \\
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "tab3_passrates.tex").write_text(tex)
    print("tab3 done")


# ─────────────────────────────────────────────────────────────────────────────
# Table 4  Robustness summary  (§4.4, RQ4)
# ─────────────────────────────────────────────────────────────────────────────
def make_tab4():
    tex = r"""% Table 4 — Robustness Analyses (RQ4)
\begin{table}[t]
\centering
\caption{RQ4 robustness results. All diff\,$\to$\,gold figures use BART-MNLI
         unless labelled DeBERTa. \emph{Prompt sensitivity}: three CodeLlama-7B
         variants. \emph{Intent-marker}: z-test comparing has-marker vs.\ plain
         commits; markers include revert, hotfix, ticket-reference, CE flag,
         bracket-tag.}
\label{tab:robustness}
\small
\begin{tabular}{llrl}
\toprule
\textbf{Analysis} & \textbf{Condition / group} & \textbf{diff\,$\to$\,gold (\%)} & \textbf{Note} \\
\midrule
\multirow{8}{*}{\makecell[l]{Per-language\\(BART-MNLI)}}
  & Python     & 25.5 & highest \\
  & JavaScript & 23.5 & \\
  & C++        & 21.0 & \\
  & Rust       & 20.5 & \\
  & Java       & 17.5 & \\
  & Go         & 16.0 & \\
  & C\#        & 15.0 & \\
  & PHP        & 11.5 & lowest \\
\cmidrule(lr){1-4}
\multirow{3}{*}{\makecell[l]{Prompt\\sensitivity\\(CodeLlama)}}
  & Baseline prompt      & \multicolumn{1}{l}{diff\,$\to$\,gen 37.2\%} & gold\,$\to$\,gen 9.8\% \\
  & Intent-oriented      & \multicolumn{1}{l}{diff\,$\to$\,gen 45.8\%} & gold\,$\to$\,gen 10.3\% \\
  & Content-oriented     & \multicolumn{1}{l}{diff\,$\to$\,gen 46.4\%} & gold\,$\to$\,gen 2.1\% \\
\cmidrule(lr){1-4}
\multirow{2}{*}{\makecell[l]{Intent-marker\\confound}}
  & Has marker ($n$=159) & 11.9 & $z$\,=\,$-$2.33, $p$\,=\,0.020 \\
  & Plain commit ($n$=1,441) & 19.6 & \\
\bottomrule
\end{tabular}
\end{table}
"""
    (OUT / "tab4_robustness.tex").write_text(tex)
    print("tab4 done")


# ─────────────────────────────────────────────────────────────────────────────
# Run all
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_tab1()
    make_tab2()
    make_tab3()
    make_tab4()
    print("All done — figures and tables written to", OUT)
