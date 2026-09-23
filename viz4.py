#!/usr/bin/env python3
"""Render experiment-4 charts for LinkedIn — now a FOUR-way comparison.

Two non-generative "System 1" models head to head (Jev, the commercial one; Laya,
the open one), plus a LoRA-fine-tuned Qwen2.5-1.5B risk gate served on-device, plus
Claude Sonnet 4.6. Charts:
  09 — twin accuracy, four arms, bare vs enriched   (who beats the coin flip once state is given?)
  10 — false-negative rate on traps, four arms       (who waves catastrophes through?)
  11 — confidence on the impossible bare calls        (who bluffs when the answer isn't in the text?)

Palette: Jev #2a78d6, Laya #b8497e (rose), SLM #6b3fa0 (violet), Sonnet #eb6834.
The quartet passes the skill's CVD checks (adjacent-pair OKLab ΔE>=8 across
deuter/protan/tritan, normal-vision ΔE>=15) — see tools/cvd_check.py.
Condition (bare vs enriched) is encoded by POSITION, never by color.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SUM = json.load(open(os.path.join(HERE, "results", "summary4.json")))
VIZ = os.path.join(HERE, "viz")
os.makedirs(VIZ, exist_ok=True)

JEV = "#2a78d6"
LAYA = "#b8497e"
SLM = "#6b3fa0"
SONNET = "#eb6834"
SURFACE = "#fcfcfb"
PAGE = "#f9f9f7"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
GOOD = "#0ca30c"
CRIT = "#d03b3b"

# order keeps the two System-1 models (Jev, Laya) adjacent for direct comparison
ARMS = [("jev", "Jev — commercial System One", JEV),
        ("laya", "Laya — open System One (typed-decisions, un-refit)", LAYA),
        ("slm", "Fine-tuned Qwen2.5-1.5B (LoRA, on-device)", SLM),
        ("sonnet", "Claude Sonnet 4.6 (LLM)", SONNET)]

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": PAGE, "axes.facecolor": SURFACE, "savefig.facecolor": PAGE,
    "text.color": INK, "axes.edgecolor": BASELINE, "axes.labelcolor": INK2,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlecolor": INK,
})

FOOT = ("n = 20 twin pairs + 16 trap pairs (hand-labeled, held out of fine-tuning)  ·  directional pilot, not a benchmark  ·  "
        "Jev-1.13 (Backboard)  ·  Laya typed-decisions, out-of-the-box, no temperature refit  ·  "
        "Qwen2.5-1.5B LoRA on R-Judge + synthetic twins (MLX, on-device)  ·  Claude Sonnet 4.6 (Azure Foundry)")


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def err(vals, cis):
    lo = [max(0.0, v - (c[0] * 100 if c else v)) for v, c in zip(vals, cis)]
    hi = [max(0.0, (c[1] * 100 if c else v) - v) for v, c in zip(vals, cis)]
    return [lo, hi]


def grouped(setname, metric, title, subtitle, ylabel, fname, annotate=None, ymax=112,
            legend_loc="upper left", legend_anchor=(0.0, 0.995)):
    """metric in {'acc','fn_rate'}; two condition groups x four arms."""
    fig, ax = plt.subplots(figsize=(12.2, 7.0))
    style_ax(ax)
    conds = ["bare", "enriched"]
    xlabels = ["BARE\n(command text only)", "ENRICHED\n(command + environment state)"]
    x = [0, 1]
    w = 0.19
    offs = [-1.5 * w, -0.5 * w, 0.5 * w, 1.5 * w]
    for (arm, lab, col), off in zip(ARMS, offs):
        vals = [SUM[setname][c][metric][arm] * 100 for c in conds]
        cis = [SUM[setname][c]["ci95"].get(f"{arm}_{'acc' if metric=='acc' else 'fn'}") for c in conds]
        bars = ax.bar([xi + off for xi in x], vals, w, color=col, edgecolor=SURFACE,
                      linewidth=2, zorder=3, label=lab, yerr=err(vals, cis),
                      error_kw=dict(ecolor=INK2, elinewidth=1.2, capsize=3, capthick=1.2))
        for b, v, c in zip(bars, vals, cis):
            top = max(v, (c[1] * 100 if c else v))
            ax.text(b.get_x() + b.get_width() / 2, top + 1.8, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=9.5, color=INK, weight="bold")
    if metric == "acc":
        ax.axhline(50, color=CRIT, lw=1.3, ls=(0, (5, 3)), zorder=2)
        ax.text(1.52, 51.5, "coin flip", color=CRIT, fontsize=9, weight="bold", ha="right")
    style_ax(ax)
    ax.set_xlim(-0.6, 1.75)
    ax.set_ylim(0, ymax)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=10.5, color=INK2)
    ax.set_ylabel(ylabel, fontsize=10.5, color=INK2)
    if annotate:
        annotate(ax)
    ax.legend(loc=legend_loc, frameon=False, fontsize=9.5, bbox_to_anchor=legend_anchor,
              ncol=1, handlelength=1.1, labelspacing=0.35)
    fig.suptitle(title, x=0.5, y=0.99, fontsize=15.5, weight="bold")
    ax.set_title(subtitle, fontsize=10.5, color=MUTED, pad=10)
    fig.text(0.5, -0.02, FOOT, ha="center", fontsize=7.2, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(os.path.join(VIZ, fname), dpi=200, bbox_inches="tight")
    plt.close(fig)


def chart_confidence_bare():
    """On the bare calls the answer isn't in the text. How sure was each arm anyway?

    This is the one axis where Laya wins: it stays near the 50% truth line instead
    of bluffing. The three others answer with 87-95% confidence and are right half
    the time.
    """
    fig, ax = plt.subplots(figsize=(11.4, 7.0))
    style_ax(ax)
    mc = SUM["twins"]["bare"]["mean_conf_bare"]
    arms = [("jev", "Jev", JEV), ("laya", "Laya\n(open)", LAYA),
            ("slm", "Fine-tuned\nQwen-1.5B", SLM), ("sonnet", "Claude\nSonnet 4.6", SONNET)]
    x = list(range(len(arms)))
    conf = [mc[a] * 100 for a, _, _ in arms]
    cols = [c for _, _, c in arms]
    bars = ax.bar(x, conf, 0.58, color=cols, edgecolor=SURFACE, linewidth=2, zorder=3)
    for b, v in zip(bars, conf):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}%", ha="center",
                va="bottom", fontsize=12, color=INK, weight="bold")
    ax.axhline(50, color=CRIT, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.text(len(arms) + 0.32, 50, "actual accuracy\non these calls\n≈ 50% (coin flip)",
            color=CRIT, fontsize=9.5, weight="bold", ha="right", va="center",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=2))
    for xi, v in zip(x, conf):
        over = v - 50
        col = GOOD if over <= 12 else CRIT
        ax.annotate("", xy=(xi + 0.36, 50), xytext=(xi + 0.36, v),
                    arrowprops=dict(arrowstyle="<->", color=col, lw=1.5))
        ax.text(xi + 0.41, (v + 50) / 2, f"+{over:.0f} pt", color=col, fontsize=9.5,
                weight="bold", va="center", ha="left")
    style_ax(ax)
    ax.set_xlim(-0.5, len(arms) + 0.4)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([n for _, n, _ in arms], fontsize=11, color=INK2, weight="bold")
    ax.set_ylabel("mean confidence on bare twins  (the answer is provably not in the text)",
                  fontsize=10.5, color=INK2)
    fig.suptitle("Only the open model won't bluff when it's blind",
                 x=0.5, y=0.99, fontsize=16, weight="bold")
    ax.set_title("Three of four answer at 87–93% on calls whose answer isn't in the text, and are right half the time. "
                 "The catch: the same near-0.5 hedging left Laya near-chance even WITH the state — humble everywhere, decisive nowhere",
                 fontsize=9.8, color=MUTED, pad=10)
    fig.text(0.5, -0.02, FOOT, ha="center", fontsize=7.2, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(os.path.join(VIZ, "11_slm_overconfidence.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    grouped("twins", "acc",
            "Two System-1 models, a fine-tuned SLM, and an LLM — who beats the coin flip?",
            "20 twin pairs · byte-identical command, only the environment flips the label · bars = 95% CI (clustered bootstrap; Wilson at 0/100%)",
            "accuracy  (higher is better)",
            "09_slm_twin_accuracy.png")
    grouped("traps", "fn_rate",
            "Who waves catastrophes through?",
            "16 trap pairs · benign-looking commands that are catastrophic given hidden state · bars = 95% CI (clustered bootstrap; Wilson at 0/100%)",
            "false-negative rate — catastrophes approved  (lower is safer)",
            "10_slm_false_negatives.png", ymax=116,
            legend_loc="upper right", legend_anchor=(1.0, 0.995))
    chart_confidence_bare()
    print("wrote:", ", ".join(sorted(f for f in os.listdir(VIZ) if f.startswith(("09", "10", "11")))))


if __name__ == "__main__":
    main()
