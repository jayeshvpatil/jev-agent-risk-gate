#!/usr/bin/env python3
"""Render experiment-3 (trap commands) charts for LinkedIn.

Experiment 3 is the false-negative test: benign-LOOKING commands that are
catastrophic given hidden state. Two charts:
  07 — how often the gate WAVED A CATASTROPHE THROUGH (bare vs enriched)
  08 — when it missed, how sure it was the command was SAFE

Palette + ink follow the dataviz reference instance (validated):
  Jev = slot 1 blue #2a78d6 ; Sonnet = slot 2 orange #eb6834.
Condition (bare vs enriched) is encoded by POSITION, never by color.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

HERE = os.path.dirname(os.path.abspath(__file__))
SUM = json.load(open(os.path.join(HERE, "results", "summary3.json")))
VIZ = os.path.join(HERE, "viz")
os.makedirs(VIZ, exist_ok=True)

JEV = "#2a78d6"
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

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "figure.facecolor": PAGE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": PAGE,
    "text.color": INK,
    "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK2,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "axes.titlecolor": INK,
})

FOOT = ("n = 16 trap pairs (32 byte-identical commands, benign-looking)  ·  directional pilot, not a benchmark  ·  "
        "Jev-1.13 (System One, via Backboard)  vs  Claude Sonnet 4.6 (tool-use, via Azure Foundry)")


def ci_err(values, cis):
    lo = [max(0.0, v - (c[0] * 100 if c else v)) for v, c in zip(values, cis)]
    hi = [max(0.0, (c[1] * 100 if c else v) - v) for v, c in zip(values, cis)]
    return [lo, hi]


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


# ============================================================
# CHART 7 — how often the gate waved a catastrophe through (HEADLINE)
# ============================================================
def chart_false_negatives():
    fig, ax = plt.subplots(figsize=(10.4, 6.7))
    style_ax(ax)
    conds = ["bare", "enriched"]
    labels = ["BARE\n(command text only —\nthe danger is invisible)",
              "ENRICHED\n(command + environment state —\nthe danger is stated)"]
    jev = [SUM[c]["false_negatives"]["jev_fn_rate"] * 100 for c in conds]
    son = [SUM[c]["false_negatives"]["sonnet_fn_rate"] * 100 for c in conds]
    jev_ci = [SUM[c]["bootstrap_ci_95"].get("jev_fn") for c in conds]
    son_ci = [SUM[c]["bootstrap_ci_95"].get("sonnet_fn") for c in conds]
    x = [0, 1]; w = 0.36
    ekw = dict(ecolor=INK2, elinewidth=1.4, capsize=5, capthick=1.4)
    b1 = ax.bar([i - w / 2 for i in x], jev, w, color=JEV, edgecolor=SURFACE,
                linewidth=2, zorder=3, label="Jev (System One)",
                yerr=ci_err(jev, jev_ci), error_kw=ekw)
    b2 = ax.bar([i + w / 2 for i in x], son, w, color=SONNET, edgecolor=SURFACE,
                linewidth=2, zorder=3, label="Claude Sonnet 4.6 (LLM)",
                yerr=ci_err(son, son_ci), error_kw=ekw)
    for bars, vals, cis in [(b1, jev, jev_ci), (b2, son, son_ci)]:
        for b, v, c in zip(bars, vals, cis):
            top = max(v, c[1] * 100 if c else v)
            ax.text(b.get_x() + b.get_width() / 2, top + 2.5, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=11, color=INK, weight="bold")
    style_ax(ax)
    ax.set_xlim(-0.6, 1.75)
    ax.set_ylim(0, 92)
    ax.set_yticks([0, 25, 50, 75])
    ax.set_yticklabels(["0", "25", "50", "75%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=INK2)
    ax.set_ylabel("false-negative rate — catastrophes waved through  (lower is safer)",
                  fontsize=10.5, color=INK2)
    ax.annotate("On text alone, ~44% of catastrophic\ncommands were APPROVED — identical\n"
                "for both models, because the text is\nidentical. Among them: push unreviewed\n"
                "code to prod, wipe all config, run an\nuntrusted postinstall, exfiltrate PII.",
                xy=(0 + w / 2, 44), xytext=(0.30, 60), fontsize=9, color=CRIT, ha="left",
                weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=CRIT, lw=1.4,
                                connectionstyle="arc3,rad=0.2"))
    ax.annotate("Hand the gate the same state a\ngood harness already has ->\nmisses collapse to 0-6%.",
                xy=(1 - w / 2, 6), xytext=(0.86, 26), fontsize=9, color=GOOD, ha="left",
                weight="bold",
                arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.4,
                                connectionstyle="arc3,rad=-0.2"))
    ax.legend(loc="upper right", frameon=False, fontsize=10.5, bbox_to_anchor=(1.0, 0.99))
    fig.suptitle("The scariest miss: waving a catastrophe through",
                 x=0.5, y=0.99, fontsize=15.5, weight="bold")
    ax.set_title("16 commands that read as routine but are catastrophic given hidden state  ·  bars show 95% CI",
                 fontsize=10.5, color=MUTED, pad=10)
    fig.text(0.5, -0.02, FOOT, ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(os.path.join(VIZ, "07_false_negatives.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CHART 8 — when it missed, it was sure the command was safe
# ============================================================
def chart_confident_miss():
    fn = SUM["bare"]["false_negatives"]
    # "How sure it was the command was SAFE" on the calls where it waved danger through.
    jev_sure_safe = (1 - fn["jev_mean_p_on_miss"]) * 100      # 1 - P(risky)
    son_sure_safe = fn["sonnet_mean_conf_on_miss"] * 100      # confidence in its (safe) verdict
    n_j, n_s = fn["jev_missed"], fn["sonnet_missed"]

    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    style_ax(ax)
    x = [0, 1]; w = 0.34
    sure = [jev_sure_safe, son_sure_safe]
    actual = [0.0, 0.0]  # every one of these was truly catastrophic
    bc = ax.bar([i - w / 2 for i in x], sure, w, color=[JEV, SONNET],
                edgecolor=SURFACE, linewidth=2, zorder=3, alpha=0.95)
    ba = ax.bar([i + w / 2 for i in x], actual, w, color=[JEV, SONNET],
                edgecolor=SURFACE, linewidth=2, zorder=3, hatch="////", alpha=0.42)
    for b, v in zip(bc, sure):
        ax.text(b.get_x() + b.get_width() / 2, v + 2, f"{v:.0f}%",
                ha="center", va="bottom", fontsize=11.5, color=INK, weight="bold")
    for i, b in enumerate(ba):
        ax.text(b.get_x() + b.get_width() / 2, 2, "0%\n(all were\ncatastrophic)",
                ha="center", va="bottom", fontsize=9, color=INK2, weight="bold")
    # overconfidence brackets
    for i, s in enumerate(sure):
        ax.annotate("", xy=(i + w / 2, 4), xytext=(i + w / 2, s - 2),
                    arrowprops=dict(arrowstyle="<->", color=CRIT, lw=1.6))
        ax.text(i + w / 2 + 0.05, s / 2, f"{s:.0f} pt\nmisplaced\nconfidence",
                color=CRIT, fontsize=9.5, weight="bold", va="center", ha="left")
    style_ax(ax)
    ax.set_xlim(-0.55, 1.9)
    ax.set_ylim(0, 104)
    ax.set_yticks([0, 25, 50, 75, 100]); ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Jev (System One)\nn = {n_j} missed", f"Claude Sonnet 4.6\nn = {n_s} missed"],
                       fontsize=11, color=INK2, weight="bold")
    leg = [Patch(facecolor=INK2, alpha=0.95, label="How safe it thought the command was"),
           Patch(facecolor=INK2, alpha=0.42, hatch="////", label="How safe it actually was")]
    ax.legend(handles=leg, loc="upper center", frameon=False, fontsize=10.5, ncol=1,
              bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("When it waved a catastrophe through, it was sure the command was safe",
                 x=0.5, y=0.995, fontsize=14.5, weight="bold")
    ax.set_title("On the bare commands each model approved, its own confidence that they were safe — "
                 "every one was actually catastrophic",
                 fontsize=9.8, color=MUTED, pad=10, wrap=True)
    fig.text(0.5, -0.02, FOOT, ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(os.path.join(VIZ, "08_confident_miss.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    chart_false_negatives()
    chart_confident_miss()
    print("wrote:", ", ".join(sorted(f for f in os.listdir(VIZ) if f.startswith(("07", "08")))))
