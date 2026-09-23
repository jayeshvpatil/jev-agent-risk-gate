#!/usr/bin/env python3
"""Render experiment-2 (twin commands) charts for LinkedIn.

Palette + ink follow the dataviz reference instance (validated):
  Jev    = categorical slot 1 (blue   #2a78d6)
  Sonnet = categorical slot 2 (orange #eb6834)
Condition (bare vs enriched) is encoded by POSITION, never by color.
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, "results", "results2.json")))
SUM = json.load(open(os.path.join(HERE, "results", "summary2.json")))
VIZ = os.path.join(HERE, "viz")
os.makedirs(VIZ, exist_ok=True)

# ---- palette (dataviz reference, light mode) ----
JEV = "#2a78d6"      # slot 1 blue
SONNET = "#eb6834"   # slot 2 orange
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

FOOT = ("n = 20 twin pairs (40 byte-identical commands)  ·  directional pilot, not a benchmark  ·  "
        "Jev-1.13 (System One, via Backboard)  vs  Claude Sonnet 4.6 (tool-use, via Azure Foundry)")


def ci_err(values, cis):
    """Convert value + [lo,hi] CI pairs into matplotlib asymmetric yerr (percent)."""
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


def bar_labels(ax, bars, fmt="{:.0f}%", dy=2):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontsize=10.5, color=INK, weight="bold")


# ============================================================
# CHART 4 — a stronger model can't beat missing information
# ============================================================
def chart_twin_accuracy():
    fig, ax = plt.subplots(figsize=(10.2, 6.6))
    style_ax(ax)
    conds = ["bare", "enriched"]
    labels = ["BARE\n(command text only —\ninformation is absent)",
              "ENRICHED\n(command + environment state —\ninformation is present)"]
    jev = [SUM[c]["overall"]["jev_acc"] * 100 for c in conds]
    son = [SUM[c]["overall"]["sonnet_acc"] * 100 for c in conds]
    jev_ci = [SUM[c]["bootstrap_ci_95"].get("jev_overall") for c in conds]
    son_ci = [SUM[c]["bootstrap_ci_95"].get("sonnet_overall") for c in conds]
    x = [0, 1]; w = 0.36
    ekw = dict(ecolor=INK2, elinewidth=1.4, capsize=5, capthick=1.4)
    # coin-flip reference
    ax.axhline(50, color=CRIT, lw=1.6, ls=(0, (5, 4)), zorder=1)
    ax.text(1.46, 50, "coin flip (50%)", color=CRIT, fontsize=9.5, va="center",
            ha="left", weight="bold",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    b1 = ax.bar([i - w / 2 for i in x], jev, w, color=JEV, edgecolor=SURFACE,
                linewidth=2, zorder=3, label="Jev (System One)",
                yerr=ci_err(jev, jev_ci), error_kw=ekw)
    b2 = ax.bar([i + w / 2 for i in x], son, w, color=SONNET, edgecolor=SURFACE,
                linewidth=2, zorder=3, label="Claude Sonnet 4.6 (LLM)",
                yerr=ci_err(son, son_ci), error_kw=ekw)
    for bars, vals, cis in [(b1, jev, jev_ci), (b2, son, son_ci)]:
        for b, v, c in zip(bars, vals, cis):
            top = max(v, c[1] * 100 if c else v)
            ax.text(b.get_x() + b.get_width() / 2, top + 3, f"{v:.0f}%",
                    ha="center", va="bottom", fontsize=10.5, color=INK, weight="bold")
    style_ax(ax)
    ax.set_xlim(-0.6, 1.9)
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10, color=INK2)
    ax.set_ylabel("accuracy vs. hand-labeled risk", fontsize=11, color=INK2)
    # annotate the bare tie
    ax.annotate("Byte-identical commands.\nSonnet is the stronger model —\n"
                "and ties Jev at the coin flip.\nThe ceiling is the missing\ninformation, not model size.",
                xy=(0.0, 50), xytext=(-0.55, 74), fontsize=9.5, color=INK2, ha="left",
                arrowprops=dict(arrowstyle="-|>", color=INK2, lw=1.4,
                                connectionstyle="arc3,rad=0.15"))
    ax.annotate("Same state a good harness\nalready has -> both jump.",
                xy=(1 + w / 2, 100), xytext=(0.62, 108), fontsize=9.5, color=GOOD,
                weight="bold", ha="left")
    ax.legend(loc="upper left", frameon=False, fontsize=10.5, bbox_to_anchor=(0.005, 0.99))
    fig.suptitle("A smarter model can't beat missing information",
                 x=0.5, y=0.99, fontsize=15.5, weight="bold")
    ax.set_title("Risk-gating 20 pairs of commands where only the environment flips the label  ·  bars show 95% CI",
                 fontsize=10.5, color=MUTED, pad=10)
    fig.text(0.5, -0.02, FOOT, ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    fig.savefig(os.path.join(VIZ, "04_twin_accuracy.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CHART 5 — neither model knows when it's guessing (HEADLINE)
# ============================================================
def chart_overconfidence():
    bare = [r for r in RES if r["condition"] == "bare"]
    # Jev "expressed confidence in its pick" = mean of max(p, 1-p)
    jp = [r["jev_p"] for r in bare if isinstance(r.get("jev_p"), float)]
    jev_conf = st.mean([max(p, 1 - p) for p in jp]) * 100
    sc = [r["llm_confidence"] for r in bare if isinstance(r.get("llm_confidence"), float)]
    son_conf = st.mean(sc) * 100
    jev_acc = SUM["bare"]["overall"]["jev_acc"] * 100
    son_acc = SUM["bare"]["overall"]["sonnet_acc"] * 100

    fig, ax = plt.subplots(figsize=(10.2, 6.8))
    style_ax(ax)
    # groups: Jev, Sonnet; bars: confidence vs accuracy
    x = [0, 1]; w = 0.34
    conf = [jev_conf, son_conf]
    acc = [jev_acc, son_acc]
    ax.axhline(50, color=MUTED, lw=1.4, ls=(0, (5, 4)), zorder=1)
    ax.text(1.52, 50, "coin flip", color=MUTED, fontsize=9.5, va="center", ha="left",
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5))
    bc = ax.bar([i - w / 2 for i in x], conf, w, color=[JEV, SONNET],
                edgecolor=SURFACE, linewidth=2, zorder=3, alpha=0.95)
    ba = ax.bar([i + w / 2 for i in x], acc, w, color=[JEV, SONNET],
                edgecolor=SURFACE, linewidth=2, zorder=3, hatch="////", alpha=0.42)
    bar_labels(ax, bc); bar_labels(ax, ba)
    # gap brackets
    for i, (c, a) in enumerate(zip(conf, acc)):
        ax.annotate("", xy=(i + w / 2, a + 4), xytext=(i + w / 2, c - 2),
                    arrowprops=dict(arrowstyle="<->", color=CRIT, lw=1.6))
        ax.text(i + w / 2 + 0.06, (c + a) / 2, f"{c - a:.0f} pt\noverconfidence",
                color=CRIT, fontsize=9.5, weight="bold", va="center", ha="left")
    style_ax(ax)
    ax.set_xlim(-0.55, 2.0)
    ax.set_ylim(0, 116)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_yticklabels(["0", "25", "50", "75", "100%"])
    ax.yaxis.grid(True, color=GRID, linewidth=1)
    ax.set_xticks(x); ax.set_xticklabels(["Jev (System One)", "Claude Sonnet 4.6"],
                                         fontsize=11.5, color=INK2, weight="bold")
    # legend by fill pattern
    from matplotlib.patches import Patch
    leg = [Patch(facecolor=INK2, alpha=0.95, label="How confident it sounded"),
           Patch(facecolor=INK2, alpha=0.42, hatch="////", label="How often it was right")]
    ax.legend(handles=leg, loc="upper center", frameon=False, fontsize=10.5, ncol=2,
              bbox_to_anchor=(0.5, 0.90))
    fig.suptitle("Neither model knows when it's guessing",
                 x=0.5, y=0.995, fontsize=16, weight="bold")
    ax.set_title("On byte-identical commands the answer isn't in the text — yet both answer with "
                 "high confidence and land at a coin flip",
                 fontsize=10, color=MUTED, pad=10, wrap=True)
    fig.text(0.5, -0.02,
             "Twin pairs correctly separated from text alone: 0 of 20 for BOTH models  ·  "
             + FOOT, ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(os.path.join(VIZ, "05_overconfidence.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CHART 6 — where the big model earns its price (enriched) + the trade
# ============================================================
def chart_where_sonnet_wins():
    fig, (axA, axC) = plt.subplots(1, 2, figsize=(12.6, 5.8),
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    # ---- left: enriched accuracy by variant ----
    style_ax(axA)
    bv = SUM["enriched"]["by_variant"]
    groups = ["Safe commands\n(look scary, are fine)", "Dangerous commands"]
    jev = [bv["safe"]["jev_acc"] * 100, bv["danger"]["jev_acc"] * 100]
    son = [bv["safe"]["sonnet_acc"] * 100, bv["danger"]["sonnet_acc"] * 100]
    ci = SUM["enriched"]["bootstrap_ci_95"]
    jev_ci = [ci.get("jev_safe"), None]     # danger = 100%, no whisker
    son_ci = [ci.get("sonnet_safe"), None]
    x = [0, 1.6]; w = 0.38
    ekw = dict(ecolor=INK2, elinewidth=1.4, capsize=5, capthick=1.4)
    b1 = axA.bar([i - w / 2 for i in x], jev, w, color=JEV, edgecolor=SURFACE,
                 linewidth=2, zorder=3, label="Jev (System One)",
                 yerr=ci_err(jev, jev_ci), error_kw=ekw)
    b2 = axA.bar([i + w / 2 for i in x], son, w, color=SONNET, edgecolor=SURFACE,
                 linewidth=2, zorder=3, label="Claude Sonnet 4.6",
                 yerr=ci_err(son, son_ci), error_kw=ekw)
    for bars, vals, cis in [(b1, jev, jev_ci), (b2, son, son_ci)]:
        for b, v, c in zip(bars, vals, cis):
            top = max(v, c[1] * 100 if c else v)
            axA.text(b.get_x() + b.get_width() / 2, top + 3, f"{v:.0f}%",
                     ha="center", va="bottom", fontsize=10.5, color=INK, weight="bold")
    axA.set_xlim(-0.6, 2.2)
    axA.set_ylim(0, 116)
    axA.set_yticks([0, 25, 50, 75, 100]); axA.set_yticklabels(["0", "25", "50", "75", "100%"])
    axA.yaxis.grid(True, color=GRID, linewidth=1)
    axA.set_xticks(x); axA.set_xticklabels(groups, fontsize=9.5, color=INK2)
    axA.legend(loc="lower center", frameon=False, fontsize=9.5, bbox_to_anchor=(0.5, 0.02))
    axA.annotate("Even with context, Jev still\nover-blocks 2 in 5 safe-but-\n"
                 "scary commands.\nSonnet reasons through them.",
                 xy=(0 - w / 2 + 0.02, 60), xytext=(0.44, 34), fontsize=9, color=CRIT,
                 weight="bold", ha="left",
                 arrowprops=dict(arrowstyle="-|>", color=CRIT, lw=1.5,
                                 connectionstyle="arc3,rad=-0.25"))
    axA.set_title("Enriched: where Sonnet's reasoning earns its price\n"
                  "Both catch 100% of true danger — Sonnet also clears the false alarms",
                  fontsize=10.8, weight="bold", pad=10)
    # ---- right: the durable trade (cost log + latency) ----
    style_ax(axC)
    c = SUM["cost_per_1k_calls_usd"]
    lat_j = SUM["bare"]["latency"]["jev_ms"]["p50"]
    lat_s = SUM["bare"]["latency"]["sonnet_ms"]["p50"]
    vals = [c["jev"], c["sonnet"]]
    bars = axC.bar([0, 1], vals, 0.55, color=[JEV, SONNET], edgecolor=SURFACE,
                   linewidth=2, zorder=3)
    axC.set_yscale("log")
    for b, v in zip(bars, vals):
        axC.text(b.get_x() + b.get_width() / 2, v * 1.18, f"${v:,.3f}",
                 ha="center", va="bottom", fontsize=12, color=INK, weight="bold")
    axC.set_xticks([0, 1]); axC.set_xticklabels(["Jev", "Sonnet 4.6"], fontsize=11, color=INK2)
    axC.set_ylim(vals[0] * 0.4, vals[1] * 3.2)
    axC.yaxis.grid(True, color=GRID, linewidth=1, which="both")
    axC.set_ylabel("USD per 1,000 gate checks  (log scale)", fontsize=10.5, color=INK2)
    axC.set_title(f"The durable advantage: cost & latency\n"
                  f"~{c.get('sonnet_over_jev_x',0):.0f}x cheaper  ·  "
                  f"{lat_j:.0f} ms vs {lat_s:.0f} ms median (~{lat_s/lat_j:.1f}x faster)",
                  fontsize=10.8, weight="bold", pad=10)
    fig.suptitle("The honest scorecard: reasoning buys the last mile, price and speed stay with Jev",
                 x=0.5, y=1.0, fontsize=13.5, weight="bold")
    fig.text(0.5, -0.03,
             "latency is END-TO-END through hosted APIs, not raw inference  ·  real token counts  ·  "
             + FOOT, ha="center", fontsize=7.6, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.94])
    fig.savefig(os.path.join(VIZ, "06_where_sonnet_wins.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    chart_twin_accuracy()
    chart_overconfidence()
    chart_where_sonnet_wins()
    print("wrote:", ", ".join(sorted(f for f in os.listdir(VIZ) if f.startswith(("04", "05", "06")))))
