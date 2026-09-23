#!/usr/bin/env python3
"""Render LinkedIn-ready charts from the experiment results.

Palette + ink follow the dataviz reference instance (validated):
  Jev  = categorical slot 1 (blue  #2a78d6)
  Haiku= categorical slot 2 (orange#eb6834)
Condition (bare vs enriched) is encoded by PANEL, never by color.
"""
import json
import os
import statistics as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))
RES = json.load(open(os.path.join(HERE, "results", "results.json")))
SUM = json.load(open(os.path.join(HERE, "results", "summary.json")))
VIZ = os.path.join(HERE, "viz")
os.makedirs(VIZ, exist_ok=True)

# ---- palette (dataviz reference, light mode) ----
JEV = "#2a78d6"      # slot 1 blue
HAIKU = "#eb6834"    # slot 2 orange
BARE = "#2a78d6"     # condition series (calibration chart only)
ENR = "#eb6834"
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

QORDER = ["low_stakes_low_ambig", "low_stakes_high_ambig",
          "high_stakes_low_ambig", "high_stakes_high_ambig"]
QSHORT = {
    "low_stakes_low_ambig": "Low stakes\nLow ambiguity",
    "low_stakes_high_ambig": "Low stakes\nHigh ambiguity\n(looks scary,\nis safe)",
    "high_stakes_low_ambig": "High stakes\nLow ambiguity",
    "high_stakes_high_ambig": "High stakes\nHigh ambiguity\n(looks routine,\nis dangerous)",
}
FOOT = ("n = 10 per quadrant  ·  directional pilot, not a benchmark  ·  "
        "Jev-1.13 (System One, via Backboard)  vs  Claude Haiku 4.5 (tool-use)")


def style_ax(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASELINE)
    ax.tick_params(length=0)
    ax.set_axisbelow(True)


def bar_labels(ax, bars, fmt="{:.0f}%", threshold_flag=None):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + 2, fmt.format(h),
                ha="center", va="bottom", fontsize=10.5, color=INK, weight="bold")


# ============================================================
# CHART 1 — accuracy by quadrant, bare vs enriched (the hero)
# ============================================================
def chart_accuracy():
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 6.2), sharey=True)
    conds = [("bare", "State = command text only", axes[0]),
             ("enriched", "State = command text + environment context", axes[1])]
    x = list(range(len(QORDER)))
    w = 0.38
    for cond, subtitle, ax in conds:
        q = SUM[cond]["quadrants"]
        jev = [q[k]["jev_acc"] * 100 for k in QORDER]
        hai = [q[k]["llm_acc"] * 100 for k in QORDER]
        b1 = ax.bar([i - w / 2 for i in x], jev, w, label="Jev (System One)",
                    color=JEV, edgecolor=SURFACE, linewidth=2, zorder=3)
        b2 = ax.bar([i + w / 2 for i in x], hai, w, label="Haiku 4.5 (LLM)",
                    color=HAIKU, edgecolor=SURFACE, linewidth=2, zorder=3)
        bar_labels(ax, b1)
        bar_labels(ax, b2)
        style_ax(ax)
        ax.set_ylim(0, 112)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_yticklabels(["0", "25", "50", "75", "100%"])
        ax.yaxis.grid(True, color=GRID, linewidth=1)
        ax.set_xticks(x)
        ax.set_xticklabels([QSHORT[k] for k in QORDER], fontsize=8.5, color=INK2)
        ov = SUM[cond]["overall"]
        ax.set_title(f"{cond.upper()}   ·   overall  Jev {ov['jev_acc']*100:.0f}%  |  "
                     f"Haiku {ov['llm_acc']*100:.0f}%",
                     fontsize=12, weight="bold", pad=26)
        ax.text(0.5, 1.005, subtitle, transform=ax.transAxes, ha="center",
                va="bottom", fontsize=9.5, color=MUTED)
    # annotate the collapse
    ax0 = axes[0]
    ax0.annotate("Jev over-blocks\nsafe commands here",
                 xy=(1 - w / 2, 31), xytext=(0.2, 52),
                 fontsize=9.5, color=CRIT, weight="bold", ha="left",
                 arrowprops=dict(arrowstyle="-|>", color=CRIT, lw=1.6,
                                 connectionstyle="arc3,rad=-0.2"))
    axes[1].annotate("context lifts it\n30% to 90%",
                     xy=(1 - w / 2, 90), xytext=(1.1, 55),
                     fontsize=9.5, color=GOOD, weight="bold", ha="left",
                     arrowprops=dict(arrowstyle="-|>", color=GOOD, lw=1.6))
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=2, frameon=False, fontsize=11,
               bbox_to_anchor=(0.5, 0.99))
    fig.suptitle("Can a classifier gate an agent's shell commands?  Accuracy vs. hand-labeled risk",
                 x=0.5, y=1.055, fontsize=15, weight="bold")
    fig.text(0.5, -0.02, FOOT, ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.01, 1, 0.94])
    fig.savefig(os.path.join(VIZ, "01_accuracy_by_quadrant.png"), dpi=200,
                bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CHART 2 — Jev calibration (reliability), bare vs enriched
# ============================================================
def chart_calibration():
    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    style_ax(ax)
    # perfect-calibration reference
    ax.plot([0, 1], [0, 1], color=BASELINE, lw=1.5, ls=(0, (4, 4)), zorder=1)
    ax.text(0.80, 0.845, "perfect calibration", rotation=34, color=MUTED,
            fontsize=9, ha="center", va="center")
    # shade the region below the diagonal = overconfident
    ax.fill_between([0, 1], [0, 1], [0, 0], color=CRIT, alpha=0.04, zorder=0)
    ax.text(0.82, 0.12, "below the line =\noverconfident", color=CRIT, alpha=0.55,
            fontsize=8.5, ha="center", va="center", style="italic")

    # dots only (no connecting line: sparse, non-monotonic buckets)
    nudges = {  # (dx, dy) for n-labels, keyed by (cond, bucket-mid rounded)
    }
    for cond, color, lab in [("bare", BARE, "Bare state"),
                             ("enriched", ENR, "Enriched state")]:
        rel = [b for b in SUM[cond]["calibration"]["reliability"] if b["n"]]
        xs = [b["mean_pred"] for b in rel]
        ys = [b["empirical_risky_rate"] for b in rel]
        ns = [b["n"] for b in rel]
        ax.scatter(xs, ys, s=[70 + n * 30 for n in ns], color=color,
                   edgecolor=SURFACE, linewidth=1.8, zorder=4, alpha=0.92,
                   label=f"{lab}   (Brier {SUM[cond]['calibration']['brier_overall']:.3f})")
        for xx, yy, nn in zip(xs, ys, ns):
            dx = -0.028 if cond == "bare" else 0.028
            ax.text(xx + dx, yy + 0.045, f"n={nn}", ha="center", va="bottom",
                    fontsize=8, color=color, weight="bold")
    # overconfidence callout on the bare 0.6-0.8 bucket (pred ~0.70, actual 0.20)
    ax.annotate("Bare state: Jev says ~70% risky here,\n"
                "but only 20% actually are —\noverconfident in the ambiguous zone",
                xy=(0.70, 0.20), xytext=(0.06, 0.52), fontsize=9.5, color=CRIT,
                weight="bold", ha="left",
                arrowprops=dict(arrowstyle="-|>", color=CRIT, lw=1.7,
                                connectionstyle="arc3,rad=-0.2"))
    ax.set_xlim(0, 1.02); ax.set_ylim(-0.02, 1.05)
    ax.set_xticks([0, .2, .4, .6, .8, 1.0]); ax.set_yticks([0, .2, .4, .6, .8, 1.0])
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_xlabel("Jev's stated probability the command is risky",
                  fontsize=11, color=INK2, labelpad=8)
    ax.set_ylabel("Fraction that were actually risky (hand-labeled)",
                  fontsize=11, color=INK2)
    ax.set_title("Does Jev's confidence mean what it says?\nReliability of the risk probability",
                 fontsize=13.5, weight="bold", pad=14)
    ax.legend(loc="center right", frameon=False, fontsize=10.5, bbox_to_anchor=(1.0, 0.42))
    fig.text(0.5, 0.005, "n = 40 per condition  ·  dot size grows with commands in that "
             "probability bucket  ·  enriched Brier 0.015 vs bare 0.101 (lower = better)",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.03, 1, 1])
    fig.savefig(os.path.join(VIZ, "02_calibration.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# CHART 3 — latency + cost (absolute numbers, not multipliers)
# ============================================================
def pooled():
    jl = sorted(r["jev_latency"] * 1000 for r in RES if isinstance(r.get("jev_latency"), float))
    ll = sorted(r["llm_latency"] * 1000 for r in RES if isinstance(r.get("llm_latency"), float))
    def pct(xs, p):
        k = (len(xs) - 1) * p / 100
        lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
        return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)
    jev_in = st.mean([r["jev_in_tok"] for r in RES if isinstance(r.get("jev_in_tok"), (int, float))])
    hin = st.mean([r["llm_in_tok"] for r in RES if isinstance(r.get("llm_in_tok"), (int, float))])
    hout = st.mean([r["llm_out_tok"] for r in RES if isinstance(r.get("llm_out_tok"), (int, float))])
    jev_cost = 1000 * (jev_in / 1e6) * 0.042
    hai_cost = 1000 * ((hin / 1e6) * 1.0 + (hout / 1e6) * 5.0)
    return {
        "jev": {"p50": pct(jl, 50), "p95": pct(jl, 95)},
        "hai": {"p50": pct(ll, 50), "p95": pct(ll, 95)},
        "jev_cost": jev_cost, "hai_cost": hai_cost,
    }


def chart_latency_cost():
    p = pooled()
    fig, (axL, axC) = plt.subplots(1, 2, figsize=(12.5, 5.6),
                                   gridspec_kw={"width_ratios": [1.35, 1]})
    # ---- latency (grouped: p50 / p95) ----
    style_ax(axL)
    groups = ["p50 (median)", "p95 (tail)"]
    x = [0, 1]; w = 0.38
    jev = [p["jev"]["p50"], p["jev"]["p95"]]
    hai = [p["hai"]["p50"], p["hai"]["p95"]]
    b1 = axL.bar([i - w / 2 for i in x], jev, w, color=JEV, edgecolor=SURFACE,
                 linewidth=2, zorder=3, label="Jev (System One)")
    b2 = axL.bar([i + w / 2 for i in x], hai, w, color=HAIKU, edgecolor=SURFACE,
                 linewidth=2, zorder=3, label="Haiku 4.5 (LLM)")
    for b in list(b1) + list(b2):
        axL.text(b.get_x() + b.get_width() / 2, b.get_height() + 25,
                 f"{b.get_height():.0f} ms", ha="center", va="bottom",
                 fontsize=10.5, color=INK, weight="bold")
    axL.set_xticks(x); axL.set_xticklabels(groups, fontsize=10.5, color=INK2)
    axL.set_ylim(0, max(hai) * 1.22)
    axL.yaxis.grid(True, color=GRID, linewidth=1)
    axL.set_ylabel("end-to-end latency (ms)", fontsize=10.5, color=INK2)
    axL.set_title(f"Latency — both routed through hosted APIs\n"
                  f"median {p['jev']['p50']:.0f} ms vs {p['hai']['p50']:.0f} ms  "
                  f"(~{p['hai']['p50']/p['jev']['p50']:.1f}×, not the 200× headline)",
                  fontsize=11.5, weight="bold", pad=12)
    axL.legend(loc="upper left", frameon=False, fontsize=10)
    # ---- cost (log scale, huge gap) ----
    style_ax(axC)
    vals = [p["jev_cost"], p["hai_cost"]]
    cols = [JEV, HAIKU]
    bars = axC.bar([0, 1], vals, 0.55, color=cols, edgecolor=SURFACE,
                   linewidth=2, zorder=3)
    axC.set_yscale("log")
    for b, v in zip(bars, vals):
        axC.text(b.get_x() + b.get_width() / 2, v * 1.15, f"${v:.3f}",
                 ha="center", va="bottom", fontsize=12, color=INK, weight="bold")
    axC.set_xticks([0, 1]); axC.set_xticklabels(["Jev", "Haiku 4.5"], fontsize=11, color=INK2)
    axC.set_ylim(vals[0] * 0.4, vals[1] * 3)
    axC.yaxis.grid(True, color=GRID, linewidth=1, which="both")
    axC.set_ylabel("USD per 1,000 calls  (log scale)", fontsize=10.5, color=INK2)
    axC.set_title(f"Cost per 1,000 gate checks\nHaiku costs ~{p['hai_cost']/p['jev_cost']:.0f}× more",
                  fontsize=11.5, weight="bold", pad=12)
    fig.suptitle("The real trade: modest latency win, order-of-magnitude cost win",
                 x=0.5, y=1.02, fontsize=14.5, weight="bold")
    fig.text(0.5, -0.03,
             "n = 80 calls per model (both conditions pooled)  ·  latency is END-TO-END through "
             "Backboard / Azure Foundry, not raw model inference  ·  real token counts",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, 0.02, 1, 0.95])
    fig.savefig(os.path.join(VIZ, "03_latency_cost.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    chart_accuracy()
    chart_calibration()
    chart_latency_cost()
    print("wrote:", ", ".join(sorted(os.listdir(VIZ))))
