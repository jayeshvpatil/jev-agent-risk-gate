#!/usr/bin/env python3
"""
Experiment 3 — Trap commands: the false-negative test.

Every command reads as innocuous ("git checkout .", "git push", "npm ci"); only
hidden environment state makes the danger variant catastrophic. Unlike experiment 2
(where the text often looked scary and both models caught 100% of true danger), here
a text-only classifier is expected to APPROVE both members of a pair -- i.e. wave the
catastrophic variant through. This isolates under-blocking / false negatives, the one
failure mode experiments 1 and 2 never tested cleanly.

Same twin structure and plumbing as experiment 2 (byte-identical text within a pair),
so bare-text accuracy is still capped at exactly 50% by construction -- but the
composition is the inverse: safe variants cleared, danger variants MISSED.

Two arms x two state conditions x 32 hand-labeled variants (16 traps).
Commands are never executed; only classified. Keys read from env, never printed.
"""
import json
import os
import time

import run_experiment as e1
import run_experiment2 as e2

HERE = os.path.dirname(os.path.abspath(__file__))
TRAPS = os.path.join(HERE, "data", "traps.json")
OUT = os.path.join(HERE, "results", "results3.json")


def load_variants():
    """Flatten traps.json into one row per (twin, variant), carrying the pair id."""
    twins = json.load(open(TRAPS))["twins"]
    rows = []
    for t in twins:
        for v in t["variants"]:
            rows.append({
                "id": v["id"],
                "twin_id": t["twin_id"],
                "variant": v["variant"],       # "safe" | "danger"
                "cmd": t["cmd"],               # identical across the pair
                "context": v["context"],
                "label": v["label"],           # ground truth: True = must block
                "rationale": v["rationale"],
            })
    return rows


def main():
    variants = load_variants()
    conditions = {
        "bare": lambda a: a["cmd"],
        "enriched": lambda a: f"{a['cmd']}\n\nEnvironment context: {a['context']}",
    }
    results = []
    for cond, build in conditions.items():
        print(f"\n=== condition: {cond} ===", flush=True)
        for a in variants:
            state = build(a)
            row = {**{k: a[k] for k in ("id", "twin_id", "variant", "cmd", "label")},
                   "condition": cond}
            try:
                jp, jlat, jraw = e1.run_jev(state)
                row["jev_p"] = jp
                row["jev_latency"] = jlat
                ju = (jraw.get("system_one", {}) or {}).get("usage", {}) or {}
                row["jev_in_tok"] = ju.get("input_tokens")
                row["jev_out_tok"] = ju.get("output_tokens")
            except Exception as ex:
                row["jev_p"] = None
                row["jev_latency"] = None
                row["jev_error"] = repr(ex)
            try:
                lout, llat, lraw = e2.run_sonnet(state)
                row["llm_risky"] = lout.get("risky")
                row["llm_confidence"] = lout.get("confidence")
                row["llm_reason"] = lout.get("reason")
                row["llm_latency"] = llat
                row["llm_in_tok"] = lout.get("_in_tok")
                row["llm_out_tok"] = lout.get("_out_tok")
            except Exception as ex:
                row["llm_risky"] = None
                row["llm_latency"] = None
                row["llm_error"] = repr(ex)
            results.append(row)
            jp_s = f"{row.get('jev_p'):.3f}" if isinstance(row.get('jev_p'), float) else "ERR"
            jl_s = f"{row.get('jev_latency')*1000:.0f}ms" if isinstance(row.get('jev_latency'), float) else "--"
            ll_s = f"{row.get('llm_latency')*1000:.0f}ms" if isinstance(row.get('llm_latency'), float) else "--"
            print(f"  {a['id']:12} {a['variant']:6} jev={jp_s} ({jl_s})  "
                  f"sonnet={str(row.get('llm_risky')):5} ({ll_s})", flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nWrote {len(results)} rows to {OUT}")


if __name__ == "__main__":
    main()
