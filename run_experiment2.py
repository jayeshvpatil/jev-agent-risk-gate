#!/usr/bin/env python3
"""
Experiment 2 — Twin commands: Jev (System One) vs Claude Sonnet 4.6 (System Two).

The command text within each twin pair is BYTE-IDENTICAL; only the environment
context flips the ground-truth label. So bare-text accuracy is capped near a coin
flip by construction, and the only way to be right on both members of a pair is to
consume the state. Sonnet replaces Haiku (exp 1) to test whether a STRONGER System
Two model can beat that coin flip on bare text. It cannot — the information isn't there.

Two arms x two state conditions x 40 hand-labeled variants (20 twins).
Reuses the Jev + Foundry plumbing from run_experiment.py; only the System Two
deployment name changes to Sonnet.

Key is read from BACKBOARD_API_KEY / ANTHROPIC_FOUNDRY_*. Never printed.
"""
import json
import os
import time

import run_experiment as e1

HERE = os.path.dirname(os.path.abspath(__file__))
TWINS = os.path.join(HERE, "data", "twins.json")
OUT = os.path.join(HERE, "results", "results2.json")

# System Two arm for this experiment: Claude Sonnet 4.6 (newest Sonnet — strongest
# available falsification of "a smarter model can beat missing information").
# Overridable via env; earlier run used claude-sonnet-4-5 (archived as *_s45.json).
SONNET_MODEL = os.environ.get("SONNET_MODEL", "claude-sonnet-4-6")


def run_sonnet(state):
    """Returns (verdict_dict, latency_s, raw). Same contract as e1.run_llm, Sonnet model."""
    payload = {
        "model": SONNET_MODEL,
        "max_tokens": 300,
        "system": e1.LLM_SYSTEM,
        "tools": [e1.ASSESS_TOOL],
        "tool_choice": {"type": "tool", "name": "assess_risk"},
        "messages": [{"role": "user", "content": f"Assess this command / state:\n{state}"}],
    }
    t0 = time.perf_counter()
    body = e1._foundry_post(payload)
    lat = time.perf_counter() - t0
    verdict = {"risky": None, "confidence": None, "reason": "PARSE_FAIL"}
    for block in body.get("content", []):
        if block.get("type") == "tool_use" and block.get("name") == "assess_risk":
            inp = block.get("input", {})
            verdict = {
                "risky": bool(inp.get("risky")),
                "confidence": float(inp.get("confidence")) if inp.get("confidence") is not None else None,
                "reason": str(inp.get("reason", "")),
            }
            break
    usage = body.get("usage", {})
    verdict["_in_tok"] = usage.get("input_tokens")
    verdict["_out_tok"] = usage.get("output_tokens")
    return verdict, lat, body


def load_variants():
    """Flatten twins.json into one row per (twin, variant), carrying the pair id."""
    twins = json.load(open(TWINS))["twins"]
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
                lout, llat, lraw = run_sonnet(state)
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
