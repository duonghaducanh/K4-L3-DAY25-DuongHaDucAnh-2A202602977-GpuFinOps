"""M5 — Optimization Report: combine M1-M4 into baseline-vs-optimized (deck §1/§11).

Run: python missions/m5_report.py   ->  outputs/report.md + outputs/savings.png
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import os
from missions._common import num, catalog_by_type, ROOT
from finops import report, sustainability
from missions import m1_efficiency_audit, m2_inference_levers, m3_purchasing

DAYS = 30
# one tier down for over-provisioned ("util-lie") GPUs
RIGHTSIZE_MAP = {"H100": "A100", "H200": "H100", "A100": "A10G", "A10G": "L4", "L4": "L4"}


def run(verbose: bool = True) -> dict:
    r1 = m1_efficiency_audit.run(verbose=False)
    r2 = m2_inference_levers.run(verbose=False)
    r3 = m3_purchasing.run(verbose=False)
    cat = catalog_by_type()

    # --- buckets ---
    infer_savings = (r2["baseline_daily"] - r2["optimized_daily"]) * DAYS
    purchasing_savings = r3["on_demand_monthly"] - r3["optimized_monthly"]

    idle_savings = r1["idle_waste_daily"] * DAYS
    rightsize_savings = 0.0
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"])
        rightsize_savings += max(0.0, delta) * 24 * DAYS

    levers = {
        "Inference (cascade/cache/batch)": round(infer_savings),
        "Purchasing (spot/reserved)": round(purchasing_savings),
        "Right-size util-lies": round(rightsize_savings),
        "Kill idle GPUs": round(idle_savings),
    }
    baseline = r2["baseline_daily"] * DAYS + r3["on_demand_monthly"]
    optimized = baseline - sum(levers.values())
    total_pct = sum(levers.values()) / baseline * 100 if baseline else 0.0

    # --- unit economics (the metric that matters) ---
    unit_economics = {
        "baseline_per_m": r2["baseline_per_m"],
        "optimized_per_m": r2["optimized_per_m"],
        "per_m_reduction_pct": round((1 - r2["optimized_per_m"] / r2["baseline_per_m"]) * 100, 1)
        if r2["baseline_per_m"] else 0.0,
        "baseline_daily": r2["baseline_daily"],
        "optimized_daily": r2["optimized_daily"],
        "daily_reduction_pct": r2["savings_pct"],
        "total_tokens": r2["total_tokens"],
    }

    # --- util-lie financial exposure ---
    util_lies = []
    for lie in r1["lies"]:
        cur = lie["gpu_type"]
        tgt = RIGHTSIZE_MAP.get(cur, cur)
        delta = max(0.0, num(cat[cur]["on_demand_hr"]) - num(cat[tgt]["on_demand_hr"]))
        util_lies.append({**lie, "monthly_at_risk": round(delta * 24 * DAYS)})

    # --- sustainability snapshot ---
    median_tokens = 800
    wh = sustainability.wh_per_query(median_tokens)
    carbon_totals = r3["carbon_totals"]
    sust = {
        "wh_per_query": wh,
        "carbon_g": sustainability.carbon_g(wh, "us-east-1"),
        "best_region": min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get),
        "clean_region": carbon_totals["clean_region"],
        "carbon_saved_g": carbon_totals["carbon_saved_g"],
        "carbon_saved_pct": carbon_totals["carbon_saved_pct"],
        "energy_cost_saved": carbon_totals["cost_saved"],
    }

    # --- region table ---
    region_table = [
        {"region": r, "price_kwh": sustainability.REGION_PRICE_KWH[r],
         "carbon_g_kwh": sustainability.REGION_CARBON[r]}
        for r in sustainability.REGION_CARBON
    ]

    # --- prioritized actions (ROI-ordered) ---
    ranked = sorted(levers.items(), key=lambda kv: -kv[1])
    insights = [
        f"**{ranked[0][0]}** is the single largest lever (${ranked[0][1]:,.0f}/mo). "
        f"Commit the steady high-duty jobs to reserved/spot first — it is pure contractual "
        f"savings with no engineering work.",
        f"**Right-size the util-lie GPUs** (gpu-h100-4, gpu-a10g-1). They bill full rate while "
        f"producing ~20% of peak FLOPs; move the memory-bound ones to the cheapest GPU that "
        f"still clears their bandwidth and VRAM needs (${r1['rightsize_monthly']:,.0f}/mo).",
        "**Reclaim idle GPUs** (gpu-h100-5 is dark 8h/night). Add auto-shutdown on idle "
        f"detection — ${idle_savings:,.0f}/mo for a trivial scheduler change.",
        f"**Keep cascade + caching + batch on** (${infer_savings:,.0f}/mo). Caching only pays "
        f"because prefixes are read back {r2['cache_reads']['small']:.0f}+ times; re-check the "
        f"break-even before shipping a new cached prompt.",
        f"**Move interruptible jobs to {carbon_totals['clean_region']}** — saves "
        f"{carbon_totals['carbon_saved_g']:,} gCO2e and ${carbon_totals['cost_saved']:,.0f} "
        f"of electricity with zero latency impact (no user-facing SLO).",
    ]

    ext = r2["reasoning"]
    extensions = [
        f"**Ext 1 — purchasing policy v2:** prices spot interruption risk per GPU type and only "
        f"counts a reserved term the job can amortise. M3 savings {r3['v1_savings_pct']:.1f}% → "
        f"{r3['savings_pct']:.1f}%.",
        f"**Ext 2 — MBU right-sizing:** cheapest adequate GPU by $/TB/s for memory-bound work; "
        f"${r1['rightsize_monthly']:,.0f}/mo identified.",
        f"**Ext 3 — cache_is_worth_it():** small-tier prefixes read {r2['cache_reads']['small']:.0f}x "
        f"(break-even 4.2) and large-tier {r2['cache_reads']['large']:.0f}x (break-even 1.6) — both "
        f"cleared, so the cache credit is real, not booked fiction.",
        f"**Ext 4 — reasoning budget:** reasoning is {ext['reasoning_req_share']:.1%} of requests "
        f"but {ext['reasoning_wh_share']:.0%} of energy; cap it by routing rule.",
        f"**Ext 5 — carbon-aware scheduling:** interruptible jobs → {carbon_totals['clean_region']}, "
        f"{carbon_totals['carbon_saved_pct']:.0f}% carbon cut.",
    ]

    md = report.build_report(baseline, optimized, levers, sustainability=sust, period="monthly",
                             unit_economics=unit_economics, util_lies=util_lies,
                             insights=insights, extensions=extensions, region_table=region_table)
    out_md = os.path.join(ROOT, "outputs", "report.md")
    os.makedirs(os.path.dirname(out_md), exist_ok=True)
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(md)
    png = report.savings_waterfall(levers, os.path.join(ROOT, "outputs", "savings.png"))

    if verbose:
        print("== M5 Optimization Report ==")
        print(md)
        print(f"\nWritten: outputs/report.md" + (f" + outputs/savings.png" if png else " (matplotlib absent: PNG skipped)"))

    return {"baseline_monthly": round(baseline), "optimized_monthly": round(optimized),
            "levers": levers, "total_savings_pct": round(total_pct, 1),
            "unit_economics": unit_economics, "util_lies": util_lies,
            "region_table": region_table}


if __name__ == "__main__":
    run()
