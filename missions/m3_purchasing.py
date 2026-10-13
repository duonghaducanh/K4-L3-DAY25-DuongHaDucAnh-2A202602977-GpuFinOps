"""M3 — Purchasing Strategy: break-even, tier choice, spot-checkpoint sim (deck §4).

Run: python missions/m3_purchasing.py

Extensions implemented here:
  * Extension 1 — cost-minimising purchasing policy (`pricing.recommend_tier_v2`)
    that prices spot's interruption risk per GPU type and only counts a reserved
    commitment the job can actually amortise (1yr vs 3yr term).
  * Extension 5 — carbon-aware scheduling: move interruptible jobs to the
    cheapest + cleanest region and quantify $ and gCO2e saved.
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from missions._common import load_csv, num, catalog_by_type
from finops import pricing, sustainability

DAYS = 30


def _carbon_aware_schedule(jobs, cat, home_region="us-east-1", clean_region=None):
    """Extension 5 — reschedule interruptible jobs to the cheapest+cleanest region.

    Interruptible jobs have no user-facing latency SLO, so they are free to run
    anywhere. We compute the electricity $ and gCO2e each one incurs in the
    current region (us-east-1) vs. the cleanest grid, using GPU hours x TDP as
    the energy proxy.
    """
    clean_region = clean_region or min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    rows = []
    for j in jobs:
        if not bool(int(num(j["interruptible"]))):
            continue
        gtype = j["gpu_type"]
        gpu_hours = num(j["hours_per_day"]) * num(j["days"]) * int(num(j["num_gpus"]))
        kwh = gpu_hours * num(cat[gtype]["watts"]) / 1000.0
        wh = kwh * 1000.0
        home_cost = sustainability.energy_cost_usd(wh, home_region)
        clean_cost = sustainability.energy_cost_usd(wh, clean_region)
        home_carbon = sustainability.carbon_g(wh, home_region)
        clean_carbon = sustainability.carbon_g(wh, clean_region)
        rows.append({
            "job_id": j["job_id"], "gpu_type": gtype, "gpu_hours": round(gpu_hours),
            "kwh": round(kwh, 1),
            "home_region": home_region, "clean_region": clean_region,
            "home_cost": round(home_cost, 2), "clean_cost": round(clean_cost, 2),
            "cost_saved": round(home_cost - clean_cost, 2),
            "home_carbon_g": round(home_carbon), "clean_carbon_g": round(clean_carbon),
            "carbon_saved_g": round(home_carbon - clean_carbon),
        })
    totals = {
        "cost_home": round(sum(r["home_cost"] for r in rows), 2),
        "cost_clean": round(sum(r["clean_cost"] for r in rows), 2),
        "cost_saved": round(sum(r["cost_saved"] for r in rows), 2),
        "carbon_home_g": round(sum(r["home_carbon_g"] for r in rows)),
        "carbon_clean_g": round(sum(r["clean_carbon_g"] for r in rows)),
        "carbon_saved_g": round(sum(r["carbon_saved_g"] for r in rows)),
        "clean_region": clean_region,
    }
    totals["carbon_saved_pct"] = round(
        100.0 * totals["carbon_saved_g"] / totals["carbon_home_g"], 1) if totals["carbon_home_g"] else 0.0
    return rows, totals


def run(verbose: bool = True) -> dict:
    jobs = load_csv("workloads.csv")
    cat = catalog_by_type()
    on_demand_monthly = optimized_monthly = 0.0
    v1_monthly = 0.0
    one_yr_monthly = 0.0
    recs = []
    for j in jobs:
        gtype = j["gpu_type"]
        ngpu = int(num(j["num_gpus"]))
        hpd = num(j["hours_per_day"])
        days = num(j["days"])
        interruptible = bool(int(num(j["interruptible"])))
        c = cat[gtype]
        gpu_hours = hpd * DAYS * ngpu
        od = num(c["on_demand_hr"])
        on_demand_cost = gpu_hours * od

        # --- Extension 1: cost-minimising policy v2 ---
        # Workloads in this fleet are *recurring* pipelines, so the full reserved
        # commitment horizon is available (job_days=None -> cheapest term). The
        # short-lived-job case is reported separately as a sensitivity below.
        rec = pricing.recommend_tier_v2(
            hours_per_day=hpd, interruptible=interruptible,
            on_demand_hr=od, spot_hr=num(c["spot_hr"]),
            reserved_1yr_hr=num(c["reserved_1yr_hr"]), reserved_3yr_hr=num(c["reserved_3yr_hr"]),
            gpu_type=gtype, job_days=None,
        )
        # sensitivity: if a job could only commit 1 year, what would it pay?
        rec_1yr = pricing.recommend_tier_v2(
            hours_per_day=hpd, interruptible=interruptible,
            on_demand_hr=od, spot_hr=num(c["spot_hr"]),
            reserved_1yr_hr=num(c["reserved_1yr_hr"]), reserved_3yr_hr=num(c["reserved_3yr_hr"]),
            gpu_type=gtype, job_days=days,
        )
        if rec_1yr["tier"] == "reserved":
            one_yr_cost = gpu_hours * rec_1yr["effective_hr"]
        elif rec_1yr["tier"] == "spot":
            one_yr_cost = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od,
                                                       interrupt_rate=rec_1yr["interrupt_rate"])["spot_cost"]
        else:
            one_yr_cost = on_demand_cost
        one_yr_monthly += one_yr_cost
        tier = rec["tier"]
        if tier == "spot":
            sim = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od,
                                               interrupt_rate=rec["interrupt_rate"])
            opt_cost = sim["spot_cost"]
        elif tier == "reserved":
            opt_cost = gpu_hours * rec["effective_hr"]
        else:
            opt_cost = on_demand_cost

        # --- baseline for comparison: the documented v1 rule ---
        v1_tier = pricing.recommend_tier(hpd, interruptible)
        if v1_tier == "spot":
            v1_cost = pricing.spot_checkpoint_cost(gpu_hours, num(c["spot_hr"]), od)["spot_cost"]
        elif v1_tier == "reserved":
            v1_cost = gpu_hours * num(c["reserved_3yr_hr"])
        else:
            v1_cost = on_demand_cost
        v1_monthly += v1_cost

        on_demand_monthly += on_demand_cost
        optimized_monthly += opt_cost
        recs.append({"job_id": j["job_id"], "gpu_type": gtype, "tier": tier,
                     "term": rec["term"], "interrupt_rate": rec["interrupt_rate"],
                     "v1_tier": v1_tier,
                     "on_demand": round(on_demand_cost), "optimized": round(opt_cost)})

    savings = on_demand_monthly - optimized_monthly
    savings_pct = savings / on_demand_monthly * 100 if on_demand_monthly else 0.0
    v1_savings_pct = (on_demand_monthly - v1_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0
    one_yr_savings_pct = (on_demand_monthly - one_yr_monthly) / on_demand_monthly * 100 if on_demand_monthly else 0.0

    carbon_rows, carbon_totals = _carbon_aware_schedule(jobs, cat)

    if verbose:
        print("== M3 Purchasing Strategy ==")
        print(f"break-even utilization @ 45% reserved discount = {pricing.break_even_utilization(0.45):.0%}")
        print(f"{'job':18}{'gpu':7}{'tier':11}{'term':6}{'int%':>6}{'on-demand':>12}{'optimized':>12}")
        for r in recs:
            term = r["term"] or "-"
            print(f"{r['job_id']:18}{r['gpu_type']:7}{r['tier']:11}{term:6}"
                  f"{r['interrupt_rate']*100:>5.0f}%${r['on_demand']:>11,}${r['optimized']:>11,}")
        print(f"\nmonthly: on-demand ${on_demand_monthly:,.0f} -> optimized ${optimized_monthly:,.0f}  ({savings_pct:.1f}% saved)")
        print(f"policy v1 (documented rule) would save {v1_savings_pct:.1f}%  ->  "
              f"Extension 1 gains {savings_pct - v1_savings_pct:+.1f} pp")
        print(f"sensitivity — if only a 1yr commitment were allowed: {one_yr_savings_pct:.1f}% saved "
              f"(vs {savings_pct:.1f}% with the best available term)")

        print("\n-- Extension 1: tier matrix (interruptible, recurring jobs) --")
        print(f"{'gpu':8}" + "".join(f"{d*100:>8.0f}% duty" for d in (0.25, 0.55, 0.85, 1.0)))
        matrix = pricing.tier_matrix(cat, duty_cycles=(0.25, 0.55, 0.85, 1.0),
                                     interruptible=True, job_days=None)
        by_gpu = {}
        for m in matrix:
            by_gpu.setdefault(m["gpu_type"], []).append(m["tier"])
        for gtype, tiers in by_gpu.items():
            print(f"{gtype:8}" + "".join(f"{t:>13}" for t in tiers))

        print("\n-- Extension 5: carbon-aware scheduling (interruptible jobs) --")
        print(f"{'job':18}{'kWh':>8}{'$home':>8}{'$clean':>8}{'gCO2 home':>11}{'gCO2 clean':>11}{'saved_g':>9}")
        for r in carbon_rows:
            print(f"{r['job_id']:18}{r['kwh']:>8.1f}{r['home_cost']:>8.2f}{r['clean_cost']:>8.2f}"
                  f"{r['home_carbon_g']:>11,}{r['clean_carbon_g']:>11,}{r['carbon_saved_g']:>9,}")
        print(f"move all interruptible jobs to {carbon_totals['clean_region']}: "
              f"save ${carbon_totals['cost_saved']:,.0f}/run and "
              f"{carbon_totals['carbon_saved_g']:,} gCO2e ({carbon_totals['carbon_saved_pct']:.0f}%)")

    return {"recommendations": recs, "on_demand_monthly": round(on_demand_monthly),
            "optimized_monthly": round(optimized_monthly), "savings_pct": round(savings_pct, 1),
            "v1_savings_pct": round(v1_savings_pct, 1),
            "carbon_rows": carbon_rows, "carbon_totals": carbon_totals}


if __name__ == "__main__":
    run()