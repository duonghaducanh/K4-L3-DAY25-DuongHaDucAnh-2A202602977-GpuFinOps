"""M1 — Efficiency Audit: MFU/MBU, the GPU-Util lie, and idle waste (deck §5).

Run: python missions/m1_efficiency_audit.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import defaultdict
from missions._common import load_csv, num, catalog_by_type
from finops import metrics

DAYS = 30
# Workloads that are bandwidth-bound in production: decode / embedding lookups.
MEMORY_BOUND_WORKLOADS = ("infer", "embed")


def _rightsize_memory_bound(tel, cat):
    """Extension 2 — right-size memory-bound GPUs by MBU, not by sticker price.

    For every inference/embed GPU we compute its *observed* achieved bandwidth
    and peak VRAM footprint, then ask `metrics.rightsize_candidate` for the
    cheapest GPU type that still clears both bars (with a 5% safety margin).
    The point: the cheapest $/GPU-hr box is often NOT the cheapest way to serve
    a memory-bound workload — you must compare $ per TB/s of bandwidth.
    """
    agg = defaultdict(lambda: {"type": None, "workload": None, "bw": [], "mem": [], "mbu": []})
    for r in tel:
        if r["workload"] not in MEMORY_BOUND_WORKLOADS:
            continue
        gt = r["gpu_type"]
        a = agg[r["gpu_id"]]
        a["type"] = gt
        a["workload"] = r["workload"]
        a["bw"].append(num(r["achieved_bw_tbs"]))
        a["mem"].append(num(r["mem_used_gb"]))
        a["mbu"].append(metrics.compute_mbu(num(r["achieved_bw_tbs"]), num(cat[gt]["peak_bw_tbs"])))

    out = []
    monthly_total = 0.0
    for gid, a in sorted(agg.items()):
        gt = a["type"]
        need_bw = sum(a["bw"]) / len(a["bw"])
        need_mem = max(a["mem"])
        cur_hr = num(cat[gt]["on_demand_hr"])
        cand = metrics.rightsize_candidate(need_bw, need_mem, cat, current_type=gt)
        row = {
            "gpu_id": gid, "gpu_type": gt, "workload": a["workload"],
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "need_bw_tbs": round(need_bw, 3), "need_vram_gb": round(need_mem, 1),
            "cur_hr": cur_hr,
            "cur_dollars_per_tbps": round(metrics.dollars_per_tbps(cur_hr, num(cat[gt]["peak_bw_tbs"])), 3),
            "candidate": cand,
        }
        if cand:
            monthly = (cur_hr - cand["on_demand_hr"]) * 24 * DAYS
            row["monthly_savings"] = round(monthly, 0)
            monthly_total += monthly
        else:
            row["monthly_savings"] = 0.0
        out.append(row)
    return out, round(monthly_total, 0)


def run(verbose: bool = True) -> dict:
    tel = load_csv("gpu_telemetry.csv")
    cat = catalog_by_type()

    # per-row MFU/MBU, then aggregate per GPU
    agg = defaultdict(lambda: {"util": [], "mfu": [], "mbu": [], "type": None, "idle_hours": 0})
    for r in tel:
        gtype = r["gpu_type"]
        peak_fp16 = num(cat[gtype]["peak_tflops_fp16"])
        peak_bw = num(cat[gtype]["peak_bw_tbs"])
        mfu = metrics.compute_mfu(num(r["achieved_tflops"]), peak_fp16)
        mbu = metrics.compute_mbu(num(r["achieved_bw_tbs"]), peak_bw)
        a = agg[r["gpu_id"]]
        a["type"] = gtype
        a["util"].append(num(r["gpu_util_pct"]))
        a["mfu"].append(mfu)
        a["mbu"].append(mbu)
        if num(r["gpu_util_pct"]) < 10:  # effectively idle this interval (1h)
            a["idle_hours"] += 1

    summary = []
    for gid, a in agg.items():
        summary.append({
            "gpu_id": gid, "gpu_type": a["type"],
            "gpu_util_pct": round(sum(a["util"]) / len(a["util"]), 1),
            "mfu": round(sum(a["mfu"]) / len(a["mfu"]), 3),
            "mbu": round(sum(a["mbu"]) / len(a["mbu"]), 3),
            "idle_hours": a["idle_hours"],
        })

    lies = metrics.flag_util_lies(summary)
    idle_waste = 0.0
    for s in summary:
        on_demand = num(catalog_by_type()[s["gpu_type"]]["on_demand_hr"])
        idle_waste += metrics.idle_waste_usd(s["idle_hours"], on_demand)

    rightsize, rightsize_monthly = _rightsize_memory_bound(tel, cat)

    if verbose:
        print("== M1 Efficiency Audit ==")
        print(f"{'GPU':14}{'type':7}{'util%':>7}{'MFU':>7}{'MBU':>7}{'idle_h':>8}")
        for s in sorted(summary, key=lambda x: x["mfu"]):
            print(f"{s['gpu_id']:14}{s['gpu_type']:7}{s['gpu_util_pct']:>7}{s['mfu']:>7}{s['mbu']:>7}{s['idle_hours']:>8}")
        print(f"\nGPU-Util LIES (util>=90% but MFU<30%): {[l['gpu_id'] for l in lies]}")
        print(f"Idle waste (1 day): ${idle_waste:,.2f}  ->  ${idle_waste*30:,.0f}/month")

        # Extension 2 output
        print("\n-- Extension 2: MBU right-sizing for memory-bound (infer/embed) GPUs --")
        print(f"{'GPU':14}{'type':7}{'MBU':>6}{'need_BW':>9}{'$/TBps':>8}   recommendation")
        for r in rightsize:
            c = r["candidate"]
            if c:
                rec = (f"{c['gpu_type']} (${c['on_demand_hr']}/h, {c['peak_bw_tbs']} TB/s) "
                       f"saves ${r['monthly_savings']:,.0f}/mo")
            else:
                rec = "already optimal (no cheaper GPU meets BW+VRAM)"
            print(f"{r['gpu_id']:14}{r['gpu_type']:7}{r['mbu']:>6}{r['need_bw_tbs']:>9}{r['cur_dollars_per_tbps']:>8}   {rec}")
        print(f"\nRight-size all memory-bound GPUs: ${rightsize_monthly:,.0f}/month")

    return {"summary": summary, "lies": lies, "idle_waste_daily": round(idle_waste, 2),
            "rightsize": rightsize, "rightsize_monthly": rightsize_monthly}


if __name__ == "__main__":
    run()
