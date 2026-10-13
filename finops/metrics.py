"""Efficiency metrics — the numbers that actually drive GPU cost.

Key teaching point (deck §5): nvidia-smi "GPU-Util %" is a *time-active* clock,
not an efficiency metric. A GPU can read 100% util while its MFU is ~20% — you
are paying the full GPU-hour for a fraction of the FLOPs you rented.
"""
from __future__ import annotations


def compute_mfu(achieved_tflops: float, peak_tflops: float) -> float:
    """Model FLOPs Utilization = achieved / peak (clamped to 0..1).

    Good training MFU is ~0.35-0.45; >0.50 is excellent. Returns 0 if peak<=0.
    """
    if peak_tflops <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_tflops / peak_tflops))


def compute_mbu(achieved_bw_tbs: float, peak_bw_tbs: float) -> float:
    """Model Bandwidth Utilization = achieved HBM BW / peak BW (clamped 0..1).

    The right metric for memory-bound decode; target ~0.60 on H100-80GB batch-1.
    """
    if peak_bw_tbs <= 0:
        return 0.0
    return max(0.0, min(1.0, achieved_bw_tbs / peak_bw_tbs))


def arithmetic_intensity(flops: float, bytes_moved: float) -> float:
    """FLOP / byte for a workload (the x-axis of the roofline model)."""
    if bytes_moved <= 0:
        return 0.0
    return flops / bytes_moved


def roofline_regime(intensity: float, ridge_point: float) -> str:
    """Below the ridge point a workload is memory-bound; at/above it is compute-bound.

    H100 ridge ~295 FLOP/byte (BF16). LLM decode (~1-2) is memory-bound; prefill
    (~455) is compute-bound — which is *why* prefill/decode disaggregation pays off.
    """
    return "compute-bound" if intensity >= ridge_point else "memory-bound"


def flag_util_lies(rows, util_threshold: float = 0.90, mfu_threshold: float = 0.30):
    """Return the rows where GPU-Util is high but MFU is low — money leaking.

    `rows` is an iterable of dicts each having 'gpu_util_pct' (0-100) and 'mfu' (0-1).
    These are GPUs you are billed full-rate for while they do little real compute.
    """
    out = []
    for r in rows:
        util = float(r.get("gpu_util_pct", 0)) / 100.0
        mfu = float(r.get("mfu", 0))
        if util >= util_threshold and mfu < mfu_threshold:
            out.append(r)
    return out


def idle_waste_usd(idle_hours: float, on_demand_hr: float) -> float:
    """Dollars burned by a GPU left running idle (training done, instance up)."""
    return max(0.0, idle_hours) * max(0.0, on_demand_hr)


# ---------------------------------------------------------------------------
# Extension 2 — right-sizing by MBU / bandwidth, not by $/GPU-hr
# ---------------------------------------------------------------------------
def dollars_per_gb_vram(on_demand_hr: float, hbm_gb: float) -> float:
    """Rental $ per GB of HBM — capacity efficiency of a GPU type."""
    if hbm_gb <= 0:
        return float("inf")
    return max(0.0, on_demand_hr) / hbm_gb


def dollars_per_tbps(on_demand_hr: float, peak_bw_tbs: float) -> float:
    """Rental $ per TB/s of HBM bandwidth — the metric that matters for decode.

    Memory-bound inference is limited by how fast weights/KV move, so the GPU
    that costs least *per unit of bandwidth* is the efficient one, even if its
    sticker $/GPU-hr is higher. This is why "pick the cheapest GPU" is wrong.
    """
    if peak_bw_tbs <= 0:
        return float("inf")
    return max(0.0, on_demand_hr) / peak_bw_tbs


def rightsize_candidate(
    needed_bw_tbs: float,
    needed_vram_gb: float,
    catalog: dict,
    current_type: str | None = None,
    safety: float = 1.05,
) -> dict | None:
    """Cheapest GPU type that still satisfies a memory-bound workload's needs.

    `catalog` maps gpu_type -> dict with 'on_demand_hr', 'peak_bw_tbs', 'hbm_gb'.
    A candidate must deliver at least `safety` x the observed achieved bandwidth
    AND `safety` x the observed VRAM footprint — right-sizing may not degrade the
    workload. Returns None when no cheaper adequate GPU exists (already optimal),
    so the caller never downsizes into a slower device.
    """
    cur_hr = float(catalog.get(current_type, {}).get("on_demand_hr", float("inf"))) if current_type else float("inf")
    best = None
    for gtype, c in catalog.items():
        if gtype == current_type:
            continue
        if float(c["peak_bw_tbs"]) < needed_bw_tbs * safety:
            continue
        if float(c["hbm_gb"]) < needed_vram_gb * safety:
            continue
        hr = float(c["on_demand_hr"])
        if hr < cur_hr and (best is None or hr < best["on_demand_hr"]):
            best = {
                "gpu_type": gtype,
                "on_demand_hr": hr,
                "peak_bw_tbs": float(c["peak_bw_tbs"]),
                "hbm_gb": float(c["hbm_gb"]),
                "dollars_per_tbps": dollars_per_tbps(hr, float(c["peak_bw_tbs"])),
                "dollars_per_gb_vram": dollars_per_gb_vram(hr, float(c["hbm_gb"])),
            }
    return best
