"""Pricing & purchasing economics — measure in $/1M-token, not $/GPU-hr.

Figures are June-2026 as-of snapshots from the deck's RESEARCH dossier; treat
live prices as fast-moving (re-baseline before each cohort).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Extension 1 — purchasing policy v2
# ---------------------------------------------------------------------------
# Per-GPU spot interruption probability (per hour). Smaller/older parts get
# reclaimed far more often than the flagship H100/H200, so a single global
# interrupt rate (the v1 default of 5%/h) misprices spot for both ends.
INTERRUPTION_RATE = {
    "H100": 0.03, "H200": 0.04, "A100": 0.05,
    "A10G": 0.09, "L4": 0.11, "B200": 0.06, "MI300X": 0.05,
}
DEFAULT_INTERRUPTION_RATE = 0.05


def effective_spot_hour(
    spot_hr: float,
    interrupt_rate: float = DEFAULT_INTERRUPTION_RATE,
    ckpt_overhead_frac: float = 0.03,
    rework_hours_per_interrupt: float = 0.5,
) -> float:
    """Risk-adjusted $/GPU-hr of spot: the headline spot price is *not* the true cost.

    Expected rework from interruptions (rate x rework_hours) plus the steady
    checkpoint-writing overhead both inflate the effective hourly cost. Only the
    effective number is comparable to reserved/on-demand rates.
    """
    spot_hr = max(0.0, spot_hr)
    return spot_hr * (1.0 + ckpt_overhead_frac + max(0.0, interrupt_rate) * rework_hours_per_interrupt)


def reserved_term_hour(reserved_1yr_hr: float, reserved_3yr_hr: float, job_days: float | None = None) -> tuple:
    """Return (term, $/hr) for the cheapest *usable* reserved commitment.

    A 3yr commitment only pays off if the job actually runs long enough to
    amortise it. For short jobs (job_days < 365) the 1yr rate is the honest
    comparison even though it is priced higher — you cannot commit 3 years to a
    22-day sandbox.
    """
    r1 = max(0.0, reserved_1yr_hr)
    r3 = max(0.0, reserved_3yr_hr)
    if job_days is not None and job_days < 365:
        return ("1yr", r1)
    return ("3yr", r3) if r3 <= r1 else ("1yr", r1)


def recommend_tier_v2(
    hours_per_day: float,
    interruptible: bool,
    on_demand_hr: float,
    spot_hr: float,
    reserved_1yr_hr: float,
    reserved_3yr_hr: float,
    gpu_type: str | None = None,
    job_days: float | None = None,
    reserved_discount: float = 0.45,
) -> dict:
    """Cost-minimising purchasing policy (Extension 1).

    v1 chose a tier by *rule of thumb* (interruptible -> spot, duty >= break-even
    -> reserved) and never compared the actual dollar cost of the surviving
    options. v2 builds the candidate set and picks the cheapest by effective
    $/GPU-hr, which fixes two real mispricings:

      * a high-duty-cycle job that is *technically* interruptible is often
        cheaper on reserved once spot's risk premium is priced in;
      * small/older GPUs (L4, A10G) are reclaimed so often that spot's effective
        rate can exceed the reserved rate even for interruptible work.

    Returns the chosen tier plus the effective rate of every candidate, so the
    decision is auditable rather than a black box.
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    rate = INTERRUPTION_RATE.get(gpu_type, DEFAULT_INTERRUPTION_RATE) if gpu_type else DEFAULT_INTERRUPTION_RATE

    candidates: dict = {"on_demand": max(0.0, on_demand_hr)}
    if interruptible and hours_per_day < 24:
        candidates["spot"] = effective_spot_hour(spot_hr, rate)
    if duty >= be:
        term, rhr = reserved_term_hour(reserved_1yr_hr, reserved_3yr_hr, job_days)
        candidates["reserved"] = rhr
    else:
        term = None

    tier = min(candidates, key=candidates.get)
    chosen = candidates[tier]
    od = max(0.0, on_demand_hr)
    return {
        "tier": tier,
        "term": term if tier == "reserved" else None,
        "interrupt_rate": round(rate, 3),
        "effective_hr": round(chosen, 4),
        "on_demand_hr": round(od, 4),
        "candidates": {k: round(v, 4) for k, v in candidates.items()},
        "savings_vs_on_demand_pct": round((1.0 - chosen / od) * 100.0, 1) if od > 0 else 0.0,
    }


def tier_matrix(catalog: dict, gpu_types=None, duty_cycles=(0.25, 0.55, 0.85, 1.0),
                interruptible: bool = True, job_days: float = 30) -> list:
    """Build a recommendation matrix: GPU type x duty cycle -> chosen tier.

    `catalog` maps gpu_type -> row dict with on_demand_hr / spot_hr /
    reserved_1yr_hr / reserved_3yr_hr. Demonstrates that the "right" tier is a
    function of both the hardware and the duty cycle, not one global rule.
    """
    out = []
    types = gpu_types or list(catalog.keys())
    for gtype in types:
        c = catalog[gtype]
        for duty in duty_cycles:
            rec = recommend_tier_v2(
                hours_per_day=duty * 24.0,
                interruptible=interruptible,
                on_demand_hr=float(c["on_demand_hr"]),
                spot_hr=float(c["spot_hr"]),
                reserved_1yr_hr=float(c["reserved_1yr_hr"]),
                reserved_3yr_hr=float(c["reserved_3yr_hr"]),
                gpu_type=gtype,
                job_days=job_days,
            )
            out.append({"gpu_type": gtype, "duty": round(duty, 2), **rec})
    return out


# ---------------------------------------------------------------------------
# Extension 3 — cache economics
# ---------------------------------------------------------------------------
# A prompt cache only pays for itself when a prefix is read back often enough to
# amortise the cache-write premium. Below the break-even read count the "cache
# savings" line in a FinOps deck is fiction.
CACHE_WRITE_MULTIPLIER = 1.25   # Anthropic-style: cache write bills ~1.25x input
CACHE_READ_DISCOUNT = 0.10      # cached read bills ~0.10x input (-90%)


def cache_break_even_reads(
    write_cost_per_m: float,
    price_in_per_m: float,
    read_discount: float = CACHE_READ_DISCOUNT,
) -> float:
    """Number of cached reads needed before caching beats not caching a prefix.

    You pay `write_cost_per_m` once to write the prefix; every subsequent read
    saves `(1 - read_discount) * price_in_per_m`. Break-even is where the
    cumulative read saving equals the write premium.
    """
    per_read_saving = (1.0 - read_discount) * max(0.0, price_in_per_m)
    if per_read_saving <= 0:
        return float("inf")
    return max(0.0, write_cost_per_m) / per_read_saving


def cache_is_worth_it(
    avg_cache_reads: float,
    write_cost_per_m: float = None,
    read_discount: float = CACHE_READ_DISCOUNT,
    price_in_per_m: float = None,
) -> bool:
    """True when a cached prefix is read back enough times to pay for its write.

    Defaults model the common case: writing the cache costs 1.25x the input
    price, reads cost 0.10x. Returns False for degenerate inputs (no reads, no
    write cost modelled, free reads).
    """
    if avg_cache_reads is None or avg_cache_reads < 1:
        return False
    if price_in_per_m is None:
        price_in_per_m = 1.0  # normalised unit price
    if write_cost_per_m is None:
        write_cost_per_m = CACHE_WRITE_MULTIPLIER * price_in_per_m
    be = cache_break_even_reads(write_cost_per_m, price_in_per_m, read_discount)
    return avg_cache_reads >= be


def request_cost(
    input_tok: int,
    output_tok: int,
    price_in_per_m: float,
    price_out_per_m: float,
    cached_in: int = 0,
    cache_discount: float = 0.10,   # Anthropic cached-read ~0.1x (=-90%)
    batch: bool = False,
    batch_discount: float = 0.50,   # Batch API ~ -50%
) -> float:
    """USD cost of a single request. Cached input billed at cache_discount x price."""
    cached_in = min(max(0, cached_in), input_tok)
    uncached_in = input_tok - cached_in
    cost = (
        (uncached_in / 1e6) * price_in_per_m
        + (cached_in / 1e6) * price_in_per_m * cache_discount
        + (output_tok / 1e6) * price_out_per_m
    )
    if batch:
        cost *= batch_discount
    return cost


def dollars_per_million(total_cost_usd: float, total_tokens: int) -> float:
    """Aggregate unit economics: $ per 1,000,000 tokens served."""
    if total_tokens <= 0:
        return 0.0
    return total_cost_usd / (total_tokens / 1e6)


def discount_stack(
    batch: bool = False,
    cache_hit_frac: float = 0.0,
    batch_discount: float = 0.50,
    cache_discount: float = 0.10,
) -> float:
    """Effective fraction of the naive bill after stacking discounts (input-heavy view).

    Discounts MULTIPLY: cache applies to the cached share of input, batch to the
    whole bill. batch + 100% cache-hit -> 0.5 * 0.1 = 0.05 (~95% off).
    """
    cache_mult = cache_hit_frac * cache_discount + (1.0 - cache_hit_frac)
    batch_mult = batch_discount if batch else 1.0
    return cache_mult * batch_mult


def break_even_utilization(discount_frac: float) -> float:
    """Utilization at which a commitment pays off ~= 1 - discount.

    A 45% reserved discount needs ~55% utilization (~13.2h/day) to beat on-demand.
    """
    return max(0.0, min(1.0, 1.0 - discount_frac))


def recommend_tier(hours_per_day: float, interruptible: bool, reserved_discount: float = 0.45) -> str:
    """Pick a purchasing tier from a workload's duty cycle + interruptibility.

    DOCUMENTED simple policy (instructor extension point — swap in your own):
      - interruptible & not 24/7  -> 'spot'      (checkpoint and ride the discount)
      - duty cycle >= break-even  -> 'reserved'  (steady, high utilization)
      - otherwise                 -> 'on_demand' (spiky / low duty)
    """
    duty = max(0.0, hours_per_day) / 24.0
    be = break_even_utilization(reserved_discount)
    if interruptible and hours_per_day < 24:
        return "spot"
    if duty >= be:
        return "reserved"
    return "on_demand"


def spot_checkpoint_cost(
    job_hours: float,
    spot_hr: float,
    on_demand_hr: float,
    interrupt_rate: float = 0.05,      # per-hour chance (H100 spot ~<5%)
    ckpt_overhead_frac: float = 0.03,  # steady cost of writing checkpoints
    rework_hours_per_interrupt: float = 0.5,
) -> dict:
    """Effective cost of running a checkpointable job on spot vs on-demand.

    Interruptions waste the compute since the last checkpoint (rework); checkpointing
    adds a small steady overhead. Spot still wins for interruptible jobs.
    """
    expected_interrupts = job_hours * interrupt_rate
    rework_hours = expected_interrupts * rework_hours_per_interrupt
    effective_hours = job_hours * (1.0 + ckpt_overhead_frac) + rework_hours
    spot_cost = effective_hours * spot_hr
    on_demand_cost = job_hours * on_demand_hr
    savings_pct = (1.0 - spot_cost / on_demand_cost) * 100.0 if on_demand_cost > 0 else 0.0
    return {
        "spot_effective_hours": round(effective_hours, 2),
        "spot_cost": round(spot_cost, 2),
        "on_demand_cost": round(on_demand_cost, 2),
        "savings_pct": round(savings_pct, 1),
    }
