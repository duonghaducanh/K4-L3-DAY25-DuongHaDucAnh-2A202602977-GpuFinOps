"""M2 — Inference Cost Levers: $/1M-token, batch x cache x cascade (deck §7).

Run: python missions/m2_inference_levers.py
"""
from __future__ import annotations
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from collections import Counter
from missions._common import load_csv, num
from finops import pricing, sustainability

# $/1M tokens (input, output) — illustrative 2026.
MODEL_PRICES = {"small": (0.20, 0.40), "large": (3.00, 15.00)}
# Cache storage charge ($/1M-token held) — Extension 3 needs a write cost to test.
CACHE_STORAGE_PER_M = 0.50


def _avg_cache_reads(rows) -> dict:
    """Extension 3 input: average number of times a cached prefix is read back.

    We have no explicit prefix ids in the telemetry, so we proxy a "prefix" by
    (team, route_tier) — a team's shared system prompt at a given model tier —
    and count the requests that hit it. That count is the number of reads that
    must amortise the one-time cache write.
    """
    hits = Counter()
    for r in rows:
        if int(num(r["cached_input_tokens"])) > 0:
            hits[(r["team"], r["route_tier"])] += 1
    by_tier = {}
    for tier in MODEL_PRICES:
        reads = [n for (team, t), n in hits.items() if t == tier]
        by_tier[tier] = round(sum(reads) / len(reads), 1) if reads else 0.0
    return by_tier


def _reasoning_budget(rows) -> dict:
    """Extension 4 — split $ and Wh for reasoning vs non-reasoning traffic.

    Reasoning requests are ~6x the output tokens of a normal request and the
    energy model multiplies their Wh by ~80x, so a small share of requests can
    dominate the energy bill even when it is a minority of the dollar bill.
    """
    r_cost = n_cost = r_wh = n_wh = 0.0
    r_req = n_req = 0
    r_out = n_out = 0
    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        pin, pout = MODEL_PRICES[r["route_tier"]]
        c = pricing.request_cost(inp, out, pin, pout,
                                 cached_in=int(num(r["cached_input_tokens"])),
                                 batch=bool(int(num(r["is_batch"]))))
        wh = sustainability.wh_per_query(inp + out, is_reasoning=bool(int(num(r["is_reasoning"]))))
        if bool(int(num(r["is_reasoning"]))):
            r_cost += c; r_wh += wh; r_req += 1; r_out += out
        else:
            n_cost += c; n_wh += wh; n_req += 1; n_out += out
    total_cost = r_cost + n_cost
    total_wh = r_wh + n_wh
    total_req = r_req + n_req
    return {
        "reasoning_reqs": r_req, "total_reqs": total_req,
        "reasoning_req_share": round(r_req / total_req, 4) if total_req else 0.0,
        "reasoning_output_share": round(r_out / (r_out + n_out), 4) if (r_out + n_out) else 0.0,
        "reasoning_cost": round(r_cost, 2), "non_reasoning_cost": round(n_cost, 2),
        "reasoning_cost_share": round(r_cost / total_cost, 4) if total_cost else 0.0,
        "reasoning_wh": round(r_wh, 1), "non_reasoning_wh": round(n_wh, 1),
        "reasoning_wh_share": round(r_wh / total_wh, 4) if total_wh else 0.0,
        "wh_per_reasoning_req": round(r_wh / r_req, 1) if r_req else 0.0,
        "wh_per_plain_req": round(n_wh / n_req, 2) if n_req else 0.0,
    }


def run(verbose: bool = True) -> dict:
    rows = load_csv("token_usage.csv")
    base_cost = opt_cost = 0.0
    total_tokens = 0
    cache_reads = _avg_cache_reads(rows)
    # Extension 3 — only credit cache savings for a tier when the cache pays for
    # itself; otherwise the "optimized" number would book savings we never earn.
    cache_worth = {}
    for tier, (pin, pout) in MODEL_PRICES.items():
        write_cost = pricing.CACHE_WRITE_MULTIPLIER * pin + CACHE_STORAGE_PER_M
        cache_worth[tier] = pricing.cache_is_worth_it(
            cache_reads[tier], write_cost_per_m=write_cost, price_in_per_m=pin)

    for r in rows:
        inp, out = int(num(r["input_tokens"])), int(num(r["output_tokens"]))
        cached = int(num(r["cached_input_tokens"]))
        is_batch = bool(int(num(r["is_batch"])))
        tier = r["route_tier"]
        total_tokens += inp + out
        # BASELINE: naive deployment — everything on the large model, no cache, no batch
        lin, lout = MODEL_PRICES["large"]
        base_cost += pricing.request_cost(inp, out, lin, lout)
        # OPTIMIZED: cascade (route_tier), prompt caching, batch API
        pin, pout = MODEL_PRICES[tier]
        # Extension 3 gate: drop the cache credit when it does not pay for itself
        credit_cache = cached if cache_worth[tier] else 0
        opt_cost += pricing.request_cost(inp, out, pin, pout, cached_in=credit_cache, batch=is_batch)

    base_pm = pricing.dollars_per_million(base_cost, total_tokens)
    opt_pm = pricing.dollars_per_million(opt_cost, total_tokens)
    savings_pct = (1 - opt_cost / base_cost) * 100 if base_cost else 0.0

    reasoning = _reasoning_budget(rows)

    if verbose:
        print("== M2 Inference Cost Levers ==")
        print(f"requests={len(rows)}  tokens={total_tokens:,}")
        print(f"baseline  : ${base_cost:,.2f}/day   ${base_pm:.3f}/1M-token")
        print(f"optimized : ${opt_cost:,.2f}/day   ${opt_pm:.3f}/1M-token")
        print(f"savings   : {savings_pct:.1f}%  (cascade + caching + batch)")
        print(f"discount stack (batch + 100% cache): {pricing.discount_stack(batch=True, cache_hit_frac=1.0):.3f} of naive")

        print("\n-- Extension 3: is the cache worth it? (reads needed to amortise the write) --")
        for tier, (pin, pout) in MODEL_PRICES.items():
            write_cost = pricing.CACHE_WRITE_MULTIPLIER * pin + CACHE_STORAGE_PER_M
            be = pricing.cache_break_even_reads(write_cost, pin)
            print(f"  {tier:6} avg_reads={cache_reads[tier]:>6}  break_even_reads={be:>5.2f}  "
                  f"cache_worth_it={cache_worth[tier]}")

        print("\n-- Extension 4: reasoning budget ($ and Wh) --")
        print(f"  reasoning = {reasoning['reasoning_req_share']:.1%} of requests, "
              f"{reasoning['reasoning_output_share']:.1%} of output tokens")
        print(f"  reasoning = {reasoning['reasoning_cost_share']:.1%} of $ cost, "
              f"{reasoning['reasoning_wh_share']:.1%} of energy (Wh)")
        print(f"  Wh/req: reasoning {reasoning['wh_per_reasoning_req']:.1f} vs plain "
              f"{reasoning['wh_per_plain_req']:.2f}  "
              f"(~{reasoning['wh_per_reasoning_req']/max(reasoning['wh_per_plain_req'],1e-9):.0f}x)")

    return {
        "baseline_daily": round(base_cost, 2), "optimized_daily": round(opt_cost, 2),
        "baseline_per_m": round(base_pm, 3), "optimized_per_m": round(opt_pm, 3),
        "savings_pct": round(savings_pct, 1), "total_tokens": total_tokens,
        "cache_reads": cache_reads, "cache_worth": cache_worth, "reasoning": reasoning,
    }


if __name__ == "__main__":
    run()
