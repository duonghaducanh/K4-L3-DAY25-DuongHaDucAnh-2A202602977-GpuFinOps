# NimbusAI — GPU Cost Optimization Report

**Period:** monthly  
**Baseline spend:** $27,133  
**Optimized spend:** $13,747  
**Projected savings:** $13,386  (**49%**)

## Savings by lever

| Lever | Savings (USD) | Share of savings |
|---|---|---|
| Inference (cascade/cache/batch) | $1,212 | 9% |
| Purchasing (spot/reserved) | $10,919 | 82% |
| Right-size util-lies | $655 | 5% |
| Kill idle GPUs | $600 | 4% |

## Unit economics — measured in $/1M-token

| Metric | Baseline | Optimized | Change |
|---|---|---|---|
| $/1M-token (inference) | $6.488 | $1.126 | 83% lower |
| Inference $/day | $48.87 | $8.48 | 83% lower |
| Total tokens served (day) | 7,533,027 | 7,533,027 | unchanged |

_$ /GPU-hour tells you what you paid; $ /1M-token tells you what you got._

## The GPU-Util lie (why utilization is not efficiency)

| GPU | GPU-Util % | MFU | Monthly $ at risk |
|---|---|---|---|
| gpu-h100-4 | 98.2 | 0.19 | $511 |
| gpu-a10g-1 | 96.9 | 0.27 | $144 |

`nvidia-smi` reports the fraction of time a clock is active, not the FLOPs actually produced. A GPU stalled on memory or waiting on kernel launch reads ~100% util while its MFU stays near 0.20 — you are billed the full H100 hour for roughly a fifth of the compute.

## Sustainability

- Energy per query: 0.24 Wh
- Carbon per query: 0.091 gCO2e
- Cheapest+cleanest region: europe-north1
- Moving interruptible jobs to europe-north1 saves 626,150 gCO2e (92%) and $54 per run

### Region comparison ($ vs carbon)

| Region | $/kWh | gCO2/kWh | Cheapest | Cleanest |
|---|---|---|---|---|
| us-east-1 | $0.120 | 380 |  |  |
| us-west-2 | $0.070 | 120 |  |  |
| europe-north1 | $0.090 | 30 |  | ✓ |
| europe-central2 | $0.180 | 660 |  |  |
| us-east-wa | $0.055 | 90 | ✓ |  |

## Prioritized actions

1. **Purchasing (spot/reserved)** is the single largest lever ($10,919/mo). Commit the steady high-duty jobs to reserved/spot first — it is pure contractual savings with no engineering work.
2. **Right-size the util-lie GPUs** (gpu-h100-4, gpu-a10g-1). They bill full rate while producing ~20% of peak FLOPs; move the memory-bound ones to the cheapest GPU that still clears their bandwidth and VRAM needs ($288/mo).
3. **Reclaim idle GPUs** (gpu-h100-5 is dark 8h/night). Add auto-shutdown on idle detection — $600/mo for a trivial scheduler change.
4. **Keep cascade + caching + batch on** ($1,212/mo). Caching only pays because prefixes are read back 476+ times; re-check the break-even before shipping a new cached prompt.
5. **Move interruptible jobs to europe-north1** — saves 626,150 gCO2e and $54 of electricity with zero latency impact (no user-facing SLO).

## Extensions implemented

- **Ext 1 — purchasing policy v2:** prices spot interruption risk per GPU type and only counts a reserved term the job can amortise. M3 savings 39.1% → 42.5%.
- **Ext 2 — MBU right-sizing:** cheapest adequate GPU by $/TB/s for memory-bound work; $288/mo identified.
- **Ext 3 — cache_is_worth_it():** small-tier prefixes read 476x (break-even 4.2) and large-tier 124x (break-even 1.6) — both cleared, so the cache credit is real, not booked fiction.
- **Ext 4 — reasoning budget:** reasoning is 8.4% of requests but 94% of energy; cap it by routing rule.
- **Ext 5 — carbon-aware scheduling:** interruptible jobs → europe-north1, 92% carbon cut.

_Figures are June-2026 as-of snapshots; re-baseline before acting._