"""Report assembly — the lab's deliverable: baseline vs optimized + savings chart.

`build_report` stays backward compatible (the unit test calls it with just
baseline / optimized / levers) but accepts optional sections so Mission 5 can
emit a full FinOps deliverable: unit economics ($/1M-token), the GPU-Util-lie
finding, a prioritized action list, extensions, and the region/carbon table.
"""
from __future__ import annotations


def build_report(baseline_usd: float, optimized_usd: float, levers: dict,
                 sustainability: dict | None = None, period: str = "monthly",
                 unit_economics: dict | None = None, util_lies: list | None = None,
                 insights: list | None = None, extensions: list | None = None,
                 region_table: list | None = None) -> str:
    """Return a markdown cost-optimization report."""
    savings = baseline_usd - optimized_usd
    pct = (savings / baseline_usd * 100.0) if baseline_usd > 0 else 0.0
    lines = [
        "# NimbusAI — GPU Cost Optimization Report",
        "",
        f"**Period:** {period}  ",
        f"**Baseline spend:** ${baseline_usd:,.0f}  ",
        f"**Optimized spend:** ${optimized_usd:,.0f}  ",
        f"**Projected savings:** ${savings:,.0f}  (**{pct:.0f}%**)",
        "",
        "## Savings by lever",
        "",
        "| Lever | Savings (USD) | Share of savings |",
        "|---|---|---|",
    ]
    for name, amount in levers.items():
        share = (amount / savings * 100.0) if savings else 0.0
        lines.append(f"| {name} | ${amount:,.0f} | {share:.0f}% |")

    if unit_economics:
        lines += [
            "",
            "## Unit economics — measured in $/1M-token",
            "",
            "| Metric | Baseline | Optimized | Change |",
            "|---|---|---|---|",
            f"| $/1M-token (inference) | ${unit_economics.get('baseline_per_m', 0):.3f} "
            f"| ${unit_economics.get('optimized_per_m', 0):.3f} "
            f"| {unit_economics.get('per_m_reduction_pct', 0):.0f}% lower |",
            f"| Inference $/day | ${unit_economics.get('baseline_daily', 0):,.2f} "
            f"| ${unit_economics.get('optimized_daily', 0):,.2f} "
            f"| {unit_economics.get('daily_reduction_pct', 0):.0f}% lower |",
            f"| Total tokens served (day) | {unit_economics.get('total_tokens', 0):,} "
            f"| {unit_economics.get('total_tokens', 0):,} | unchanged |",
            "",
            "_$ /GPU-hour tells you what you paid; $ /1M-token tells you what you got._",
        ]

    if util_lies:
        lines += [
            "",
            "## The GPU-Util lie (why utilization is not efficiency)",
            "",
            "| GPU | GPU-Util % | MFU | Monthly $ at risk |",
            "|---|---|---|---|",
        ]
        for lie in util_lies:
            lines.append(f"| {lie.get('gpu_id', '?')} | {lie.get('gpu_util_pct', 0):.1f} "
                         f"| {lie.get('mfu', 0):.2f} | ${lie.get('monthly_at_risk', 0):,.0f} |")
        lines += [
            "",
            "`nvidia-smi` reports the fraction of time a clock is active, not the FLOPs "
            "actually produced. A GPU stalled on memory or waiting on kernel launch reads "
            "~100% util while its MFU stays near 0.20 — you are billed the full H100 hour "
            "for roughly a fifth of the compute.",
        ]

    if sustainability:
        lines += [
            "",
            "## Sustainability",
            "",
            f"- Energy per query: {sustainability.get('wh_per_query', 0):.2f} Wh",
            f"- Carbon per query: {sustainability.get('carbon_g', 0):.3f} gCO2e",
            f"- Cheapest+cleanest region: {sustainability.get('best_region', 'n/a')}",
        ]
        if sustainability.get("carbon_saved_g"):
            lines += [
                f"- Moving interruptible jobs to {sustainability.get('clean_region', 'n/a')} saves "
                f"{sustainability.get('carbon_saved_g', 0):,} gCO2e "
                f"({sustainability.get('carbon_saved_pct', 0):.0f}%) and "
                f"${sustainability.get('energy_cost_saved', 0):,.0f} per run",
            ]

    if region_table:
        lines += [
            "",
            "### Region comparison ($ vs carbon)",
            "",
            "| Region | $/kWh | gCO2/kWh | Cheapest | Cleanest |",
            "|---|---|---|---|---|",
        ]
        cheapest = min(region_table, key=lambda r: r["price_kwh"])["region"]
        cleanest = min(region_table, key=lambda r: r["carbon_g_kwh"])["region"]
        for r in region_table:
            lines.append(f"| {r['region']} | ${r['price_kwh']:.3f} | {r['carbon_g_kwh']} "
                         f"| {'✓' if r['region'] == cheapest else ''} "
                         f"| {'✓' if r['region'] == cleanest else ''} |")

    if insights:
        lines += ["", "## Prioritized actions", ""]
        for i, ins in enumerate(insights, 1):
            lines.append(f"{i}. {ins}")

    if extensions:
        lines += ["", "## Extensions implemented", ""]
        for ext in extensions:
            lines.append(f"- {ext}")

    lines += ["", "_Figures are June-2026 as-of snapshots; re-baseline before acting._"]
    return "\n".join(lines)


def savings_waterfall(levers: dict, path: str) -> str:
    """Write a savings waterfall (baseline -> each lever -> optimized). Returns the path.

    Rendered as a true waterfall so the reader sees each lever *reduce* the bill,
    ending at the optimized spend — not just four disconnected bars. No-op if
    matplotlib is absent.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return ""
    names = list(levers.keys())
    vals = [levers[n] for n in names]
    baseline = sum(vals)
    fig, ax = plt.subplots(figsize=(9, 4.8))

    running = baseline
    positions = []
    labels = ["Baseline"]
    # baseline bar
    ax.bar(0, baseline, color="#7a1f2b")
    positions.append(baseline)
    for i, (n, v) in enumerate(zip(names, vals), start=1):
        ax.bar(i, -v, bottom=running, color="#2e548a")
        labels.append(n)
        running -= v
    # optimized bar
    ax.bar(len(names) + 1, running, color="#1f7a4d")
    labels.append("Optimized")

    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("USD / month")
    ax.set_title("GPU cost waterfall: baseline → levers → optimized")
    for i, v in enumerate(vals, start=1):
        ax.annotate(f"-${v:,.0f}", (i, baseline - sum(vals[:i - 1]) - v / 2),
                    ha="center", va="center", color="white", fontsize=8)
    plt.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path
