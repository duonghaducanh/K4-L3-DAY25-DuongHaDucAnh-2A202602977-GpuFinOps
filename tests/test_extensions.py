"""Unit tests for the "Your Turn" extensions (student-authored).

These are NEW tests added for Extensions 1, 2, 3 and 5. They do not modify the
provided test files. Run with: pytest tests/test_extensions.py -q
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from finops import pricing, metrics


# --- Extension 1: purchasing policy v2 -------------------------------------
def test_effective_spot_hour_prices_risk():
    # risk premium raises the effective rate above the headline spot price
    eff = pricing.effective_spot_hour(1.00, interrupt_rate=0.10)
    assert eff > 1.00
    # zero risk + zero checkpoint overhead -> headline price
    assert pricing.effective_spot_hour(1.00, interrupt_rate=0.0, ckpt_overhead_frac=0.0) == 1.00


def test_recommend_tier_v2_spot_vs_reserved():
    # low duty, interruptible, cheap risk -> spot wins
    low = pricing.recommend_tier_v2(
        hours_per_day=4, interruptible=True, on_demand_hr=2.5, spot_hr=1.5,
        reserved_1yr_hr=2.0, reserved_3yr_hr=1.4, gpu_type="H100", job_days=30)
    assert low["tier"] == "spot"
    # 24/7 non-interruptible -> reserved
    full = pricing.recommend_tier_v2(
        hours_per_day=24, interruptible=False, on_demand_hr=2.5, spot_hr=1.5,
        reserved_1yr_hr=2.0, reserved_3yr_hr=1.4, gpu_type="H100", job_days=None)
    assert full["tier"] == "reserved"


def test_high_interruption_makes_spot_lose():
    # price the interruption risk into spot; when the risk-adjusted spot rate
    # exceeds the reserved rate, reserved must win even for interruptible work.
    rec = pricing.recommend_tier_v2(
        hours_per_day=20, interruptible=True, on_demand_hr=0.80, spot_hr=0.35,
        reserved_1yr_hr=0.30, reserved_3yr_hr=0.28, gpu_type="L4", job_days=None)
    assert rec["tier"] == "reserved"
    # the risk-adjusted spot rate is above the headline spot price ...
    assert rec["candidates"]["spot"] > 0.35
    # ... and above reserved, which is why reserved wins
    assert rec["candidates"]["spot"] > rec["candidates"]["reserved"]


def test_reserved_term_prefers_3yr_for_long_jobs():
    term, hr = pricing.reserved_term_hour(2.00, 1.40, job_days=None)
    assert term == "3yr" and hr == 1.40
    # a 22-day job cannot amortise a 3yr commitment -> 1yr rate used
    term, hr = pricing.reserved_term_hour(2.00, 1.40, job_days=22)
    assert term == "1yr" and hr == 2.00


# --- Extension 2: right-sizing by bandwidth / VRAM --------------------------
def test_dollars_per_tbps_and_candidate():
    assert metrics.dollars_per_tbps(2.0, 4.0) == 0.5
    catalog = {
        "H100": {"on_demand_hr": 2.5, "peak_bw_tbs": 3.35, "hbm_gb": 80},
        "A10G": {"on_demand_hr": 1.0, "peak_bw_tbs": 0.6, "hbm_gb": 24},
        "L4": {"on_demand_hr": 0.8, "peak_bw_tbs": 0.3, "hbm_gb": 24},
    }
    # need 0.18 TB/s and 20 GB: A10G is over-provisioned, L4 is adequate+cheaper
    cand = metrics.rightsize_candidate(0.18, 20, catalog, current_type="A10G")
    assert cand is not None and cand["gpu_type"] == "L4"
    # need more bandwidth than L4 can give -> no cheap adequate option from A10G
    assert metrics.rightsize_candidate(0.5, 20, catalog, current_type="A10G") is None


# --- Extension 3: cache economics -------------------------------------------
def test_cache_break_even_and_is_worth_it():
    # write 1.25x, read 0.10x -> need ~1.39 reads to break even at unit price
    be = pricing.cache_break_even_reads(write_cost_per_m=1.25, price_in_per_m=1.0)
    assert abs(be - 1.25 / 0.9) < 1e-9
    assert pricing.cache_is_worth_it(avg_cache_reads=10, write_cost_per_m=1.25, price_in_per_m=1.0) is True
    assert pricing.cache_is_worth_it(avg_cache_reads=0, write_cost_per_m=1.25, price_in_per_m=1.0) is False


# --- Extension 5: carbon-aware region choice --------------------------------
def test_cleanest_region_is_lowest_carbon():
    from finops import sustainability
    clean = min(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    assert clean == "europe-north1"
    dirty = max(sustainability.REGION_CARBON, key=sustainability.REGION_CARBON.get)
    assert sustainability.carbon_g(1000, clean) < sustainability.carbon_g(1000, dirty)
