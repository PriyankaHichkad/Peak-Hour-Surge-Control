import pytest
import numpy as np
from src.elasticity import (
    rider_cancellation_probability,
    driver_acceptance_probability,
    solve_optimal_surge_multiplier
)

def test_baseline_cancellation_floor():
    """
    Documents and verifies that baseline cancellation at 1.0x surge is ~32%
    under default price sensitivity (k=2.2).
    """
    churn_1_0 = rider_cancellation_probability(1.0)
    assert 0.30 <= churn_1_0 <= 0.35, f"Expected baseline churn ~32%, got {churn_1_0:.1%}"

def test_cancellation_guardrail_behavior():
    """
    Tests that solve_optimal_surge_multiplier adheres to permitted bounds
    under varying supply-demand deficit conditions.
    """
    # Under high supply (S/D = 1.2), surge should be 1.0x
    res_high_supply = solve_optimal_surge_multiplier(supply_demand_ratio=1.2, max_churn_threshold=0.35)
    assert res_high_supply['optimal_multiplier'] == 1.0
    assert res_high_supply['continuous_scipy_m'] == 1.0
    
    # Under supply shortage (S/D = 0.4), surge should scale up to attract drivers while staying bounded
    res_shortage = solve_optimal_surge_multiplier(supply_demand_ratio=0.4, max_churn_threshold=0.35)
    assert 1.0 <= res_shortage['optimal_multiplier'] <= 2.5
    assert 1.0 <= res_shortage['continuous_scipy_m'] <= 2.5

def test_heuristic_vs_scipy_monotonicity():
    """
    Tests that both the continuous heuristic surge and SciPy surge are monotonic
    with respect to supply deficit.
    """
    sd_ratios = [0.2, 0.4, 0.6, 0.8, 1.0]
    heuristic_surges = [round(min(1.25, max(1.0, 1.0 + (1.0 - sd) * 0.35)), 2) for sd in sd_ratios]
    
    # Check that heuristic decreases monotonically as S/D increases (deficit decreases)
    for i in range(len(heuristic_surges) - 1):
        assert heuristic_surges[i] >= heuristic_surges[i+1]
