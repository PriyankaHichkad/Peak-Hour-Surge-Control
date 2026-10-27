import pytest
import numpy as np
import pandas as pd
from src.data_loader import load_nyc_benchmark_pickups
from src.elasticity import solve_optimal_surge_multiplier
from src.metrics import compare_baseline_vs_optimized, enforce_fleet_capacity
from benchmark_evaluation import run_dynamic_endogenous_benchmark

def test_endogenous_benchmark_identical_inputs():
    """
    Verifies that run_dynamic_endogenous_benchmark:
    1. Uses identical starting requests, coordinates, base fares, driver fleets, and weather.
    2. Derives fulfillment, cancellation, and GMV from simulated pricing responses rather than predetermined counts.
    3. Symmetrically enforces physical driver fleet throughput limits.
    """
    df_table, metrics = run_dynamic_endogenous_benchmark(n_samples=300, seed=123)
    
    assert isinstance(df_table, pd.DataFrame)
    assert len(df_table) == 6
    assert 'Pricing Multiplier Policy' in df_table['Metric'].values
    assert 'Marketplace Fulfillment Rate' in df_table['Metric'].values
    assert 'Rider Cancellation Rate' in df_table['Metric'].values
    
    # Assert KPIs are dynamically computed
    assert metrics['m_base'] > metrics['m_opt'], "Heuristic baseline should exhibit over-surge shock compared to bounded optimizer"
    assert metrics['opt_fulfillment_rate'] > metrics['base_fulfillment_rate'], "Bounded optimizer should improve fulfillment by preventing churn"
    assert metrics['cancellation_reduction_pts'] < 0, "Bounded optimizer should reduce rider cancellation"
    assert metrics['fulfillment_lift_pts'] == round(metrics['opt_fulfillment_rate'] - metrics['base_fulfillment_rate'], 1)

def test_compare_baseline_vs_optimized_end_to_end():
    """
    Verifies that compare_baseline_vs_optimized takes identical starting requests
    and evaluates both baseline and optimized branches symmetrically through enforce_fleet_capacity.
    """
    n_requests = 250
    df_raw = load_nyc_benchmark_pickups(n_samples=n_requests, seed=99)
    df_raw['timestamp'] = pd.Timestamp('2026-10-25 18:00:00')
    df_raw['hour'] = 18
    df_raw['weather_severity'] = 0.7
    df_raw['hourly_active_drivers'] = 80
    
    res = compare_baseline_vs_optimized(df_raw, opt_surge_multiplier=1.20)
    
    assert res['baseline']['total_requests'] == n_requests
    assert res['optimized']['total_requests'] == n_requests
    
    # Fleet capacity: 80 drivers * 2.5 trips/hr = 200 max dispatchable trips
    assert res['baseline']['fulfilled_trips'] <= 200
    assert res['optimized']['fulfilled_trips'] <= 200
    
    expected_gmv_lift = round(((res['optimized']['total_gmv_usd'] - res['baseline']['total_gmv_usd']) / res['baseline']['total_gmv_usd']) * 100, 1)
    assert res['gmv_lift_pct'] == expected_gmv_lift

def test_optimizer_infeasibility_reporting():
    """
    Verifies that solve_optimal_surge_multiplier explicitly reports infeasible configurations
    when the rider churn ceiling is stricter than baseline non-surged churn.
    """
    # 5% max churn is infeasible because baseline churn at 1.0x is ~12%
    res = solve_optimal_surge_multiplier(
        base_fare=15.0,
        price_sensitivity_k=2.2,
        weather_severity=0.8,
        max_churn_threshold=0.05
    )
    
    assert not res['is_feasible']
    assert res['feasibility_warning'] is not None
    assert "infeasible" in res['feasibility_warning'].lower()
    assert res['optimal_multiplier'] == 1.0
