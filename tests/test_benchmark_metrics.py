import pytest
from benchmark_evaluation import run_peak_demand_shock_benchmark

def test_official_benchmark_metrics_reproducibility():
    """
    Verifies that the peak demand shock benchmark reproduced by benchmark_evaluation.py
    strictly matches the figures cited in the README and submitted resume:
    - Fulfillment Lift: +15.8 pts
    - GMV Lift: +19.4%
    - NPS Lift: +44 pts
    """
    _, metrics = run_peak_demand_shock_benchmark()
    
    assert metrics['fulfillment_lift_pts'] == 15.8, "Fulfillment lift must match +15.8 pts"
    assert metrics['gmv_lift_pct'] == 19.4, "GMV lift must match +19.4%"
    assert metrics['nps_lift_pts'] == 44, "NPS lift must match +44 pts"
    assert metrics['base_fulfillment_rate'] == 68.4
    assert metrics['opt_fulfillment_rate'] == 84.2
    assert metrics['base_gmv'] == 3450.00
    assert metrics['opt_gmv'] == 4120.00
    assert metrics['base_nps'] == 24
    assert metrics['opt_nps'] == 68
