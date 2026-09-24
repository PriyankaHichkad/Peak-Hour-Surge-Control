import pytest
from src.elasticity import (
    rider_cancellation_probability,
    driver_acceptance_probability,
    solve_optimal_surge_multiplier
)

def test_rider_cancellation_probability():
    prob_low_surge = rider_cancellation_probability(1.0)
    prob_high_surge = rider_cancellation_probability(2.5)
    
    assert prob_low_surge < prob_high_surge
    assert 0.0 <= prob_low_surge <= 1.0
    assert 0.0 <= prob_high_surge <= 1.0

def test_driver_acceptance_probability():
    prob_low_surge = driver_acceptance_probability(1.0)
    prob_high_surge = driver_acceptance_probability(2.0)
    
    assert prob_low_surge < prob_high_surge

def test_solve_optimal_surge_multiplier():
    result = solve_optimal_surge_multiplier(base_fare=15.0, weather_severity=0.5, max_churn_threshold=0.35)
    
    assert 'optimal_multiplier' in result
    assert 'continuous_scipy_m' in result
    assert 1.0 <= result['optimal_multiplier'] <= 3.0
    assert 0.0 <= result['expected_rider_churn'] <= 0.35
