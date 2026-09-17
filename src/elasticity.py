import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

def rider_cancellation_probability(multiplier, price_sensitivity_k=2.0, inflection_m0=1.4, weather_severity=0.0):
    """
    Rider Churn Probability as a function of Surge Multiplier and Weather Urgency.
    Rain/storm reduces price sensitivity.
    """
    effective_k = price_sensitivity_k / (1.0 + 0.6 * weather_severity)
    prob = 1.0 / (1.0 + np.exp(-effective_k * (multiplier - inflection_m0)))
    return np.clip(prob, 0.02, 0.95)

def driver_acceptance_probability(multiplier, incentive_lambda=3.2):
    """
    Driver Acceptance Probability as a function of Surge Multiplier Bonus.
    """
    prob = 1.0 / (1.0 + np.exp(-incentive_lambda * (multiplier - 1.05)))
    return np.clip(prob, 0.25, 0.98)

def compute_marketplace_fulfillment_curve(multipliers, base_fare=15.0, price_sensitivity_k=2.0, weather_severity=0.0):
    """
    Evaluates Fulfillment Rate %, Gross Merchandise Value (GMV), and Churn % across a range of multipliers.
    """
    curve_data = []
    for m in multipliers:
        p_cancel = rider_cancellation_probability(m, price_sensitivity_k, weather_severity=weather_severity)
        p_accept = driver_acceptance_probability(m)
        
        conversion_rider = 1.0 - p_cancel
        fulfillment_rate = conversion_rider * p_accept
        
        effective_fare = base_fare * m
        expected_gmv_per_req = effective_fare * fulfillment_rate
        
        curve_data.append({
            'surge_multiplier': round(m, 2),
            'rider_cancel_prob': round(p_cancel, 4),
            'rider_conversion_rate': round(conversion_rider, 4),
            'driver_accept_prob': round(p_accept, 4),
            'fulfillment_rate': round(fulfillment_rate, 4),
            'effective_fare_usd': round(effective_fare, 2),
            'expected_gmv_per_req': round(expected_gmv_per_req, 2)
        })
        
    return pd.DataFrame(curve_data)

def solve_optimal_surge_multiplier(base_fare=15.0, price_sensitivity_k=2.0, weather_severity=0.0, max_churn_threshold=0.35):
    """
    Finds optimal surge multiplier M* that maximizes Expected GMV subject to Cancellation Rate <= max_churn_threshold.
    """
    multipliers = np.linspace(1.0, 3.0, 201)
    df_curve = compute_marketplace_fulfillment_curve(multipliers, base_fare, price_sensitivity_k, weather_severity)
    
    # Filter candidate multipliers meeting maximum churn constraint
    valid_candidates = df_curve[df_curve['rider_cancel_prob'] <= max_churn_threshold]
    
    if valid_candidates.empty:
        # Fallback to minimum cancellation multiplier if all exceed threshold
        best_row = df_curve.loc[df_curve['rider_cancel_prob'].idxmin()]
    else:
        # Select multiplier maximizing expected GMV
        best_row = valid_candidates.loc[valid_candidates['expected_gmv_per_req'].idxmax()]
        
    return {
        'optimal_multiplier': float(best_row['surge_multiplier']),
        'expected_fulfillment_rate': float(best_row['fulfillment_rate']),
        'expected_rider_churn': float(best_row['rider_cancel_prob']),
        'expected_driver_acceptance': float(best_row['driver_accept_prob']),
        'expected_gmv_per_request': float(best_row['expected_gmv_per_req']),
        'curve_df': df_curve
    }

if __name__ == "__main__":
    opt_result = solve_optimal_surge_multiplier(base_fare=18.0, weather_severity=0.6, max_churn_threshold=0.30)
    print("Optimal Surge Solver Output:")
    print(f"Optimal Multiplier M*: {opt_result['optimal_multiplier']}x")
    print(f"Expected Fulfillment Rate: {opt_result['expected_fulfillment_rate']*100:.1f}%")
    print(f"Expected Rider Churn: {opt_result['expected_rider_churn']*100:.1f}%")
    print(f"Expected GMV / Request: ${opt_result['expected_gmv_per_request']:.2f}")
