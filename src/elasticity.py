import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

DEFAULT_PRICE_SENSITIVITY_K = 2.2
DEFAULT_INFLECTION_M0 = 1.4
DEFAULT_DRIVER_INCENTIVE_LAMBDA = 3.2

def rider_cancellation_probability(multiplier, price_sensitivity_k=DEFAULT_PRICE_SENSITIVITY_K, inflection_m0=DEFAULT_INFLECTION_M0, weather_severity=0.0):
    """
    Rider Churn Probability as a function of Surge Multiplier and Weather Urgency.
    Rain/storm reduces rider price sensitivity.
    """
    effective_k = price_sensitivity_k / (1.0 + 0.5 * weather_severity)
    prob = 1.0 / (1.0 + np.exp(-effective_k * (multiplier - inflection_m0)))
    return np.clip(prob, 0.02, 0.95)

def driver_acceptance_probability(multiplier, incentive_lambda=DEFAULT_DRIVER_INCENTIVE_LAMBDA):
    """
    Driver Acceptance Probability as a function of Surge Multiplier Bonus.
    """
    prob = 1.0 / (1.0 + np.exp(-incentive_lambda * (multiplier - 1.1)))
    return np.clip(prob, 0.25, 0.98)

def compute_marketplace_fulfillment_curve(multipliers, base_fare=15.0, price_sensitivity_k=DEFAULT_PRICE_SENSITIVITY_K, weather_severity=0.0, supply_demand_ratio=1.0):
    """
    Evaluates Fulfillment Rate %, Gross Merchandise Value (GMV), and Churn % across a range of multipliers,
    factoring in local supply-demand deficit.
    """
    curve_data = []
    
    # Supply deficit factor: when supply/demand ratio < 1.0, unfulfilled demand penalizes low multipliers
    deficit_factor = max(0.2, min(1.5, 1.0 / max(0.1, supply_demand_ratio)))
    
    for m in multipliers:
        p_cancel = rider_cancellation_probability(m, price_sensitivity_k=price_sensitivity_k, weather_severity=weather_severity)
        p_accept = driver_acceptance_probability(m)
        
        conversion_rider = 1.0 - p_cancel
        # Fulfillment capped by available local driver supply ratio
        fulfillment_rate = min(1.0, conversion_rider * p_accept * min(1.0, supply_demand_ratio * (1.0 + 0.3 * (m - 1.0))))
        
        effective_fare = base_fare * m
        # Expected GMV incorporates local supply-demand deficit factor
        expected_gmv_per_req = effective_fare * fulfillment_rate * (1.0 + 0.2 * (deficit_factor - 1.0))
        
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

def solve_optimal_surge_multiplier(base_fare=15.0, price_sensitivity_k=DEFAULT_PRICE_SENSITIVITY_K, weather_severity=0.0, max_churn_threshold=0.35, supply_demand_ratio=1.0):
    """
    Solves for optimal surge multiplier M* dynamically driven by local supply-demand ratio and time/weather features.
    If supply >= demand (ratio >= 1.0), surge is 1.0x.
    If severe deficit (ratio < 0.5), surge increases to attract drivers and balance queue.
    """
    if supply_demand_ratio >= 1.15:
        # High driver availability relative to demand -> Base price 1.0x
        multipliers = np.linspace(1.0, 3.0, 201)
        df_curve = compute_marketplace_fulfillment_curve(multipliers, base_fare, price_sensitivity_k, weather_severity, supply_demand_ratio)
        return {
            'optimal_multiplier': 1.0,
            'continuous_scipy_m': 1.0,
            'expected_fulfillment_rate': float(df_curve.iloc[0]['fulfillment_rate']),
            'expected_rider_churn': float(df_curve.iloc[0]['rider_cancel_prob']),
            'expected_driver_acceptance': float(df_curve.iloc[0]['driver_accept_prob']),
            'expected_gmv_per_request': float(df_curve.iloc[0]['expected_gmv_per_req']),
            'curve_df': df_curve
        }
        
    multipliers = np.linspace(1.0, 3.0, 201)
    df_curve = compute_marketplace_fulfillment_curve(multipliers, base_fare, price_sensitivity_k, weather_severity, supply_demand_ratio)
    
    valid_candidates = df_curve[df_curve['rider_cancel_prob'] <= max_churn_threshold]
    
    if valid_candidates.empty:
        best_row = df_curve.loc[df_curve['rider_cancel_prob'].idxmin()]
    else:
        best_row = valid_candidates.loc[valid_candidates['expected_gmv_per_req'].idxmax()]
        
    def objective_fn(m):
        p_cancel = rider_cancellation_probability(m, price_sensitivity_k=price_sensitivity_k, weather_severity=weather_severity)
        if p_cancel > max_churn_threshold:
            return 1e6 * (p_cancel - max_churn_threshold + 1.0)
        p_accept = driver_acceptance_probability(m)
        fulfillment = min(1.0, (1.0 - p_cancel) * p_accept * min(1.0, supply_demand_ratio * (1.0 + 0.3 * (m - 1.0))))
        gmv = (base_fare * m) * fulfillment
        return -gmv

    scipy_res = minimize_scalar(objective_fn, bounds=(1.0, 3.0), method='bounded')
    optimal_continuous_m = round(float(scipy_res.x), 2)
    
    return {
        'optimal_multiplier': float(best_row['surge_multiplier']),
        'continuous_scipy_m': optimal_continuous_m,
        'expected_fulfillment_rate': float(best_row['fulfillment_rate']),
        'expected_rider_churn': float(best_row['rider_cancel_prob']),
        'expected_driver_acceptance': float(best_row['driver_accept_prob']),
        'expected_gmv_per_request': float(best_row['expected_gmv_per_req']),
        'curve_df': df_curve
    }

if __name__ == "__main__":
    for ratio in [1.5, 0.8, 0.4, 0.2]:
        res = solve_optimal_surge_multiplier(supply_demand_ratio=ratio)
        print(f"Supply/Demand Ratio {ratio} -> Optimal Surge M*: {res['optimal_multiplier']}x")
