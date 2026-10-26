import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

DEFAULT_PRICE_SENSITIVITY_K = 2.2
DEFAULT_INFLECTION_M0 = 1.4
DEFAULT_DRIVER_INCENTIVE_LAMBDA = 3.2

def rider_cancellation_probability(multiplier, price_sensitivity_k=DEFAULT_PRICE_SENSITIVITY_K, inflection_m0=DEFAULT_INFLECTION_M0, weather_severity=0.0):
    """
    Rider Churn Probability as a function of Surge Multiplier and Weather Urgency.
    Baseline non-surged cancellation ~12%-14% at 1.0x, scaling up smoothly with surge.
    """
    effective_k = price_sensitivity_k / (1.0 + 0.5 * weather_severity)
    sigmoid = 1.0 / (1.0 + np.exp(-effective_k * (multiplier - inflection_m0)))
    base_sigmoid = 1.0 / (1.0 + np.exp(-effective_k * (1.0 - inflection_m0)))
    prob = 0.12 + 0.75 * (sigmoid - base_sigmoid) / (1.0 - base_sigmoid)
    return np.clip(prob, 0.05, 0.95)

def driver_acceptance_probability(multiplier, incentive_lambda=DEFAULT_DRIVER_INCENTIVE_LAMBDA):
    """
    Driver Acceptance Probability as a function of Surge Multiplier Bonus.
    Base acceptance ~78%-85% at 1.0x base fare, scaling up to 96% at higher surge payouts.
    """
    prob = 0.78 + 0.20 / (1.0 + np.exp(-incentive_lambda * (multiplier - 1.2)))
    return np.clip(prob, 0.50, 0.98)

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
    Solves for optimal surge multiplier M* dynamically driven by local supply-demand ratio and time/weather features
    using SciPy bounded continuous optimization (minimize_scalar).
    """
    multipliers = np.linspace(1.0, 3.0, 201)
    df_curve = compute_marketplace_fulfillment_curve(multipliers, base_fare, price_sensitivity_k, weather_severity, supply_demand_ratio)
    
    if supply_demand_ratio >= 1.0:
        return {
            'optimal_multiplier': 1.0,
            'continuous_scipy_m': 1.0,
            'expected_fulfillment_rate': float(df_curve.iloc[0]['fulfillment_rate']),
            'expected_rider_churn': float(df_curve.iloc[0]['rider_cancel_prob']),
            'expected_driver_acceptance': float(df_curve.iloc[0]['driver_accept_prob']),
            'expected_gmv_per_request': float(df_curve.iloc[0]['expected_gmv_per_req']),
            'curve_df': df_curve
        }
        
    def objective_fn(m):
        p_cancel = rider_cancellation_probability(m, price_sensitivity_k=price_sensitivity_k, weather_severity=weather_severity)
        if p_cancel > max_churn_threshold:
            return 1e5 * (p_cancel - max_churn_threshold + 1.0)
        p_accept = driver_acceptance_probability(m)
        conversion = 1.0 - p_cancel
        supply_response = min(1.0, supply_demand_ratio * (1.0 + 0.35 * (m - 1.0)))
        fulfillment = min(conversion, supply_response) * p_accept
        gmv = (base_fare * m) * fulfillment
        
        imbalance = (conversion - supply_response)**2
        churn_friction = 1.5 * (p_cancel - 0.12)**2
        
        net_welfare = gmv - 2.5 * base_fare * imbalance - 1.5 * base_fare * churn_friction
        return -net_welfare

    scipy_res = minimize_scalar(objective_fn, bounds=(1.0, 1.30), method='bounded')
    optimal_continuous_m = round(float(scipy_res.x), 2)
    
    # Lookup curve statistics at optimal multiplier
    closest_idx = (np.abs(df_curve['surge_multiplier'] - optimal_continuous_m)).argmin()
    opt_row = df_curve.iloc[closest_idx]
    
    return {
        'optimal_multiplier': optimal_continuous_m,
        'continuous_scipy_m': optimal_continuous_m,
        'expected_fulfillment_rate': float(opt_row['fulfillment_rate']),
        'expected_rider_churn': float(opt_row['rider_cancel_prob']),
        'expected_driver_acceptance': float(opt_row['driver_accept_prob']),
        'expected_gmv_per_request': float(opt_row['expected_gmv_per_req']),
        'curve_df': df_curve
    }

if __name__ == "__main__":
    for ratio in [1.5, 0.8, 0.4, 0.2]:
        res = solve_optimal_surge_multiplier(supply_demand_ratio=ratio)
        print(f"Supply/Demand Ratio {ratio} -> Optimal Surge M*: {res['optimal_multiplier']}x")
