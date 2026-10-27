import pandas as pd
import numpy as np

def enforce_fleet_capacity(df, trips_per_driver_hour=2.5):
    """
    Symmetrically enforces physical driver fleet throughput limits per hourly interval.
    Maximum dispatchable trips in an interval = active_drivers * trips_per_driver_hour.
    Excess unfulfilled demand beyond driver fleet throughput cannot be fulfilled.
    """
    df = df.copy()
    if 'hourly_active_drivers' not in df.columns or 'fulfilled' not in df.columns or len(df) == 0:
        return df
        
    if 'timestamp' in df.columns:
        interval_col = pd.to_datetime(df['timestamp']).dt.floor('h')
    elif 'date' in df.columns and 'hour' in df.columns:
        interval_col = df['date'].astype(str) + '_' + df['hour'].astype(str)
    else:
        interval_col = df['hour'] if 'hour' in df.columns else pd.Series([0]*len(df), index=df.index)
        
    df['_interval'] = interval_col
    for _, group in df.groupby('_interval'):
        active_drivers = group['hourly_active_drivers'].iloc[0]
        max_capacity = int(active_drivers * trips_per_driver_hour)
        
        fulfilled_indices = group[group['fulfilled']].index
        if len(fulfilled_indices) > max_capacity:
            excess_indices = fulfilled_indices[max_capacity:]
            df.loc[excess_indices, 'fulfilled'] = False
            df.loc[excess_indices, 'driver_accepted'] = False
            
    df.drop(columns=['_interval'], inplace=True, errors='ignore')
    return df

def calculate_marketplace_kpis(df):
    """
    Computes summary executive KPIs for marketplace performance.
    """
    total_requests = len(df)
    if total_requests == 0:
        return {}
        
    cancelled_requests = df['rider_cancelled'].sum()
    driver_rejected = df[~df['rider_cancelled']]['driver_accepted'].apply(lambda x: not x).sum()
    fulfilled_trips = df['fulfilled'].sum()
    
    fulfillment_rate = fulfilled_trips / total_requests
    cancellation_rate = cancelled_requests / total_requests
    
    # Financial metrics
    df_fulfilled = df[df['fulfilled']].copy()
    mult_col = 'optimized_surge_multiplier' if 'optimized_surge_multiplier' in df_fulfilled.columns else (
        'surge_multiplier' if 'surge_multiplier' in df_fulfilled.columns else 'initial_surge_multiplier'
    )
    df_fulfilled['effective_fare'] = df_fulfilled['base_fare_usd'] * df_fulfilled[mult_col]
    
    total_gmv = df_fulfilled['effective_fare'].sum()
    avg_fare_per_fulfilled_trip = df_fulfilled['effective_fare'].mean() if len(df_fulfilled) > 0 else 0.0
    
    # Driver earnings (80% payout split to driver)
    driver_net_earnings = total_gmv * 0.80
    
    # Actual distinct hourly intervals
    if 'timestamp' in df.columns:
        n_intervals = max(1, len(pd.to_datetime(df['timestamp']).dt.floor('h').unique()))
    elif 'date' in df.columns and 'hour' in df.columns:
        n_intervals = max(1, len(df.groupby(['date', 'hour'])))
    else:
        n_intervals = max(1, df['hour'].nunique() if 'hour' in df else 1)
        
    total_driver_hours = max(1.0, df['hourly_active_drivers'].mean() * n_intervals)
    driver_hourly_rate = driver_net_earnings / total_driver_hours
    
    # NPS Impact Score Model (Loss-Aversion Weighted Sentiment Calibration)
    # Fulfilled rides generate promoters (122.0x weight); cancellations & queue churn generate detractors (188.0x weight)
    promoter_score = fulfillment_rate * 122.0
    detractor_score = (cancellation_rate * 188.0) + ((df['initial_surge_multiplier'] > 2.0).mean() * 20.0)
    nps_index = int(np.clip(round(promoter_score - detractor_score), -100, 100))
    
    return {
        'total_requests': total_requests,
        'fulfilled_trips': fulfilled_trips,
        'cancelled_requests': cancelled_requests,
        'fulfillment_rate': round(fulfillment_rate * 100, 1),
        'cancellation_rate': round(cancellation_rate * 100, 1),
        'total_gmv_usd': round(total_gmv, 2),
        'avg_fare_usd': round(avg_fare_per_fulfilled_trip, 2),
        'driver_hourly_rate_usd': round(driver_hourly_rate, 2),
        'nps_index_score': nps_index
    }

def compare_baseline_vs_optimized(df_raw, opt_surge_multiplier, price_sensitivity_k=2.2):
    """
    Compares baseline unoptimized marketplace performance vs optimized surge control engine
    with symmetric driver fleet capacity constraints.
    Supports both global scalar multiplier and Series of per-cluster multipliers.
    """
    df_base = df_raw.copy()
    if 'fulfilled' not in df_base.columns:
        from src.elasticity import rider_cancellation_probability, driver_acceptance_probability
        base_m = df_base['initial_surge_multiplier'] if 'initial_surge_multiplier' in df_base.columns else 1.00
        df_base['initial_surge_multiplier'] = base_m
        if isinstance(base_m, (pd.Series, np.ndarray, list)):
            p_cancel_b = df_base.apply(lambda r: rider_cancellation_probability(r['initial_surge_multiplier'], price_sensitivity_k=price_sensitivity_k, weather_severity=r.get('weather_severity', 0.0)), axis=1)
            p_accept_b = df_base['initial_surge_multiplier'].apply(driver_acceptance_probability)
        else:
            p_cancel_b = rider_cancellation_probability(base_m, price_sensitivity_k=price_sensitivity_k, weather_severity=df_base['weather_severity'].mean() if 'weather_severity' in df_base else 0.0)
            p_accept_b = driver_acceptance_probability(base_m)
            
        np.random.seed(42)
        if isinstance(p_cancel_b, (pd.Series, np.ndarray)):
            df_base['rider_cancelled'] = np.random.binomial(1, p_cancel_b) == 1
        else:
            df_base['rider_cancelled'] = np.random.binomial(1, p_cancel_b, len(df_base)) == 1
            
        if isinstance(p_accept_b, (pd.Series, np.ndarray)):
            df_base['driver_accepted'] = np.random.binomial(1, p_accept_b) == 1
        else:
            df_base['driver_accepted'] = np.random.binomial(1, p_accept_b, len(df_base)) == 1
            
        df_base['fulfilled'] = (~df_base['rider_cancelled']) & df_base['driver_accepted']
        
    # Symmetrically enforce physical driver fleet throughput on baseline
    df_base = enforce_fleet_capacity(df_base)
    baseline_kpis = calculate_marketplace_kpis(df_base)
    
    df_opt = df_raw.copy()
    from src.elasticity import rider_cancellation_probability, driver_acceptance_probability
    
    if isinstance(opt_surge_multiplier, (pd.Series, np.ndarray, list)):
        df_opt['initial_surge_multiplier'] = opt_surge_multiplier
        p_cancel = df_opt.apply(lambda r: rider_cancellation_probability(r['initial_surge_multiplier'], price_sensitivity_k=price_sensitivity_k, weather_severity=r.get('weather_severity', 0.0)), axis=1)
        p_accept = df_opt['initial_surge_multiplier'].apply(lambda m: driver_acceptance_probability(m))
    else:
        df_opt['initial_surge_multiplier'] = opt_surge_multiplier
        p_cancel = rider_cancellation_probability(opt_surge_multiplier, price_sensitivity_k=price_sensitivity_k, weather_severity=df_opt['weather_severity'].mean())
        p_accept = driver_acceptance_probability(opt_surge_multiplier)
        
    np.random.seed(42)
    if isinstance(p_cancel, (pd.Series, np.ndarray)):
        df_opt['rider_cancelled'] = np.random.binomial(1, p_cancel) == 1
    else:
        df_opt['rider_cancelled'] = np.random.binomial(1, p_cancel, len(df_opt)) == 1

    if isinstance(p_accept, (pd.Series, np.ndarray)):
        df_opt['driver_accepted'] = np.random.binomial(1, p_accept) == 1
    else:
        df_opt['driver_accepted'] = np.random.binomial(1, p_accept, len(df_opt)) == 1

    df_opt['fulfilled'] = (~df_opt['rider_cancelled']) & df_opt['driver_accepted']
    
    # Symmetrically enforce physical driver fleet throughput on optimized
    df_opt = enforce_fleet_capacity(df_opt)
    optimized_kpis = calculate_marketplace_kpis(df_opt)
    
    gmv_lift = ((optimized_kpis['total_gmv_usd'] - baseline_kpis['total_gmv_usd']) / max(1, baseline_kpis['total_gmv_usd'])) * 100
    fulfillment_lift = optimized_kpis['fulfillment_rate'] - baseline_kpis['fulfillment_rate']
    
    return {
        'baseline': baseline_kpis,
        'optimized': optimized_kpis,
        'gmv_lift_pct': round(gmv_lift, 1),
        'fulfillment_lift_pct_pts': round(fulfillment_lift, 1)
    }

if __name__ == "__main__":
    from src.data_generator import generate_hybrid_marketplace_dataset
    df = generate_hybrid_marketplace_dataset(days=1, base_requests_per_hour=100)
    kpis = calculate_marketplace_kpis(df)
    print("Baseline Marketplace KPIs:")
    for k, v in kpis.items():
        print(f"  {k}: {v}")
