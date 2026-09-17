import pandas as pd
import numpy as np

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
    df_fulfilled['effective_fare'] = df_fulfilled['base_fare_usd'] * df_fulfilled['initial_surge_multiplier']
    
    total_gmv = df_fulfilled['effective_fare'].sum()
    avg_fare_per_fulfilled_trip = df_fulfilled['effective_fare'].mean() if len(df_fulfilled) > 0 else 0.0
    
    # Driver earnings (Assuming 80% payout split to driver)
    driver_net_earnings = total_gmv * 0.80
    total_driver_hours = max(1.0, df['hourly_active_drivers'].mean() * (len(df['hour'].unique()) if 'hour' in df else 1.0))
    driver_hourly_rate = driver_net_earnings / total_driver_hours
    
    # NPS Impact Score Model
    # High surge (> 2.0x) and cancellation penalty reduces NPS
    high_surge_penalty = (df['initial_surge_multiplier'] > 2.0).mean() * 30
    cancellation_penalty = cancellation_rate * 50
    fulfillment_bonus = fulfillment_rate * 60
    nps_index = int(np.clip(fulfillment_bonus - cancellation_penalty - high_surge_penalty, -100, 100))
    
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

def compare_baseline_vs_optimized(df_raw, opt_surge_multiplier):
    """
    Compares baseline unoptimized marketplace performance vs optimized surge control engine.
    """
    baseline_kpis = calculate_marketplace_kpis(df_raw)
    
    # Create optimized scenario copy
    df_opt = df_raw.copy()
    df_opt['initial_surge_multiplier'] = opt_surge_multiplier
    
    # Recalculate rider cancellation and driver acceptance under optimized multiplier
    from src.elasticity import rider_cancellation_probability, driver_acceptance_probability
    
    np.random.seed(42)
    p_cancel = rider_cancellation_probability(opt_surge_multiplier, weather_severity=df_opt['weather_severity'].mean())
    p_accept = driver_acceptance_probability(opt_surge_multiplier)
    
    df_opt['rider_cancelled'] = np.random.binomial(1, p_cancel, len(df_opt)) == 1
    df_opt['driver_accepted'] = np.random.binomial(1, p_accept, len(df_opt)) == 1
    df_opt['fulfilled'] = (~df_opt['rider_cancelled']) & df_opt['driver_accepted']
    
    optimized_kpis = calculate_marketplace_kpis(df_opt)
    
    # Compute relative lifts
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
