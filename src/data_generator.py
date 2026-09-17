import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from src.weather_api import fetch_open_meteo_weather
from src.data_loader import load_nyc_benchmark_pickups

def generate_hybrid_marketplace_dataset(days=7, base_requests_per_hour=250, seed=42):
    """
    Combines real NYC spatial pickups & weather API data with simulated 2-sided marketplace telemetry.
    """
    np.random.seed(seed)
    
    # 1. Fetch real/simulated Weather Data
    weather_df = fetch_open_meteo_weather(days_back=days)
    
    # 2. Sample NYC Benchmark Spatial Pickups
    total_records_needed = len(weather_df) * base_requests_per_hour
    pickup_pool = load_nyc_benchmark_pickups(n_samples=max(10000, total_records_needed), seed=seed)
    
    all_ride_logs = []
    pool_idx = 0
    
    for idx, w_row in weather_df.iterrows():
        ts = w_row['timestamp']
        hour = ts.hour
        day_of_week = ts.dayofweek
        is_weekend = day_of_week >= 5
        precip = w_row['precipitation_mm']
        severity = w_row['weather_severity']
        
        # Rush hour demand multipliers
        morning_rush = 1.6 if (7 <= hour <= 9 and not is_weekend) else 1.0
        evening_rush = 1.9 if (17 <= hour <= 20 and not is_weekend) else 1.0
        weekend_night = 1.5 if (21 <= hour or hour <= 2) and is_weekend else 1.0
        
        # Weather demand shock (rain boosts demand by up to 80%)
        weather_shock = 1.0 + (0.8 * severity)
        
        # Combined hourly demand multiplier
        hourly_demand_mult = morning_rush * evening_rush * weekend_night * weather_shock
        n_requests = int(base_requests_per_hour * hourly_demand_mult * np.random.uniform(0.85, 1.15))
        
        # Supply response (rain or late night reduces active drivers)
        base_supply = base_requests_per_hour * np.random.uniform(0.8, 1.1)
        weather_supply_dip = 1.0 - (0.35 * severity) # rain reduces active drivers by up to 35%
        active_drivers = int(base_supply * weather_supply_dip)
        
        # Unboosted initial surge ratio
        supply_demand_ratio = active_drivers / max(1, n_requests)
        if supply_demand_ratio < 0.6:
            initial_surge = round(min(2.8, 1.0 + (0.6 - supply_demand_ratio) * 2.5), 2)
        elif supply_demand_ratio < 0.9:
            initial_surge = round(1.0 + (0.9 - supply_demand_ratio) * 1.0, 2)
        else:
            initial_surge = 1.0
            
        for _ in range(n_requests):
            sample = pickup_pool.iloc[pool_idx % len(pickup_pool)]
            pool_idx += 1
            
            # Rider price elasticity model: higher surge = higher cancellation probability
            # Rain reduces price sensitivity (urgency to get home)
            price_sensitivity_k = 2.2 / (1.0 + 0.5 * severity)
            cancellation_prob = 1.0 / (1.0 + np.exp(-price_sensitivity_k * (initial_surge - 1.4)))
            rider_cancelled = np.random.binomial(1, cancellation_prob) == 1
            
            # Driver acceptance response: higher surge = higher driver acceptance
            driver_accept_prob = 1.0 / (1.0 + np.exp(-3.0 * (initial_surge - 1.1)))
            driver_accepted = np.random.binomial(1, driver_accept_prob) == 1 if not rider_cancelled else False
            
            all_ride_logs.append({
                'ride_id': f"RIDE_{ts.strftime('%Y%m%d%H')}_{pool_idx}",
                'timestamp': ts,
                'date': ts.date(),
                'hour': hour,
                'day_of_week': day_of_week,
                'is_weekend': is_weekend,
                'pickup_lat': sample['pickup_lat'],
                'pickup_lon': sample['pickup_lon'],
                'zone_name': sample['zone_name'],
                'trip_distance_miles': sample['trip_distance_miles'],
                'base_fare_usd': sample['base_fare_usd'],
                'temperature_c': w_row['temperature_c'],
                'precipitation_mm': precip,
                'weather_severity': severity,
                'is_rain': w_row['is_rain'],
                'hourly_active_drivers': active_drivers,
                'hourly_total_requests': n_requests,
                'initial_surge_multiplier': initial_surge,
                'rider_cancelled': rider_cancelled,
                'driver_accepted': driver_accepted,
                'fulfilled': (not rider_cancelled) and driver_accepted
            })
            
    df = pd.DataFrame(all_ride_logs)
    return df

if __name__ == "__main__":
    df = generate_hybrid_marketplace_dataset(days=3, base_requests_per_hour=100)
    print(f"Generated Hybrid Telemetry Dataset: {len(df)} ride requests.")
    print(df.head())
    print("\nFulfillment Summary:")
    print(df['fulfilled'].value_counts(normalize=True))
