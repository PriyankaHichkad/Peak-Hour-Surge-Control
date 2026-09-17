import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

def build_demand_forecast_model(df_telemetry):
    """
    Fits an hourly demand forecasting pipeline using temporal cyclic features and weather severity.
    """
    # Aggregate to hourly demand series
    hourly_df = df_telemetry.groupby(['date', 'hour', 'day_of_week', 'is_weekend']).agg({
        'ride_id': 'count',
        'precipitation_mm': 'mean',
        'weather_severity': 'mean',
        'temperature_c': 'mean'
    }).reset_index().rename(columns={'ride_id': 'demand_count'})
    
    hourly_df['hour_sin'] = np.sin(2 * np.pi * hourly_df['hour'] / 24.0)
    hourly_df['hour_cos'] = np.cos(2 * np.pi * hourly_df['hour'] / 24.0)
    
    feature_cols = ['hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos', 'weather_severity', 'precipitation_mm']
    X = hourly_df[feature_cols]
    y = hourly_df['demand_count']
    
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    
    hourly_df['predicted_demand'] = np.round(model.predict(X))
    
    return model, hourly_df

def forecast_next_24h_demand(model, base_date=None, weather_severity_forecast=0.0):
    """
    Generates 24-hour demand predictions for upcoming day.
    """
    future_hours = []
    if base_date is None:
        base_date = pd.Timestamp.now().date()
        
    for h in range(24):
        h_sin = np.sin(2 * np.pi * h / 24.0)
        h_cos = np.cos(2 * np.pi * h / 24.0)
        dow = 4 # Default Friday
        is_wknd = False
        
        future_hours.append({
            'hour': h,
            'day_of_week': dow,
            'is_weekend': is_wknd,
            'hour_sin': h_sin,
            'hour_cos': h_cos,
            'weather_severity': weather_severity_forecast,
            'precipitation_mm': weather_severity_forecast * 10.0
        })
        
    future_df = pd.DataFrame(future_hours)
    feature_cols = ['hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos', 'weather_severity', 'precipitation_mm']
    future_df['predicted_demand'] = np.maximum(0, np.round(model.predict(future_df[feature_cols])))
    
    return future_df

if __name__ == "__main__":
    from src.data_generator import generate_hybrid_marketplace_dataset
    df = generate_hybrid_marketplace_dataset(days=5, base_requests_per_hour=120)
    model, hourly_summary = build_demand_forecast_model(df)
    print("Demand Forecast Model Trained Successfully!")
    print("Sample Forecast vs Actual:")
    print(hourly_summary[['date', 'hour', 'demand_count', 'predicted_demand']].head(10))
