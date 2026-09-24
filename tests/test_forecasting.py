import pytest
import pandas as pd
from src.data_generator import generate_hybrid_marketplace_dataset
from src.forecasting import build_demand_forecast_model, forecast_next_24h_demand

def test_forecasting_pipeline():
    df_telemetry = generate_hybrid_marketplace_dataset(days=3, base_requests_per_hour=80, seed=42)
    
    # Test model training (tries Prophet, falls back to Ridge)
    model_dict, hourly_df = build_demand_forecast_model(df_telemetry, prefer_prophet=False)
    assert 'model_type' in model_dict
    assert 'predicted_demand' in hourly_df.columns
    
    # Test 24h forecasting
    future_24h = forecast_next_24h_demand(model_dict, weather_severity_forecast=0.2)
    assert len(future_24h) == 24
    assert 'predicted_demand' in future_24h.columns
    assert (future_24h['predicted_demand'] >= 0).all()
