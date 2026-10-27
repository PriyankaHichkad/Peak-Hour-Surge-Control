import pytest
import pandas as pd
from src.data_generator import generate_hybrid_marketplace_dataset
from src.forecasting import build_demand_forecast_model, forecast_next_24h_demand, evaluate_forecast_backtest, benchmark_forecasting_models

def test_forecasting_pipeline():
    df_telemetry = generate_hybrid_marketplace_dataset(days=3, base_requests_per_hour=80, seed=42)
    
    # Test model training (tries Prophet, falls back to Ridge)
    model_dict, hourly_df = build_demand_forecast_model(df_telemetry, prefer_prophet=False)
    assert 'model_type' in model_dict
    assert 'predicted_demand' in hourly_df.columns
    
    # Test 24h forecasting with custom base_date (Sunday)
    sunday_date = pd.Timestamp("2026-10-25").date() # Sunday
    future_24h = forecast_next_24h_demand(model_dict, base_date=sunday_date, weather_severity_forecast=0.2)
    assert len(future_24h) == 24
    assert 'predicted_demand' in future_24h.columns
    assert 'timestamp' in future_24h.columns
    assert (future_24h['predicted_demand'] >= 0).all()
    if 'is_weekend' in future_24h.columns:
        assert future_24h['is_weekend'].iloc[0] == True

def test_forecasting_backtest_evaluation():
    df_telemetry = generate_hybrid_marketplace_dataset(days=4, base_requests_per_hour=80, seed=42)
    backtest_res = evaluate_forecast_backtest(df_telemetry, train_days=3)
    assert 'test_mae' in backtest_res
    assert 'test_rmse' in backtest_res
    assert backtest_res['test_mae'] >= 0.0
    assert backtest_res['test_rmse'] >= 0.0
    assert backtest_res['test_hours'] > 0

def test_benchmark_forecasting_models_suite():
    df_telemetry = generate_hybrid_marketplace_dataset(days=5, base_requests_per_hour=80, seed=42)
    df_bench = benchmark_forecasting_models(df_telemetry, train_days=3)
    assert isinstance(df_bench, pd.DataFrame)
    assert len(df_bench) == 3
    assert 'Model' in df_bench.columns
    assert 'Test MAE' in df_bench.columns
    assert 'Test RMSE' in df_bench.columns
    assert 'Test WAPE (%)' in df_bench.columns
    assert 'Seasonal Naïve (Lag-24)' in df_bench['Model'].values
    assert 'Cyclic Ridge Regression' in df_bench['Model'].values
    assert any('Prophet' in m or 'Surrogate' in m for m in df_bench['Model'].values)

