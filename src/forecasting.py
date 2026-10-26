import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge

HAS_PROPHET = False
try:
    from prophet import Prophet
    HAS_PROPHET = True
except ImportError:
    HAS_PROPHET = False

def build_demand_forecast_model(df_telemetry, prefer_prophet=True):
    """
    Fits an hourly demand forecasting model.
    Uses Meta Prophet (with regressors) if available, with robust fallback to Ridge Regression on cyclic features.
    """
    hourly_df = df_telemetry.groupby(['date', 'hour', 'day_of_week', 'is_weekend']).agg({
        'ride_id': 'count',
        'precipitation_mm': 'mean',
        'weather_severity': 'mean',
        'temperature_c': 'mean'
    }).reset_index().rename(columns={'ride_id': 'demand_count'})
    
    hourly_df['timestamp'] = pd.to_datetime(hourly_df['date'].astype(str) + ' ' + hourly_df['hour'].astype(str) + ':00:00')
    hourly_df = hourly_df.sort_values('timestamp').reset_index(drop=True)
    
    if prefer_prophet and HAS_PROPHET:
        try:
            prophet_df = pd.DataFrame({
                'ds': hourly_df['timestamp'],
                'y': hourly_df['demand_count'],
                'precipitation_mm': hourly_df['precipitation_mm'],
                'weather_severity': hourly_df['weather_severity']
            })
            
            m = Prophet(yearly_seasonality=False, weekly_seasonality=True, daily_seasonality=True)
            m.add_regressor('precipitation_mm')
            m.add_regressor('weather_severity')
            m.fit(prophet_df)
            
            forecast = m.predict(prophet_df)
            hourly_df['predicted_demand'] = np.maximum(0, np.round(forecast['yhat']))
            
            return {'model_type': 'Prophet', 'model_obj': m, 'data_df': prophet_df}, hourly_df
        except Exception as e:
            print(f"[Prophet Warning] Prophet fit failed ({e}). Falling back to Ridge regression.")
            
    hourly_df['hour_sin'] = np.sin(2 * np.pi * hourly_df['hour'] / 24.0)
    hourly_df['hour_cos'] = np.cos(2 * np.pi * hourly_df['hour'] / 24.0)
    
    feature_cols = ['hour', 'day_of_week', 'is_weekend', 'hour_sin', 'hour_cos', 'weather_severity', 'precipitation_mm']
    X = hourly_df[feature_cols]
    y = hourly_df['demand_count']
    
    ridge_model = Ridge(alpha=1.0)
    ridge_model.fit(X, y)
    
    hourly_df['predicted_demand'] = np.maximum(0, np.round(ridge_model.predict(X)))
    
    return {'model_type': 'Ridge Regression', 'model_obj': ridge_model, 'feature_cols': feature_cols}, hourly_df

def forecast_next_24h_demand(model_dict, base_date=None, weather_severity_forecast=0.0):
    """
    Generates 24-hour demand predictions for upcoming day using either Prophet or Ridge model.
    """
    if base_date is None:
        base_date = pd.Timestamp.now().date()
        
    model_type = model_dict['model_type']
    
    if model_type == 'Prophet':
        m = model_dict['model_obj']
        # Pandas 2.2+ frequency compatibility: use 'h' or '1h' instead of deprecated 'H'
        future = m.make_future_dataframe(periods=24, freq='h')
        future['precipitation_mm'] = weather_severity_forecast * 10.0
        future['weather_severity'] = weather_severity_forecast
        
        forecast = m.predict(future.tail(24))
        future_24 = forecast[['ds', 'yhat']].rename(columns={'ds': 'timestamp', 'yhat': 'predicted_demand'})
        future_24['hour'] = future_24['timestamp'].dt.hour
        future_24['predicted_demand'] = np.maximum(0, np.round(future_24['predicted_demand']))
        return future_24
        
    base_ts = pd.to_datetime(base_date)
    timestamps = pd.date_range(start=base_ts, periods=24, freq='h')
    
    ridge_model = model_dict['model_obj']
    future_hours = []
    
    for ts in timestamps:
        h = ts.hour
        dow = ts.dayofweek
        is_wknd = dow >= 5
        h_sin = np.sin(2 * np.pi * h / 24.0)
        h_cos = np.cos(2 * np.pi * h / 24.0)
        
        future_hours.append({
            'timestamp': ts,
            'hour': h,
            'day_of_week': dow,
            'is_weekend': is_wknd,
            'hour_sin': h_sin,
            'hour_cos': h_cos,
            'weather_severity': weather_severity_forecast,
            'precipitation_mm': weather_severity_forecast * 10.0
        })
        
    future_df = pd.DataFrame(future_hours)
    feature_cols = model_dict['feature_cols']
    future_df['predicted_demand'] = np.maximum(0, np.round(ridge_model.predict(future_df[feature_cols])))
    return future_df

def evaluate_forecast_backtest(df_telemetry, train_days=5):
    """
    Evaluates forecasting accuracy on unseen future observations using temporal train/test split.
    Trains on first train_days and tests on holdout observations, returning MAE, RMSE, and WAPE.
    """
    unique_dates = sorted(df_telemetry['date'].unique())
    if len(unique_dates) <= 1:
        return {'test_mae': 0.0, 'test_rmse': 0.0, 'test_wape': 0.0, 'test_hours': 0, 'model_type': 'N/A'}
        
    split_idx = min(train_days, len(unique_dates) - 1)
    train_dates = unique_dates[:split_idx]
    test_dates = unique_dates[split_idx:]
    
    train_df = df_telemetry[df_telemetry['date'].isin(train_dates)]
    test_df = df_telemetry[df_telemetry['date'].isin(test_dates)]
    
    model_dict, _ = build_demand_forecast_model(train_df)
    
    test_hourly = test_df.groupby(['date', 'hour', 'day_of_week', 'is_weekend']).agg({
        'ride_id': 'count',
        'precipitation_mm': 'mean',
        'weather_severity': 'mean',
        'temperature_c': 'mean'
    }).reset_index().rename(columns={'ride_id': 'actual_demand'})
    
    test_hourly['timestamp'] = pd.to_datetime(test_hourly['date'].astype(str) + ' ' + test_hourly['hour'].astype(str) + ':00:00')
    test_hourly = test_hourly.sort_values('timestamp').reset_index(drop=True)
    
    if model_dict['model_type'] == 'Prophet':
        m = model_dict['model_obj']
        prophet_test = pd.DataFrame({
            'ds': test_hourly['timestamp'],
            'precipitation_mm': test_hourly['precipitation_mm'],
            'weather_severity': test_hourly['weather_severity']
        })
        preds = m.predict(prophet_test)
        y_pred = np.maximum(0, np.round(preds['yhat'].values))
    else:
        test_hourly['hour_sin'] = np.sin(2 * np.pi * test_hourly['hour'] / 24.0)
        test_hourly['hour_cos'] = np.cos(2 * np.pi * test_hourly['hour'] / 24.0)
        feature_cols = model_dict['feature_cols']
        ridge = model_dict['model_obj']
        y_pred = np.maximum(0, np.round(ridge.predict(test_hourly[feature_cols])))
        
    y_true = test_hourly['actual_demand'].values
    mae = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    wape = float(np.sum(np.abs(y_true - y_pred)) / max(1, np.sum(y_true)))
    
    return {
        'test_mae': round(mae, 2),
        'test_rmse': round(rmse, 2),
        'test_wape': round(wape, 4),
        'test_hours': len(y_true),
        'model_type': model_dict['model_type']
    }
