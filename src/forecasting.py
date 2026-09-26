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
        
    ridge_model = model_dict['model_obj']
    future_hours = []
    
    for h in range(24):
        h_sin = np.sin(2 * np.pi * h / 24.0)
        h_cos = np.cos(2 * np.pi * h / 24.0)
        dow = 4
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
    feature_cols = model_dict['feature_cols']
    future_df['predicted_demand'] = np.maximum(0, np.round(ridge_model.predict(future_df[feature_cols])))
    return future_df
