import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

NYC_LAT = 40.7128
NYC_LON = -74.0060

def fetch_open_meteo_weather(lat=NYC_LAT, lon=NYC_LON, days_back=7):
    """
    Fetches real hourly weather data from Open-Meteo API for NYC coordinates.
    Returns DataFrame with columns: ['timestamp', 'temperature_c', 'precipitation_mm', 'weather_code', 'is_rain', 'weather_severity']
    """
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=days_back)
    
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&start_date={start_date}&end_date={end_date}&hourly=temperature_2m,precipitation,weather_code"
    
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            data = response.json()
            hourly = data.get('hourly', {})
            timestamps = pd.to_datetime(hourly.get('time', []))
            temps = hourly.get('temperature_2m', [])
            precip = hourly.get('precipitation', [])
            codes = hourly.get('weather_code', [])
            
            df = pd.DataFrame({
                'timestamp': timestamps,
                'temperature_c': temps,
                'precipitation_mm': precip,
                'weather_code': codes
            })
            
            df['is_rain'] = df['precipitation_mm'] > 0.1
            # Severity index [0.0 to 1.0] based on precipitation mm/hr
            df['weather_severity'] = np.clip(df['precipitation_mm'] / 10.0, 0.0, 1.0)
            return df
    except Exception as e:
        print(f"[Weather API Warning] Could not fetch live Open-Meteo data ({e}). Falling back to cached simulation model.")
    
    return generate_fallback_weather(start_date, end_date)

def generate_fallback_weather(start_date, end_date):
    """
    Generates realistic NYC hourly weather patterns when offline.
    """
    timestamps = pd.date_range(start=start_date, end=end_date, freq='1h')
    np.random.seed(42)
    
    # Base temp fluctuating around 18-24°C
    temps = 20 + 4 * np.sin(np.linspace(0, 4*np.pi, len(timestamps))) + np.random.normal(0, 1, len(timestamps))
    
    # Rain events (15% probability of rain blocks)
    rain_mask = np.random.binomial(1, 0.12, len(timestamps))
    precip = rain_mask * np.random.exponential(scale=3.5, size=len(timestamps))
    precip = np.round(precip, 2)
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature_c': np.round(temps, 1),
        'precipitation_mm': precip,
        'weather_code': np.where(precip > 5.0, 63, np.where(precip > 0.1, 61, 0))
    })
    
    df['is_rain'] = df['precipitation_mm'] > 0.1
    df['weather_severity'] = np.clip(df['precipitation_mm'] / 10.0, 0.0, 1.0)
    return df

if __name__ == "__main__":
    weather_df = fetch_open_meteo_weather(days_back=3)
    print("Weather Data Sample:")
    print(weather_df.head(10))
