import pytest
import pandas as pd
from src.weather_api import fetch_open_meteo_weather, generate_fallback_weather

def test_generate_fallback_weather():
    df = generate_fallback_weather('2026-09-01', '2026-09-03')
    assert isinstance(df, pd.DataFrame)
    assert 'timestamp' in df.columns
    assert 'precipitation_mm' in df.columns
    assert 'weather_severity' in df.columns
    assert (df['weather_severity'] >= 0.0).all() and (df['weather_severity'] <= 1.0).all()

def test_fetch_open_meteo_weather():
    df = fetch_open_meteo_weather(days_back=2)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
