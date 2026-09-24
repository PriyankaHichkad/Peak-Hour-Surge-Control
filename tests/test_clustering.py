import pytest
import pandas as pd
from src.data_generator import generate_hybrid_marketplace_dataset
from src.clustering import detect_spatial_hotspots

def test_detect_spatial_hotspots():
    df_telemetry = generate_hybrid_marketplace_dataset(days=1, base_requests_per_hour=100, seed=42)
    peak_slice = df_telemetry[df_telemetry['hour'] == 18]
    
    df_clustered, cluster_summary = detect_spatial_hotspots(peak_slice, eps_km=0.8, min_samples=5)
    
    assert 'cluster_id' in df_clustered.columns
    assert isinstance(cluster_summary, pd.DataFrame)
    if not cluster_summary.empty:
        assert 'centroid_lat' in cluster_summary.columns
        assert 'request_count' in cluster_summary.columns
