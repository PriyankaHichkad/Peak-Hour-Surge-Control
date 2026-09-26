import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN

# Earth radius in kilometers for Haversine conversion
EARTH_RADIUS_KM = 6371.0088

def detect_spatial_hotspots(df_subset, eps_km=0.35, min_samples=5):
    """
    Applies granular DBSCAN clustering on ride request GPS coordinates using Haversine distance.
    Parameters tuned (eps_km=0.35 / 350 meters) to detect distinct neighborhood micro-clusters.
    """
    if len(df_subset) < min_samples:
        df_subset = df_subset.copy()
        df_subset['cluster_id'] = -1
        return df_subset, pd.DataFrame()
    
    df_copy = df_subset.copy()
    
    coords_rad = np.radians(df_copy[['pickup_lat', 'pickup_lon']].values)
    kms_per_radian = EARTH_RADIUS_KM
    eps_rad = eps_km / kms_per_radian
    
    db = DBSCAN(eps=eps_rad, min_samples=min_samples, metric='haversine')
    df_copy['cluster_id'] = db.fit_predict(coords_rad)
    
    cluster_stats = []
    unique_clusters = [c for c in df_copy['cluster_id'].unique() if c != -1]
    
    for c_id in unique_clusters:
        c_points = df_copy[df_copy['cluster_id'] == c_id]
        c_lat = c_points['pickup_lat'].mean()
        c_lon = c_points['pickup_lon'].mean()
        n_req = len(c_points)
        cancellations = c_points['rider_cancelled'].sum()
        cancel_rate = cancellations / max(1, n_req)
        avg_surge = c_points['initial_surge_multiplier'].mean()
        
        cluster_stats.append({
            'cluster_id': c_id,
            'centroid_lat': round(c_lat, 5),
            'centroid_lon': round(c_lon, 5),
            'request_count': n_req,
            'cancellation_rate': round(cancel_rate, 3),
            'avg_surge_multiplier': round(avg_surge, 2),
            'primary_zone': c_points['zone_name'].mode()[0] if not c_points['zone_name'].empty else 'Urban Core'
        })
        
    summary_df = pd.DataFrame(cluster_stats)
    if not summary_df.empty:
        summary_df = summary_df.sort_values(by='request_count', ascending=False).reset_index(drop=True)
        
    return df_copy, summary_df

if __name__ == "__main__":
    from src.data_generator import generate_hybrid_marketplace_dataset
    df = generate_hybrid_marketplace_dataset(days=1, base_requests_per_hour=150)
    peak_slice = df[df['hour'] == 18]
    df_clustered, clusters = detect_spatial_hotspots(peak_slice, eps_km=0.35, min_samples=5)
    print(f"Detected {len(clusters)} Granular Hotspots at 6 PM:")
    print(clusters)
