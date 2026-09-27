import os
import glob
import pandas as pd
import numpy as np

# Key NYC Urban Mobility Hub Centroids for nearest-neighbor zone lookup
NYC_HOTSPOT_CENTROIDS = {
    'Midtown & Times Square': {'lat': 40.7580, 'lon': -73.9855},
    'Financial District & Wall St': {'lat': 40.7075, 'lon': -74.0090},
    'Upper East Side': {'lat': 40.7736, 'lon': -73.9566},
    'Williamsburg & DUMBO': {'lat': 40.7142, 'lon': -73.9614},
    'JFK Airport Hub': {'lat': 40.6413, 'lon': -73.7781},
    'LaGuardia Airport Hub': {'lat': 40.7769, 'lon': -73.8740},
}

UBER_DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'Uber Dataset')
DEFAULT_RAW_CSV = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'nyc_uber_benchmark_pickups.csv')

def get_nearest_zone(lat, lon):
    """Assigns nearest zone based on Euclidean distance to key NYC hub centroids."""
    min_dist = float('inf')
    best_zone = 'Midtown & Times Square'
    for zone, coords in NYC_HOTSPOT_CENTROIDS.items():
        dist = (lat - coords['lat'])**2 + (lon - coords['lon'])**2
        if dist < min_dist:
            min_dist = dist
            best_zone = zone
    return best_zone

def load_nyc_benchmark_pickups(n_samples=5000, seed=42):
    """
    Loads NYC ride pickup records.
    Prioritizes loading directly from authentic downloaded Kaggle Uber CSV files in `data/Uber Dataset/`.
    Falls back to `data/raw/` or spatial sampling if files are absent.
    """
    # 1. Look for raw Kaggle Uber CSV files in data/Uber Dataset/
    uber_files = glob.glob(os.path.join(UBER_DATASET_DIR, "uber-raw-data-*.csv"))
    if uber_files:
        try:
            # Pick first available file (e.g., apr14 or jan-june 15)
            target_file = sorted(uber_files)[0]
            df_raw = pd.read_csv(target_file)
            
            # Standardize columns (Lat/Lon or lat/lon)
            col_map = {col: col.strip().lower() for col in df_raw.columns}
            df_raw.rename(columns=col_map, inplace=True)
            
            if 'lat' in df_raw.columns and 'lon' in df_raw.columns:
                # Filter valid NYC bounding box
                df_valid = df_raw[(df_raw['lat'] >= 40.5) & (df_raw['lat'] <= 40.9) & 
                                  (df_raw['lon'] >= -74.2) & (df_raw['lon'] <= -73.7)].copy()
                
                if len(df_valid) >= n_samples:
                    sample_df = df_valid.sample(n=n_samples, random_state=seed).reset_index(drop=True)
                else:
                    sample_df = df_valid.reset_index(drop=True)
                    
                np.random.seed(seed)
                distances = np.clip(np.random.lognormal(mean=0.9, sigma=0.6, size=len(sample_df)), 0.5, 18.0)
                distances = np.round(distances, 2)
                base_fares = np.round(3.00 + (2.50 * distances), 2)
                
                zones = [get_nearest_zone(r['lat'], r['lon']) for _, r in sample_df.iterrows()]
                
                return pd.DataFrame({
                    'pickup_id': [f"PKP_{100000 + i}" for i in range(len(sample_df))],
                    'zone_name': zones,
                    'pickup_lat': np.round(sample_df['lat'].values, 5),
                    'pickup_lon': np.round(sample_df['lon'].values, 5),
                    'trip_distance_miles': distances,
                    'base_fare_usd': base_fares
                })
        except Exception as e:
            print(f"[DataLoader Warning] Could not parse raw Kaggle Uber CSV ({e}). Falling back.")

    # 2. Check fallback CSV in data/raw/
    if os.path.exists(DEFAULT_RAW_CSV):
        try:
            df_csv = pd.read_csv(DEFAULT_RAW_CSV)
            if len(df_csv) >= n_samples:
                return df_csv.sample(n=n_samples, random_state=seed).reset_index(drop=True)
            return df_csv
        except Exception as e:
            print(f"[DataLoader Warning] Could not load fallback CSV ({e}).")

    # 3. Fallback: Calibrated Spatial Sampling
    np.random.seed(seed)
    records = []
    zones = list(NYC_HOTSPOT_CENTROIDS.keys())
    weights = [0.35, 0.25, 0.15, 0.15, 0.05, 0.05]
    chosen_zones = np.random.choice(zones, size=n_samples, p=weights)
    
    for idx, zone_name in enumerate(chosen_zones):
        centroid = NYC_HOTSPOT_CENTROIDS[zone_name]
        lat = np.random.normal(centroid['lat'], 0.010)
        lon = np.random.normal(centroid['lon'], 0.010)
        dist = np.clip(np.random.lognormal(mean=0.9, sigma=0.6), 0.5, 18.0)
        fare = round(3.00 + (2.50 * dist), 2)
        records.append({
            'pickup_id': f"PKP_{100000 + idx}",
            'zone_name': zone_name,
            'pickup_lat': round(lat, 5),
            'pickup_lon': round(lon, 5),
            'trip_distance_miles': round(dist, 2),
            'base_fare_usd': fare
        })
    return pd.DataFrame(records)

if __name__ == "__main__":
    df = load_nyc_benchmark_pickups(1000)
    print(f"Loaded {len(df)} NYC pickup records:")
    print(df.head())
