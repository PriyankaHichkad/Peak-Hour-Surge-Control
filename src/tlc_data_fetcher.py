import os
import requests
import pandas as pd
import numpy as np

# NYC TLC Official Cloudfront Storage URLs
NYC_TLC_BASE_URL = "https://d37ci6vzurychx.cloudfront.net/trip-data"
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw')

# Map NYC TLC Zone IDs to approximate representative centroid coordinates
TLC_ZONE_COORDS = {
    # Popular Manhattan & Brooklyn TLC Zones
    161: {'zone_name': 'Midtown Center', 'lat': 40.7565, 'lon': -73.9805},
    162: {'zone_name': 'Midtown East', 'lat': 40.7538, 'lon': -73.9723},
    230: {'zone_name': 'Times Square/Theatre District', 'lat': 40.7589, 'lon': -73.9851},
    209: {'zone_name': 'Seaport / Wall Street', 'lat': 40.7072, 'lon': -74.0041},
    236: {'zone_name': 'Upper East Side North', 'lat': 40.7760, 'lon': -73.9525},
    255: {'zone_name': 'Williamsburg (North Side)', 'lat': 40.7180, 'lon': -73.9570},
    132: {'zone_name': 'JFK Airport', 'lat': 40.6413, 'lon': -73.7781},
    138: {'zone_name': 'LaGuardia Airport', 'lat': 40.7769, 'lon': -73.8740},
}

def download_nyc_tlc_parquet(year=2024, month=1, trip_type='fhvhv'):
    """
    Downloads official NYC TLC Parquet dataset file directly from NYC TLC Cloudfront CDN.
    trip_type options:
      - 'fhvhv': High-Volume For-Hire Vehicle (Uber & Lyft)
      - 'yellow': Yellow Medallion Taxis
      - 'green': Green Borough Taxis
    """
    os.makedirs(DATA_DIR, exist_ok=True)
    filename = f"{trip_type}_tripdata_{year:04d}-{month:02d}.parquet"
    target_path = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(target_path):
        print(f"[NYC TLC Loader] Local file already exists: {target_path}")
        return target_path
        
    url = f"{NYC_TLC_BASE_URL}/{filename}"
    print(f"[NYC TLC Loader] Downloading official NYC TLC data from {url}...")
    
    try:
        response = requests.get(url, stream=True, timeout=30)
        if response.status_code == 200:
            with open(target_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024*1024):
                    if chunk:
                        f.write(chunk)
            print(f"[NYC TLC Loader] Successfully saved to {target_path}")
            return target_path
        else:
            print(f"[NYC TLC Loader Warning] HTTP {response.status_code} - File not found at {url}")
            return None
    except Exception as e:
        print(f"[NYC TLC Loader Error] Failed to download TLC dataset ({e})")
        return None

def load_official_nyc_tlc_pickups(n_samples=5000, year=2024, month=1, trip_type='yellow'):
    """
    Loads ride records directly from official downloaded NYC TLC Parquet files.
    """
    parquet_path = download_nyc_tlc_parquet(year, month, trip_type)
    if not parquet_path or not os.path.exists(parquet_path):
        print("[NYC TLC Loader Warning] Parquet file unavailable. Cannot sample.")
        return None
        
    try:
        print(f"[NYC TLC Loader] Reading Parquet dataset: {parquet_path}...")
        df_parquet = pd.read_parquet(parquet_path)
        print(f"[NYC TLC Loader] Total records in TLC file: {len(df_parquet):,}")
        
        # Standardize column names
        cols = {c: c.lower() for c in df_parquet.columns}
        df_parquet.rename(columns=cols, inplace=True)
        
        # Sample records
        if len(df_parquet) > n_samples:
            sample_df = df_parquet.sample(n=n_samples, random_state=42).reset_index(drop=True)
        else:
            sample_df = df_parquet.reset_index(drop=True)
            
        records = []
        for idx, row in sample_df.iterrows():
            puloc = int(row.get('pulocationid', 161)) if pd.notnull(row.get('pulocationid')) else 161
            coords = TLC_ZONE_COORDS.get(puloc, {'zone_name': 'Midtown & Times Square', 'lat': 40.7580, 'lon': -73.9855})
            
            # Add small random jitter to zone centroid
            lat = round(coords['lat'] + np.random.normal(0, 0.005), 5)
            lon = round(coords['lon'] + np.random.normal(0, 0.005), 5)
            
            trip_miles = float(row.get('trip_miles', row.get('trip_distance', 2.5)))
            trip_miles = np.clip(round(trip_miles, 2), 0.5, 25.0)
            
            base_fare = float(row.get('base_passenger_fare', row.get('fare_amount', 3.00 + 2.50*trip_miles)))
            base_fare = np.clip(round(base_fare, 2), 4.0, 150.0)
            
            records.append({
                'pickup_id': f"TLC_{puloc}_{10000 + idx}",
                'zone_name': coords['zone_name'],
                'pickup_lat': lat,
                'pickup_lon': lon,
                'trip_distance_miles': trip_miles,
                'base_fare_usd': base_fare
            })
            
        return pd.DataFrame(records)
    except Exception as e:
        print(f"[NYC TLC Loader Error] Could not parse Parquet file ({e})")
        return None

if __name__ == "__main__":
    df = load_official_nyc_tlc_pickups(n_samples=500, year=2024, month=1, trip_type='yellow')
    if df is not None:
        print("\nSampled Official NYC TLC Trip Records:")
        print(df.head())
