import pandas as pd
import numpy as np
from src.data_generator import generate_hybrid_marketplace_dataset
from src.clustering import detect_spatial_hotspots
from src.elasticity import solve_optimal_surge_multiplier
from src.metrics import calculate_marketplace_kpis, compare_baseline_vs_optimized

def compute_cluster_surge_series(df_slice, price_sensitivity_k=2.2, max_churn_threshold=0.30):
    """
    Computes per-request optimized surge multipliers using DBSCAN spatial clustering and 
    SciPy elasticity solver matching app.py control room logic.
    """
    hourly_requests = len(df_slice)
    hourly_drivers = df_slice['hourly_active_drivers'].mean() if not df_slice.empty else 100
    global_sd_ratio = hourly_drivers / max(1, hourly_requests)
    avg_weather_sev = df_slice['weather_severity'].mean()

    # Solve optimal global surge
    global_opt = solve_optimal_surge_multiplier(
        base_fare=15.0, 
        price_sensitivity_k=price_sensitivity_k, 
        weather_severity=avg_weather_sev, 
        max_churn_threshold=max_churn_threshold,
        supply_demand_ratio=global_sd_ratio
    )
    opt_multiplier = global_opt['optimal_multiplier']

    # DBSCAN spatial clustering
    df_clustered, cluster_summary = detect_spatial_hotspots(df_slice, eps_km=0.35, min_samples=5)

    if not cluster_summary.empty:
        avg_req_in_slice = cluster_summary['request_count'].mean()
        cluster_surges = []
        for idx, c_row in cluster_summary.iterrows():
            c_req = c_row['request_count']
            density_factor = c_req / max(1, avg_req_in_slice)
            local_sd_ratio = max(0.15, global_sd_ratio / (0.4 + 0.8 * density_factor))
            
            cluster_opt = solve_optimal_surge_multiplier(
                base_fare=15.0,
                price_sensitivity_k=price_sensitivity_k,
                weather_severity=avg_weather_sev,
                max_churn_threshold=max_churn_threshold,
                supply_demand_ratio=local_sd_ratio
            )
            
            # SciPy solver optimal surge multiplier
            final_cluster_surge = cluster_opt['optimal_multiplier']
            if density_factor > 1.8 and final_cluster_surge < 1.15:
                final_cluster_surge = 1.25
            elif density_factor > 1.2 and final_cluster_surge < 1.10:
                final_cluster_surge = 1.18
            elif density_factor < 0.6 and global_sd_ratio > 0.8:
                final_cluster_surge = 1.0
                
            cluster_surges.append(final_cluster_surge)
            
        cluster_summary['recommended_surge'] = cluster_surges
        cluster_surge_map = dict(zip(cluster_summary['cluster_id'], cluster_summary['recommended_surge']))
        opt_surge_series = df_clustered['cluster_id'].map(lambda cid: cluster_surge_map.get(cid, opt_multiplier))
    else:
        df_clustered = df_slice.copy()
        opt_surge_series = pd.Series([opt_multiplier] * len(df_slice), index=df_slice.index)

    return df_clustered, opt_surge_series, cluster_summary if not cluster_summary.empty else pd.DataFrame()

def evaluate_local_area_clusters(df_peak):
    """
    Evaluates metrics broken down by Local Area (DBSCAN Spatial Clusters).
    """
    df_clustered, opt_surge_series, cluster_summary = compute_cluster_surge_series(df_peak)
    
    cluster_metrics = []
    for c_id in cluster_summary['cluster_id']:
        if c_id == -1:
            continue # Noise points
        sub_df = df_clustered[df_clustered['cluster_id'] == c_id]
        sub_surge = opt_surge_series[sub_df.index]
        
        reqs = len(sub_df)
        drivers = sub_df['hourly_active_drivers'].mean()
        ratio = reqs / max(1, drivers)
        rec_surge = sub_surge.iloc[0] if len(sub_surge) > 0 else 1.0
        
        comp = compare_baseline_vs_optimized(sub_df, sub_surge)
        
        cluster_metrics.append({
            'Cluster ID': c_id,
            'Requests': reqs,
            'Active Drivers': int(drivers),
            'Demand/Supply Ratio': round(ratio, 2),
            'Optimal Surge': f"{rec_surge}x",
            'Baseline Fulfillment': f"{comp['baseline']['fulfillment_rate']}%",
            'Optimized Fulfillment': f"{comp['optimized']['fulfillment_rate']}%",
            'Fulfillment Lift': f"{comp['fulfillment_lift_pct_pts']:+0.1f}%",
            'GMV Lift': f"{comp['gmv_lift_pct']:+0.1f}%",
            'Baseline NPS': comp['baseline']['nps_index_score'],
            'Optimized NPS': comp['optimized']['nps_index_score'],
            'NPS Lift': f"{comp['optimized']['nps_index_score'] - comp['baseline']['nps_index_score']:+d}"
        })
        
    return pd.DataFrame(cluster_metrics)

def evaluate_hourly_breakdown(df):
    """
    Evaluates metrics broken down by Hour of Day (0 to 23).
    """
    hourly_rows = []
    for hr in range(24):
        df_hr = df[df['hour'] == hr]
        if len(df_hr) == 0:
            continue
            
        df_clustered, opt_surge_series, _ = compute_cluster_surge_series(df_hr)
        comp = compare_baseline_vs_optimized(df_clustered, opt_surge_series)
        
        reqs = len(df_hr)
        drivers = df_hr['hourly_active_drivers'].mean()
        avg_surge = round(opt_surge_series.mean(), 2)
        
        hourly_rows.append({
            'Hour': f"{hr:02d}:00",
            'Requests': reqs,
            'Drivers': int(drivers),
            'Avg Surge': f"{avg_surge}x",
            'Base Fulfillment': f"{comp['baseline']['fulfillment_rate']}%",
            'Opt Fulfillment': f"{comp['optimized']['fulfillment_rate']}%",
            'Fulfillment Lift': f"{comp['fulfillment_lift_pct_pts']:+0.1f}%",
            'Base GMV ($)': comp['baseline']['total_gmv_usd'],
            'Opt GMV ($)': comp['optimized']['total_gmv_usd'],
            'GMV Lift': f"{comp['gmv_lift_pct']:+0.1f}%",
            'Base NPS': comp['baseline']['nps_index_score'],
            'Opt NPS': comp['optimized']['nps_index_score'],
            'NPS Lift': f"{comp['optimized']['nps_index_score'] - comp['baseline']['nps_index_score']:+d}"
        })
        
    return pd.DataFrame(hourly_rows)

def evaluate_daily_macro_aggregate(df):
    """
    Evaluates metrics aggregated at the full daily / multi-day 24-hour dataset level.
    """
    all_opt_surges = []
    all_df_clustered = []
    
    for hr in sorted(df['hour'].unique()):
        df_hr = df[df['hour'] == hr]
        df_c, surge_s, _ = compute_cluster_surge_series(df_hr)
        all_df_clustered.append(df_c)
        all_opt_surges.append(surge_s)
        
    df_combined = pd.concat(all_df_clustered, ignore_index=True)
    surge_combined = pd.concat(all_opt_surges, ignore_index=True)
    
    comp = compare_baseline_vs_optimized(df_combined, surge_combined)
    return comp

if __name__ == '__main__':
    print("="*85)
    print("      MARKETPLACE METRICS EVALUATION: LOCAL AREA, HOURLY & DAILY MACRO")
    print("="*85)
    
    print("\n[1] Ingesting 7-Day / 22k+ NYC Telemetry Dataset...")
    df = generate_hybrid_marketplace_dataset(days=7, base_requests_per_hour=135, seed=42)
    print(f"Total Requests Ingested: {len(df):,} GPS pickup logs across 168 hours.")
    
    print("\n[2] LOCAL AREA BREAKDOWN (DBSCAN Spatial Micro-Zones @ 18:00 Peak Evening Rush):")
    df_peak = df[df['hour'] == 18]
    df_local = evaluate_local_area_clusters(df_peak)
    print(df_local.to_string(index=False))
    
    print("\n[3] HOURLY BREAKDOWN (24-Hour Marketplace Cycle Sample):")
    df_hourly = evaluate_hourly_breakdown(df)
    sample_hours = df_hourly[df_hourly['Hour'].isin(['03:00', '08:00', '13:00', '18:00', '22:00'])]
    print(sample_hours.to_string(index=False))
    
    print("\n[4] DAILY / MACRO AGGREGATE EVALUATION (Full 24-Hour Dataset Horizon):")
    macro = evaluate_daily_macro_aggregate(df)
    b = macro['baseline']
    o = macro['optimized']
    print(f"  • Total Requests Ingested:   {b['total_requests']:,}")
    print(f"  • Baseline Fulfillment Rate: {b['fulfillment_rate']}%")
    print(f"  • Optimized Fulfillment Rate:{o['fulfillment_rate']}% (Fulfillment Lift: {macro['fulfillment_lift_pct_pts']:+0.1f} pts)")
    print(f"  • Baseline GMV (USD):        ${b['total_gmv_usd']:,.2f}")
    print(f"  • Optimized GMV (USD):       ${o['total_gmv_usd']:,.2f} (GMV Lift: {macro['gmv_lift_pct']:+0.1f}%)")
    print(f"  • Baseline NPS Index:        {b['nps_index_score']}")
    print(f"  • Optimized NPS Index:       {o['nps_index_score']} (NPS Index Lift: {o['nps_index_score'] - b['nps_index_score']:+d} points)")
    print(f"  • Rider Churn Rate:          {b['cancellation_rate']}% -> {o['cancellation_rate']}% (Churn Reduction: {b['cancellation_rate'] - o['cancellation_rate']:-0.1f} pts)")
    print("="*85)
