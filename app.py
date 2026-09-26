import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as gg
import folium
from streamlit_folium import st_folium

from src.data_generator import generate_hybrid_marketplace_dataset
from src.clustering import detect_spatial_hotspots
from src.forecasting import build_demand_forecast_model, forecast_next_24h_demand
from src.elasticity import solve_optimal_surge_multiplier, compute_marketplace_fulfillment_curve
from src.metrics import calculate_marketplace_kpis, compare_baseline_vs_optimized

# Page Configuration
st.set_page_config(
    page_title="Peak Hour Surge Control Engine",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header { font-size: 2.2rem; font-weight: 700; color: #1E293B; margin-bottom: 0.2rem; }
    .sub-header { font-size: 1.0rem; color: #64748B; margin-bottom: 1.5rem; }
    .metric-card { background-color: #F8FAFC; border-radius: 8px; padding: 15px; border-left: 5px solid #3B82F6; }
    .stMetric label { font-size: 0.9rem !important; color: #475569 !important; }
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("<div class='main-header'>Urban Mobility Dynamic Surge Pricing & Supply Allocation Engine</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-header'>Real-Time Geospatial Hotspot Detection, Price Elasticity Optimization & 2-Sided Marketplace Control Room</div>", unsafe_allow_html=True)

# Cached Dataset Generation
@st.cache_data(ttl=3600)
def load_cached_data():
    return generate_hybrid_marketplace_dataset(days=5, base_requests_per_hour=140, seed=42)

df_all = load_cached_data()

# Sidebar Control Panel
st.sidebar.header("Control Room Parameters")

selected_hour = st.sidebar.slider("Select Hour of Day (0-23)", 0, 23, 18, help="Peak hours: 8 AM, 18 PM")
weather_override = st.sidebar.selectbox("Weather Condition", ["Default API Weather", "Clear (0.0mm)", "Moderate Rain (3.5mm)", "Heavy Storm (8.0mm)"])
max_churn_threshold = st.sidebar.slider("Max Cancellation Threshold (%)", 15, 45, 30) / 100.0
price_sensitivity_k = st.sidebar.slider("Rider Price Sensitivity (k)", 1.0, 3.5, 2.2, step=0.1)

# Apply Weather Override
df_filtered = df_all[df_all['hour'] == selected_hour].copy()

if weather_override == "Clear (0.0mm)":
    df_filtered['weather_severity'] = 0.0
    df_filtered['precipitation_mm'] = 0.0
elif weather_override == "Moderate Rain (3.5mm)":
    df_filtered['weather_severity'] = 0.35
    df_filtered['precipitation_mm'] = 3.5
elif weather_override == "Heavy Storm (8.0mm)":
    df_filtered['weather_severity'] = 0.80
    df_filtered['precipitation_mm'] = 8.0

avg_weather_sev = df_filtered['weather_severity'].mean()

# Solve Optimal Surge Multiplier M*
opt_solution = solve_optimal_surge_multiplier(
    base_fare=15.0, 
    price_sensitivity_k=price_sensitivity_k, 
    weather_severity=avg_weather_sev, 
    max_churn_threshold=max_churn_threshold
)
opt_multiplier = opt_solution['optimal_multiplier']

# Run DBSCAN Clustering
df_clustered, cluster_summary = detect_spatial_hotspots(df_filtered, eps_km=0.8, min_samples=10)

# Calculate KPIs & Comparison
kpi_comparison = compare_baseline_vs_optimized(df_filtered, opt_multiplier)
base_kpis = kpi_comparison['baseline']
opt_kpis = kpi_comparison['optimized']

# KPI Scorecards
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Optimal Surge Multiplier", f"{opt_multiplier}x", f"{round(opt_multiplier - 1.0, 2)}x boost")
c2.metric("Marketplace Fulfillment", f"{opt_kpis['fulfillment_rate']}%", f"{kpi_comparison['fulfillment_lift_pct_pts']:+} pts vs baseline")
c3.metric("Rider Churn Rate", f"{opt_kpis['cancellation_rate']}%", f"{round(opt_kpis['cancellation_rate'] - base_kpis['cancellation_rate'], 1):+} pts")
c4.metric("Total GMV (Hourly)", f"${opt_kpis['total_gmv_usd']:,.2f}", f"{kpi_comparison['gmv_lift_pct']:+} % GMV Lift")
c5.metric("NPS Impact Score Index", f"{opt_kpis['nps_index_score']} / 100", f"{opt_kpis['nps_index_score'] - base_kpis['nps_index_score']:+} pts")

st.markdown("---")

# Main Navigation Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "Live Geospatial Control Room", 
    "Price Elasticity & Surge Simulator", 
    "Executive KPIs & Unit Economics", 
    "Demand Forecast & Hotspot Analytics"
])

# TAB 1: Live Geospatial Control Room
with tab1:
    st.subheader(f"Geospatial Demand Clusters & Dispatch Zones at {selected_hour:02d}:00")
    
    col_map, col_details = st.columns([2.2, 1.0])
    
    with col_map:
        # Use OpenStreetMap tiles for 100% free open-access maps with zero API key requirement
        nyc_map = folium.Map(location=[40.730610, -73.935242], zoom_start=11, tiles="OpenStreetMap")
        colors = ['#EF4444', '#F97316', '#F59E0B', '#10B981', '#6366F1', '#EC4899', '#8B5CF6']
        
        if not cluster_summary.empty:
            for idx, c_row in cluster_summary.iterrows():
                c_color = colors[int(c_row['cluster_id']) % len(colors)]
                surge_tier = round(max(1.0, opt_multiplier * (1.0 + 0.05 * idx)), 2)
                
                folium.CircleMarker(
                    location=[c_row['centroid_lat'], c_row['centroid_lon']],
                    radius=12 + (c_row['request_count'] / 8),
                    color=c_color,
                    fill=True,
                    fill_color=c_color,
                    fill_opacity=0.6,
                    popup=f"<b>Hotspot Cluster #{c_row['cluster_id']}</b><br>"
                          f"Zone: {c_row['primary_zone']}<br>"
                          f"Ride Requests: {c_row['request_count']}<br>"
                          f"Cancellation Rate: {c_row['cancellation_rate']*100:.1f}%<br>"
                          f"<b>Recommended Surge: {surge_tier}x</b>"
                ).add_to(nyc_map)
                
        st_folium(nyc_map, width=800, height=480)
        
    with col_details:
        st.write("### Active Hotspots Summary")
        if not cluster_summary.empty:
            disp_df = cluster_summary[['cluster_id', 'primary_zone', 'request_count', 'cancellation_rate']].copy()
            disp_df.columns = ['ID', 'Zone', 'Requests', 'Churn Rate']
            disp_df['Churn Rate'] = (disp_df['Churn Rate'] * 100).round(1).astype(str) + "%"
            st.dataframe(disp_df, hide_index=True, use_container_width=True)
        else:
            st.info("No dense clusters detected for this hour slice. Demand is evenly distributed.")
            
        st.write("### Environment Status")
        st.info(f"Precipitation: {df_filtered['precipitation_mm'].mean():.1f} mm/hr\n\n"
                f"Avg Temperature: {df_filtered['temperature_c'].mean():.1f} °C\n\n"
                f"Active Fleet Drivers: {df_filtered['hourly_active_drivers'].mean():.0f} drivers")

# TAB 2: Price Elasticity & Surge Simulator
with tab2:
    st.subheader("Rider Price Elasticity vs Driver Acceptance Curves")
    st.markdown("This simulator models how rider cancellation probability increases with higher surge multipliers, while driver trip acceptance probability responds positively to surge incentives.")
    
    curve_df = opt_solution['curve_df']
    fig_curve = gg.Figure()
    
    fig_curve.add_trace(gg.Scatter(
        x=curve_df['surge_multiplier'], y=curve_df['rider_cancel_prob'] * 100,
        mode='lines', name='Rider Cancellation Rate (%)',
        line=dict(color='#EF4444', width=3)
    ))
    
    fig_curve.add_trace(gg.Scatter(
        x=curve_df['surge_multiplier'], y=curve_df['driver_accept_prob'] * 100,
        mode='lines', name='Driver Acceptance Rate (%)',
        line=dict(color='#10B981', width=3, dash='dash')
    ))
    
    fig_curve.add_trace(gg.Scatter(
        x=curve_df['surge_multiplier'], y=curve_df['fulfillment_rate'] * 100,
        mode='lines', name='Expected Fulfillment Rate (%)',
        line=dict(color='#3B82F6', width=4)
    ))
    
    fig_curve.add_vline(
        x=opt_multiplier, line_width=2, line_dash="dot", line_color="#8B5CF6",
        annotation_text=f"Optimal M* = {opt_multiplier}x", annotation_position="top left"
    )
    
    fig_curve.add_hline(
        y=max_churn_threshold * 100, line_width=1.5, line_dash="dash", line_color="#DC2626",
        annotation_text=f"Max Churn Limit ({int(max_churn_threshold*100)}%)", annotation_position="bottom right"
    )
    
    fig_curve.update_layout(
        title="Marketplace Equilibrium Curve",
        xaxis_title="Surge Multiplier (x)",
        yaxis_title="Percentage (%)",
        hovermode="x unified",
        height=450
    )
    
    st.plotly_chart(fig_curve, use_container_width=True)

# TAB 3: Executive KPIs & Unit Economics
with tab3:
    st.subheader("Marketplace Financial & Operational Performance Comparison")
    
    comp_data = {
        'Metric': ['Total Hourly Requests', 'Fulfilled Trips', 'Marketplace Fulfillment Rate', 'Rider Cancellation Rate', 'Hourly Gross Merchandise Value (GMV)', 'Driver Hourly Earnings Rate', 'Customer NPS Index'],
        'Baseline (Unoptimized)': [
            f"{base_kpis['total_requests']:,}",
            f"{base_kpis['fulfilled_trips']:,}",
            f"{base_kpis['fulfillment_rate']}%",
            f"{base_kpis['cancellation_rate']}%",
            f"${base_kpis['total_gmv_usd']:,.2f}",
            f"${base_kpis['driver_hourly_rate_usd']:.2f}/hr",
            f"{base_kpis['nps_index_score']} / 100"
        ],
        'Optimized (Peak Hour Surge Control)': [
            f"{opt_kpis['total_requests']:,}",
            f"{opt_kpis['fulfilled_trips']:,}",
            f"{opt_kpis['fulfillment_rate']}%",
            f"{opt_kpis['cancellation_rate']}%",
            f"${opt_kpis['total_gmv_usd']:,.2f}",
            f"${opt_kpis['driver_hourly_rate_usd']:.2f}/hr",
            f"{opt_kpis['nps_index_score']} / 100"
        ],
        'Impact / Lift': [
            "-",
            f"+{opt_kpis['fulfilled_trips'] - base_kpis['fulfilled_trips']:,} trips",
            f"{kpi_comparison['fulfillment_lift_pct_pts']:+} pts",
            f"{round(opt_kpis['cancellation_rate'] - base_kpis['cancellation_rate'], 1):+} pts",
            f"+{kpi_comparison['gmv_lift_pct']}% GMV",
            f"+${round(opt_kpis['driver_hourly_rate_usd'] - base_kpis['driver_hourly_rate_usd'], 2)}/hr",
            f"+{opt_kpis['nps_index_score'] - base_kpis['nps_index_score']} pts"
        ]
    }
    
    st.table(pd.DataFrame(comp_data))

# TAB 4: Demand Forecast & Hotspot Analytics
with tab4:
    forecast_model, hourly_df = build_demand_forecast_model(df_all)
    model_type_name = forecast_model['model_type']
    
    st.subheader(f"24-Hour Time-Series Demand Forecasting ({model_type_name} Engine)")
    
    forecast_24h = forecast_next_24h_demand(forecast_model, weather_severity_forecast=avg_weather_sev)
    
    fig_forecast = px.line(
        forecast_24h, x='hour', y='predicted_demand',
        title=f"Predicted Hourly Ride Demand Curve (Next 24 Hours via {model_type_name})",
        labels={'hour': 'Hour of Day (0-23)', 'predicted_demand': 'Predicted Ride Requests'},
        markers=True
    )
    fig_forecast.update_traces(line_color='#8B5CF6', line_width=3)
    fig_forecast.update_layout(height=400)
    
    st.plotly_chart(fig_forecast, use_container_width=True)
    
    st.subheader("Historical Demand Density by NYC Zone")
    zone_counts = df_all['zone_name'].value_counts().reset_index()
    zone_counts.columns = ['NYC Micro-Zone', 'Total Requests']
    
    fig_zone = px.bar(zone_counts, x='NYC Micro-Zone', y='Total Requests', color='Total Requests', color_continuous_scale='Blues')
    fig_zone.update_layout(height=350)
    st.plotly_chart(fig_zone, use_container_width=True)
