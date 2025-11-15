import json
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

# -----------------------------------------------------------------------------
# PAGE SETUP
# -----------------------------------------------------------------------------
st.set_page_config(page_title="ZIPs + Trader Joe's", layout="wide")
st.title("ZIP Code Boundaries with Trader Joe’s Locations")
st.markdown(
    """
    **Our project aims to predict housing prices by analyzing real estate data 
    with a focus on proximity to Trader Joe’s stores.**

    
    """
)

# -----------------------------------------------------------------------------
# FILE PATHS
# -----------------------------------------------------------------------------
BASE = Path(__file__).resolve().parents[1]
TJ_PATH = BASE / "data" / "raw" / "tj-locations.csv"
GEOJSON_PATH = BASE / "data" / "processed_data" / "tx_zcta.geojson"
PARQUET_PATH = BASE / "data" / "processed_data" / "tx_zcta_with_prices.parquet"

USZIPS_PATH = BASE / "data" / "simplemaps_uszips_basicv1.911" / "uszips.csv"

# -----------------------------------------------------------------------------
# 
# -----------------------------------------------------------------------------
@st.cache_data
def load_tx_zcta_geojson():
    gdf = gpd.read_file(GEOJSON_PATH)   # 
    return gdf

@st.cache_data
def load_tx_zcta_parquet():
    return gpd.read_parquet(PARQUET_PATH)

@st.cache_data
def load_tj(path: Path) -> pd.DataFrame:
    # Keep only the 5 states 
    target_states = ["CA", "TX", "FL", "WA", "NY"]
    df = pd.read_csv(path)
    df["state"] = df["state"].str.upper()
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df[df["state"].isin(target_states)].copy()

# -----------------------------------------------------------------------------
# LOAD ZCTAs + ADD STATE INFO VIA USZIPS MERGE
# -----------------------------------------------------------------------------
zcta_all = load_tx_zcta_parquet()   # currently has ZIPs + prices + density, etc.

if "zip" not in zcta_all.columns:
    raise ValueError("Expected a 'zip' column in the ZCTA parquet to join on.")

# Load SimpleMaps uszips to get state_id
uszips = (
    pd.read_csv(USZIPS_PATH, dtype={"zip": "string"})
      .assign(zip=lambda d: d["zip"].str.zfill(5))
)

# 
if "state_id" not in zcta_all.columns and "state" not in zcta_all.columns:
    zcta_all = zcta_all.merge(
        uszips[["zip", "state_id"]],
        on="zip",
        how="left"
    )
    zcta_all["state_id"] = zcta_all["state_id"].str.upper()

# Decide which column to treat as the state code
if "state_id" in zcta_all.columns:
    state_col = "state_id"
elif "state" in zcta_all.columns:
    state_col = "state"
    zcta_all[state_col] = zcta_all[state_col].str.upper()
else:
    raise ValueError("No state column found in ZCTA parquet after merge. Expected 'state_id' or 'state'.")

# -----------------------------------------------------------------------------
# LOAD TRADER JOE'S
# -----------------------------------------------------------------------------
tjs_all = load_tj(TJ_PATH)

# -----------------------------------------------------------------------------
# STATE CONFIG
# -----------------------------------------------------------------------------
state_options = ["California", "Texas", "Florida", "Washington", "New York"]
state_to_code = {
    "California": "CA",
    "Texas": "TX",
    "Florida": "FL",
    "Washington": "WA",
    "New York": "NY",
}

STATE_MAP_CONFIG = {
    "CA": {"lat": 36.7, "lon": -119.4, "zoom": 4.5},
    "TX": {"lat": 31.0, "lon": -99.0,  "zoom": 4.5},
    "FL": {"lat": 27.8, "lon": -81.7,  "zoom": 5.1},
    "WA": {"lat": 47.4, "lon": -120.7, "zoom": 5.0},
    "NY": {"lat": 43.0, "lon": -75.0,  "zoom": 5.2},
}

# -----------------------------------------------------------------------------
# CONTROLS
# -----------------------------------------------------------------------------
selected_state_name = st.selectbox(
    "Select state:",
    options=state_options,
    index=1  # default to Texas
)
selected_state_code = state_to_code[selected_state_name]

color_col = st.selectbox(
    "Color ZIPs by:",
    options=["pred_price_median", "pred_price_mean", "pred_price_max"],
    index=0
)

# -----------------------------------------------------------------------------
# FILTER ZCTAs + TRADER JOE'S TO SELECTED STATE
# -----------------------------------------------------------------------------
zcta_state = zcta_all[zcta_all[state_col] == selected_state_code].copy()

# Only ZIPs with non-null measure
visible = zcta_state[~zcta_state[color_col].isna()].copy()
geojson = json.loads(visible.to_json())

tjs_state = tjs_all[tjs_all["state"] == selected_state_code].copy()

# -----------------------------------------------------------------------------
# STATE OVERVIEW CARD
# -----------------------------------------------------------------------------
avg_pop_density = visible["density"].mean() if "density" in visible.columns else float("nan")
median_pred_price = visible["pred_price_median"].median() if "pred_price_median" in visible.columns else float("nan")
max_pred_price = visible["pred_price_median"].max() if "pred_price_median" in visible.columns else float("nan")
min_pred_price = visible["pred_price_median"].min() if "pred_price_median" in visible.columns else float("nan")
total_tjs = len(tjs_state)

avg_pop_density_str = f"{avg_pop_density:,.0f}" if pd.notna(avg_pop_density) else "N/A"
median_pred_price_str = f"${median_pred_price:,.0f}" if pd.notna(median_pred_price) else "N/A"
max_pred_price_str = f"${max_pred_price:,.0f}" if pd.notna(max_pred_price) else "N/A"
min_pred_price_str = f"${min_pred_price:,.0f}" if pd.notna(min_pred_price) else "N/A"
total_tjs_str = f"{total_tjs:,}"

st.markdown("""
<style>
.fixed-card {
    position: absolute;
    top: 220px;
    left: 17px;
    z-index: 1000;
    width: 30%;
    max-width: 320px;
    min-width: 220px;
    padding: 15px;
    background-color: rgba(255, 255, 255, 0.75);
    backdrop-filter: blur(5px);
    border-radius: 12px;
    box-shadow: 0 6px 15px rgba(0, 0, 0, 0.2);
    font-family: 'Inter', sans-serif;
}
/* Force dark text for title + description, even in dark theme */
.fixed-card h4,
.fixed-card p {
    color: #000000;
}
.card-metric {
    display: flex;
    justify-content: space-between;
    padding: 5px 0;
    border-bottom: 1px solid rgba(0, 0, 0, 0.1);
}
.card-metric:last-child {
    border-bottom: none;
}
.metric-label {
    font-weight: 500;
    font-size: 0.9rem;
    color: #4B4B4B;
}
.metric-value {
    font-weight: 700;
    font-size: 1rem;
    color: #000000;
}
</style>
""", unsafe_allow_html=True)

overlay_card_html = f"""
<div class="fixed-card">
    <h4 style="margin: 0 0 10px 0;">State Overview: {selected_state_name}</h4>
    <p style="font-size:0.85rem; margin-bottom:8px;">
        Based on our analysis, each additional mile farther from the nearest Trader Joe’s is associated with 
        approximately a <b>2.5–2.7% decrease</b> in home prices, holding other features constant.
    </p>
    <div class="card-metric">
        <span class="metric-label">Trader Joe's locations:</span>
        <span class="metric-value">{total_tjs_str}</span>
    </div>
    <div class="card-metric">
        <span class="metric-label">Average population density:</span>
        <span class="metric-value">{avg_pop_density_str} ppl/mi²</span>
    </div>
    <div class="card-metric">
        <span class="metric-label">Median predicted price:</span>
        <span class="metric-value">{median_pred_price_str}</span>
    </div>
    <div class="card-metric">
        <span class="metric-label">Max predicted price:</span>
        <span class="metric-value">{max_pred_price_str}</span>
    </div>
    <div class="card-metric">
        <span class="metric-label">Min predicted price:</span>
        <span class="metric-value">{min_pred_price_str}</span>
    </div>
</div>
"""
st.markdown(overlay_card_html, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# MAP: STATE-SPECIFIC ZIPS, BOUNDARY, TRADER JOE'S
# -----------------------------------------------------------------------------
map_cfg = STATE_MAP_CONFIG[selected_state_code]

# Compute global range across all states once per app run
global_min = zcta_all[color_col].min()
global_max = zcta_all[color_col].max()

fig = px.choropleth_mapbox(
    visible,
    geojson=geojson,
    locations="zip",
    featureidkey="properties.zip",
    color=color_col,
    hover_name="zip",
    hover_data={
        "city": True,
        "county_name": True,
        "population": True,
        "density": True,
        "pred_price_median": True,
        "pred_price_mean": True,
        "pred_price_max": True,
        "zip": False
    },
    mapbox_style="carto-positron",
    center={"lat": map_cfg["lat"], "lon": map_cfg["lon"]},
    zoom=map_cfg["zoom"],
    opacity=0.6,
    color_continuous_scale="Blues",  # dark blue = high, light blue = low
    range_color=(global_min, global_max),  # same range for all states
)

# State boundary from that state's ZCTAs
state_boundary = gpd.GeoDataFrame(
    {"state": [selected_state_code]},
    geometry=[zcta_state.geometry.unary_union],
    crs=zcta_state.crs
)
state_geojson = json.loads(state_boundary.to_json())

fig.add_trace(
    go.Choroplethmapbox(
        geojson=state_geojson,
        locations=state_boundary["state"],
        featureidkey="properties.state",
        z=[1],
        colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,0,0,0)"]],
        showscale=False,
        marker_opacity=1,
        marker_line_width=1.5,
        marker_line_color="black",
        name=f"{selected_state_name} Boundary",
        hoverinfo="skip"
    )
)

# Trader Joe’s red dots for selected state
fig.add_trace(
    go.Scattermapbox(
        lat=tjs_state["latitude"],
        lon=tjs_state["longitude"],
        mode="markers",
        marker=dict(size=8, color="red"),
        name="Trader Joe's",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "%{customdata[2]}, %{customdata[3]} %{customdata[4]}<extra></extra>"
        ),
        customdata=tjs_state[["name", "street", "city", "state", "zip"]].values
    )
)
fig.update_layout(coloraxis_reversescale=False)
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

st.plotly_chart(fig, use_container_width=True)

st.caption(
    f"Current view: {selected_state_name}. ZIP boundaries from Census ZCTA polygons; "
    "attributes from SimpleMaps uszips; Trader Joe’s locations shown as red dots."
)
