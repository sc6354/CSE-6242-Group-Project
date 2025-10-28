import json
import pandas as pd
import geopandas as gpd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from pathlib import Path

st.set_page_config(page_title="Texas ZIPs + Trader Joe's", layout="wide")
st.title("Texas ZIP Code Boundaries with Trader Joe’s Locations")

BASE = Path(__file__).resolve().parents[1]
TJ_PATH = BASE / "data" / "raw" / "tj-locations.csv"   # adjust if needed
GEOJSON_PATH = BASE / "data" / "processed_data" / "tx_zcta.geojson"  # or use Parquet below
#PARQUET_PATH = BASE / "data" / "processed_data" / "tx_zcta.parquet"
PARQUET_PATH = BASE / "data" / "processed_data" / "tx_zcta_with_prices.parquet"

# # --- Load tabular data ---
# uszips = pd.read_csv("data/simplemaps_uszips_basicv1.911/uszips.csv", dtype={"zip": str})
# tjs = pd.read_csv("data/raw/tj-locations.csv")
#
# # Keep TX rows for attributes + points
# uszips_tx = uszips[uszips["state_id"] == "TX"].copy()
# tjs_tx = tjs[tjs["state"].str.upper() == "TX"].copy()
#
# # --- Load ZCTA polygons (GeoPandas) ---
# # Use your local TIGER/Line ZCTA file (shp/gpkg/geojson). Any is fine.
# zcta = gpd.read_file("zip://data/geo_data/tl_2021_us_zcta520.zip")
# # Normalize ZIP/ZCTA codes as 5-char strings for joining
# if "ZCTA5CE20" in zcta.columns:
#     zcta["zip"] = zcta["ZCTA5CE20"].astype(str).str.zfill(5)
# elif "ZCTA5CE10" in zcta.columns:
#     zcta["zip"] = zcta["ZCTA5CE10"].astype(str).str.zfill(5)
# else:
#     raise ValueError("Couldn't find ZCTA5 column; check your file's fields.")
#
# zcta = zcta.to_crs(epsg=4326)  # Plotly expects WGS84
#
# # --- Join ZCTA polygons to SimpleMaps attributes to identify Texas ZIPs ---
# # (ZCTA ≈ ZIP; this join is standard practice for display)
# zcta_attrs = zcta.merge(
#     uszips_tx[["zip", "population", "density", "city", "county_name"]],
#     on="zip",
#     how="inner"
# )
# # --- Build GeoJSON for Plotly ---
# zcta_geojson = json.loads(zcta_attrs.to_json())

@st.cache_data
def load_tx_zcta_geojson():
    gdf = gpd.read_file(GEOJSON_PATH)   # already WGS84, TX-only, with attrs
    return gdf

@st.cache_data
def load_tx_zcta_parquet():
    return gpd.read_parquet(PARQUET_PATH)

@st.cache_data
def load_tj(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["state"] = df["state"].str.upper()
    df["zip"] = df["zip"].astype(str).str.zfill(5)
    return df[df["state"] == "TX"].copy()

# Choose either GeoJSON or Parquet (Parquet is faster)
zcta_tx = load_tx_zcta_parquet()
# zcta_tx = load_tx_zcta_geojson()

tjs_tx = load_tj(TJ_PATH)
# geojson = json.loads(zcta_tx.to_json())

# select box to choose a metric
# color_col = st.selectbox(
#     "Color ZIPs by:",
#     options=["population", "density"],
#     index=1
# )
color_col = st.selectbox(
    "Color ZIPs by:",
    options=["pred_price_median", "pred_price_mean", "pred_price_max"],
    index=0  # default to median predicted price
)

visible = zcta_tx[~zcta_tx[color_col].isna()]
geojson = json.loads(visible.to_json())

# --- Create Plotly figure with polygons ---
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
    mapbox_style="carto-positron",  # no Mapbox token needed
    center={"lat": 31.0, "lon": -99.0},  # center of Texas-ish
    zoom=4.5,
    opacity=0.6
)

# --- Add Trader Joe's point layer ---
# Use Scattermapbox for interactive points
fig.add_trace(
    go.Scattermapbox(
        lat=tjs_tx["latitude"],
        lon=tjs_tx["longitude"],
        mode="markers",
        marker=dict(size=8, color="red"),  # default color
        name="Trader Joe's",
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "%{customdata[1]}<br>"
            "%{customdata[2]}, %{customdata[3]} %{customdata[4]}<extra></extra>"
        ),
        customdata=tjs_tx[["name","street","city","state","zip"]].values
    )
)

fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})


st.plotly_chart(fig, use_container_width=True)

st.caption(
    "ZIP boundaries from Census ZCTA polygons; attributes from SimpleMaps uszips; "
    "Trader Joe’s locations overlaid as points."
)
