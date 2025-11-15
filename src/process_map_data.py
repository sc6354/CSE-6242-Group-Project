from pathlib import Path
import json
import pandas as pd
import polars as pl
import geopandas as gpd
import numpy as np

BASE = Path(__file__).resolve().parents[1]
USZIPS = BASE / "data" / "simplemaps_uszips_basicv1.911" / "uszips.csv"
ZCTA_ZIP = BASE / "data" / "geo_data" / "tl_2021_us_zcta520.zip"
MODEL_RESULTS = BASE / "data" / "output_data" / "test_data_with_predictions.csv"

ZCTA_PARQUET_IN = BASE / "data" / "processed_data" / "states_zcta.parquet"
ZCTA_PARQUET_OUT = BASE / "data" / "processed_data" / "tx_zcta_with_prices.parquet"
ZCTA_CSV_OUT = BASE / "data" / "processed_data" / "tx_zcta_with_prices.csv"
OUT_DIR = BASE / "data" / "processed_data"

OUT_DIR.mkdir(parents=True, exist_ok=True)

# load states
target_states = ["CA", "TX", "FL", "WA", "NY"]
uszips = pd.read_csv(USZIPS, dtype={"zip": "string"}).assign(zip=lambda d: d["zip"].str.zfill(5))
uszips_selected = uszips[uszips["state_id"].isin(target_states)]
#uszips_tx = uszips[uszips["state_id"] == "TX"]

# Load polygons
zcta = gpd.read_file(f"zip://{ZCTA_ZIP}")
if "ZCTA5CE20" in zcta.columns:
    zcta["zip"] = zcta["ZCTA5CE20"].astype(str).str.zfill(5)
elif "ZCTA5CE10" in zcta.columns:
    zcta["zip"] = zcta["ZCTA5CE10"].astype(str).str.zfill(5)
else:
    raise ValueError("Couldn't find ZCTA5 column in ZCTA file.")

zcta = zcta.to_crs(epsg=4326)

# join to keep only target ZCTAs and a few attributes
keep_cols = ["zip", "population", "density", "city", "county_name","state_id"]
zcta_tx = zcta.merge(uszips[keep_cols], on="zip", how="inner")[["zip", "population", "density", "city", "county_name", "state_id", "geometry"]]

# save
gpq_path = OUT_DIR / "states_zcta.parquet"
geojson_path = OUT_DIR / "states_zcta.geojson"

zcta_tx.to_parquet(gpq_path)                   # best for Streamlit loads
zcta_tx.to_file(geojson_path, driver="GeoJSON")  # easiest for Plotly choropleth_mapbox

print("Wrote:", gpq_path)
print("Wrote:", geojson_path)

# ----- ADD ZIP TO PREDICTED DATASET -----
from sklearn.neighbors import BallTree

# find nearest ZIP assignment using BallTree
zips_tx = (
    pl.read_csv(USZIPS, dtypes={"zip": pl.Utf8})
     # .filter(pl.col("state_id") == "TX")
      .with_columns(pl.col("zip").str.zfill(5))
)

zip_coords = np.deg2rad(np.c_[zips_tx["lat"].to_numpy(), zips_tx["lng"].to_numpy()])
tree = BallTree(zip_coords, metric="haversine")

houses = pl.read_csv(MODEL_RESULTS)
house_coords = np.deg2rad(np.c_[houses["latitude_original"].to_numpy(),
                                houses["longitude_original"].to_numpy()])

dist_rad, idx = tree.query(house_coords, k=1)
nearest_zip = zips_tx["zip"].to_numpy()[idx.ravel()]

houses = houses.with_columns([
    pl.Series("zip", nearest_zip),
    pl.Series("zip_dist_km", (dist_rad.ravel() * 6371.0))
])

# ZIP-level aggregates
zip_agg = (
    houses.group_by("zip")
          .agg([
              pl.col("predicted_price").median().alias("pred_price_median"),
              pl.col("predicted_price").mean().alias("pred_price_mean"),
              pl.col("predicted_price").max().alias("pred_price_max"),
          ])
          .with_columns(pl.col("zip").cast(pl.Utf8).str.zfill(5))
)

# load TX ZCTA polygons (GeoParquet)
zcta_tx = gpd.read_parquet(ZCTA_PARQUET_IN)          # GeoDataFrame (WGS84), has 'zip' + geometry
zcta_tx["zip"] = zcta_tx["zip"].astype(str).str.zfill(5)

attr_cols = [c for c in zcta_tx.columns if c != "geometry"]
zcta_attrs_pl = pl.from_pandas(zcta_tx[attr_cols]).with_columns(pl.col("zip").cast(pl.Utf8).str.zfill(5))

joined_pl = zcta_attrs_pl.join(zip_agg, on="zip", how="left")
joined_pd = joined_pl.to_pandas()

plot_gdf = zcta_tx[["zip", "geometry"]].merge(joined_pd, on="zip", how="left")

# save
plot_gdf.to_parquet(ZCTA_PARQUET_OUT)
plot_gdf.to_csv(ZCTA_CSV_OUT)
print(f"Wrote {ZCTA_PARQUET_OUT}")