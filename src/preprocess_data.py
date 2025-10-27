############################################################
# preprocess_data.py
############################################################
# This script preprocesses housing data to prepare it for models
# to predict prices. It implements the following :
# 1. Load your housing data and the Trader Joe's zip code data.
# 2. Geospatial Feature Encoding: Converts city and state into
#    latitude and longitude to create powerful, low-dimensional features.
# 3. Create the 'is_near_trader_joes' feature if a zip code is 
#    less than or equal to 25 miles away from a Traders Joes.
# 4. Create the 'min_distance_to_tj' feature representing the
#    minimum distance from each property to the nearest Trader Joe's.
# 5. Filters the data to only include properties from
#    California, Texas, Florida, Washington, and New York and with a 
#    previous sale date in 2022.
# 6. Imputation for missing lot size based on median lot size.
# 7. Imputation for missing house size based on median house size
#    Imputation for missing bed/bath counts based on median bed/bath counts.
# 8. Imputation for missing prices based on bed/bath count.
# 9. Scaling and encoding using RobustScaler and OneHotEncoder and included
#    the original, unscaled numeric columns for reference/visualization later.
# 10. A random 80/20 split for training and testing.
# 11. Save the final, model-ready datasets to new CSV files.
############################################################

import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree
from geopy.distance import geodesic

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# --- Configuration ---

# Data files
HOUSING_DATA_FILE = "data/raw/realtor-data.zip.csv"
TRADER_JOES_ZIPS_FILE = "data/raw/tj-locations.csv"
ZIP_COORDS_FILE = "data/simplemaps_uszips_basicv1.911/uszips.csv"

# Output directory
OUTPUT_DIR = "data/processed_data"

# Define column names
COL_PRICE = "price"
COL_BED = "bed"
COL_BATH = "bath"
COL_ACRE_LOT = "acre_lot"
COL_CITY = "city"
COL_STATE = "state"
COL_ZIP = "zip_code"
COL_HOUSE_SIZE = "house_size"
COL_SOLD_DATE = "prev_sold_date"
COL_IS_NEAR_TJ = "is_near_trader_joes" 
COL_LAT = "latitude"
COL_LON = "longitude"
COL_PREV_SOLD = "is_Sold"
COL_PREV_SOLD_YEAR = "prev_sold_year"
COL_MIN_DIST_TJ = "min_distance_to_tj" 
COL_NEAREST_TJ_LAT = "nearest_tj_lat"
COL_NEAREST_TJ_LON = "nearest_tj_lon"


# Define filters/thresholds
STATES_TO_KEEP = ["California", "Texas", "Florida", "Washington", "New York"]
YEAR_TO_KEEP = 2022
# Distance threshold to consider "near" a Trader Joe's
DISTANCE_THRESHOLD_MILES = 25

def main():
    """Main function to execute the preprocessing workflow."""
    print("Starting data preprocessing...")

    # --- 1. Data Loading ---
    if not os.path.exists(HOUSING_DATA_FILE) or not os.path.exists(ZIP_COORDS_FILE) or not os.path.exists(TRADER_JOES_ZIPS_FILE):
        print(f"Error: Make sure all data files are in the correct directories.")
        return

    df_housing_raw = pd.read_csv(HOUSING_DATA_FILE)
    df_tj_zips_raw = pd.read_csv(TRADER_JOES_ZIPS_FILE)
    df_zip_coords = pd.read_csv(ZIP_COORDS_FILE)
    print(f"Loaded {len(df_housing_raw)} total housing records.")

    # --- 2. Preprocess Coordinates Data ---
    df_zip_coords['zip'] = df_zip_coords['zip'].astype(str).str.zfill(5)
    df_zip_coords = df_zip_coords[['zip', 'lat', 'lng']].copy()
    df_zip_coords.dropna(subset=['lat', 'lng'], inplace=True)

    # --- 3. Preprocess Trader Joe's Locations ---
    print("Preprocessing Trader Joe's locations...")
    df_tj_zips_raw['zip'] = df_tj_zips_raw['zip'].astype(str).str.extract(r'(\d{5})', expand=False)
    df_tj_zips_raw.dropna(subset=['zip'], inplace=True)
    df_tj_locations = pd.merge(df_tj_zips_raw, df_zip_coords, on='zip', how='left')
    df_tj_locations.dropna(subset=['lat', 'lng'], inplace=True)
    # Store the coordinates in a numpy array for KDTree
    tj_coords = df_tj_locations[['lat', 'lng']].values
    print(f"Found coordinates for {len(tj_coords)} Trader Joe's locations.")
    if len(tj_coords) == 0:
        print("Error: Could not find coordinates for any Trader Joe's locations.")
        return

    # Build KDTree for efficient nearest neighbor search
    tree = cKDTree(np.radians(tj_coords)) # Use radians for haversine distance

    # --- 4. Preprocess Housing Data ---
    df_housing = df_housing_raw.copy()

    # Filter for selected states first
    df_housing = df_housing[df_housing[COL_STATE].isin(STATES_TO_KEEP)].copy()
    print(f"Filtered for states. {len(df_housing)} records remaining.")

    # Add house coordinates
    df_housing[COL_ZIP] = df_housing[COL_ZIP].astype(str).str.extract(r'(\d{5})', expand=False)
    df_housing = pd.merge(df_housing, df_zip_coords, left_on=COL_ZIP, right_on='zip', how='left')
    df_housing.rename(columns={'lat': COL_LAT, 'lng': COL_LON}, inplace=True)
    df_housing.dropna(subset=[COL_LAT, COL_LON], inplace=True) # Crucial: only keep houses we can geocode
    print("Added house coordinates.")

    # Calculate distance to nearest TJ
    print("Calculating distance to nearest Trader Joe's for each house...")
    house_coords_rad = np.radians(df_housing[[COL_LAT, COL_LON]].values)
    
    # query returns distance (in radians) and index of nearest neighbor
    # Capture the indices to find the coordinates
    distances_rad, indices = tree.query(house_coords_rad, k=1)
    
    # Get the coordinates of the nearest TJ store using the indices
    nearest_tj_coords = tj_coords[indices]
    df_housing[COL_NEAREST_TJ_LAT] = nearest_tj_coords[:, 0]
    df_housing[COL_NEAREST_TJ_LON] = nearest_tj_coords[:, 1]
    print("Added nearest Trader Joe's coordinates.")

    # Convert radians distance to miles (using Earth's radius)
    earth_radius_miles = 3958.8 # Mean radius
    df_housing[COL_MIN_DIST_TJ] = distances_rad * earth_radius_miles
    print("Distance calculation complete.")

    # Create the is_near_trader_joes flag based on distance
    df_housing[COL_IS_NEAR_TJ] = (df_housing[COL_MIN_DIST_TJ] <= DISTANCE_THRESHOLD_MILES).astype(int)
    print(f"Identified {df_housing[COL_IS_NEAR_TJ].sum()} properties within {DISTANCE_THRESHOLD_MILES} miles of a Trader Joe's.")

    # --- 5. Further Feature Engineering & Filtering ---
    df_housing[COL_PREV_SOLD] = df_housing[COL_SOLD_DATE].notnull().astype(int)

    if df_housing[COL_PRICE].dtype == 'object':
        df_housing[COL_PRICE] = df_housing[COL_PRICE].str.replace(r'[$,]', '', regex=True).astype(float)

    df_housing[COL_SOLD_DATE] = pd.to_datetime(df_housing[COL_SOLD_DATE], errors='coerce')
    df_housing[COL_PREV_SOLD_YEAR] = df_housing[COL_SOLD_DATE].dt.year
    df_housing[COL_PREV_SOLD_YEAR] = df_housing[COL_PREV_SOLD_YEAR].fillna(0).astype(int)
    
    df_housing['month_sold'] = df_housing[COL_SOLD_DATE].dt.month
    df_housing['month_sold'] = df_housing['month_sold'].fillna(0).astype(int)

    # Filter for the specified year
    year_filter_count = len(df_housing)
    df_housing = df_housing[(df_housing[COL_PREV_SOLD_YEAR] == YEAR_TO_KEEP)].copy()
    if df_housing.empty:
        print(f"Error: No data found for properties sold in {YEAR_TO_KEEP} in the selected states.")
        return
    print(f"Filtered for year {YEAR_TO_KEEP}. {len(df_housing)} of {year_filter_count} records remaining.")

    # --- 6. Data Cleaning, Imputation & Splitting ---
    print("Imputing missing or zero prices...")
    invalid_price_mask = (df_housing[COL_PRICE].isnull()) | (df_housing[COL_PRICE] <= 0)
    if invalid_price_mask.sum() > 0:
        df_housing['median_price_impute'] = df_housing.loc[~invalid_price_mask].groupby(
            [COL_BED, COL_BATH], observed=True
        )[COL_PRICE].transform('median')
        df_housing[COL_PRICE] = df_housing[COL_PRICE].mask(invalid_price_mask, df_housing['median_price_impute'])
        if df_housing[COL_PRICE].isnull().any():
            global_median = df_housing.loc[~invalid_price_mask, COL_PRICE].median()
            df_housing[COL_PRICE].fillna(global_median, inplace=True)
        df_housing.drop(columns=['median_price_impute'], inplace=True)
        print("Imputation complete.")
        
    # Impute missing lot sizes
    median_lot_size = df_housing[COL_ACRE_LOT].median()
    df_housing[COL_ACRE_LOT].fillna(median_lot_size, inplace=True)
    
    # Impute missing house sizes
    median_house_size = df_housing[COL_HOUSE_SIZE].median()
    df_housing[COL_HOUSE_SIZE].fillna(median_house_size, inplace=True)
    
    # Impute missing bed counts
    median_bed = df_housing[COL_BED].median()
    df_housing[COL_BED].fillna(median_bed, inplace=True)
    
    # Impute missing bath counts
    median_bath = df_housing[COL_BATH].median()
    df_housing[COL_BATH].fillna(median_bath, inplace=True)

    print(f"Data cleaned. {len(df_housing)} records remaining.")

    train_df, test_df = train_test_split(df_housing, test_size=0.2, random_state=42)
    print(f"Split data randomly: {len(train_df)} training, {len(test_df)} test records.")

    # --- 7. Preprocessing Pipeline ---
    
    # --- MODIFIED SECTION ---
    # Add the new nearest_tj coordinates to the list of predictors
    numerical_predictors = [
        COL_BED, COL_BATH, COL_ACRE_LOT, COL_HOUSE_SIZE, 
        COL_LAT, COL_LON, COL_MIN_DIST_TJ,
        COL_NEAREST_TJ_LAT, COL_NEAREST_TJ_LON # Added new features
    ]
    # --- END MODIFIED SECTION ---
    
    categorical_predictors = [COL_STATE]
    passthrough_predictors = [COL_IS_NEAR_TJ, 'month_sold']

    robust_scaler_pipeline = Pipeline(steps=[
        ("scaler", RobustScaler())
    ])
    one_hot_encoder_pipeline = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("robust", robust_scaler_pipeline, numerical_predictors),
            ("onehot", one_hot_encoder_pipeline, categorical_predictors),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )

    print("Fitting the transformer on the training data...")
    all_predictor_columns = numerical_predictors + categorical_predictors + passthrough_predictors
    missing_cols = [col for col in all_predictor_columns if col not in train_df.columns]
    if missing_cols:
        print(f"Error: Required columns missing for fitting: {missing_cols}")
        return
    transformer.fit(train_df[all_predictor_columns])

    feature_names = transformer.get_feature_names_out()

    # --- 8. Transform Data and Save ---
    print("Transforming training data...")
    missing_cols_train = [col for col in all_predictor_columns if col not in train_df.columns]
    if missing_cols_train:
         print(f"Error: Columns missing from training data before transform: {missing_cols_train}")
         return
    train_transformed_data = transformer.transform(train_df[all_predictor_columns])
    train_df_processed = pd.DataFrame(train_transformed_data, columns=feature_names, index=train_df.index)
    train_df_processed[COL_PRICE] = train_df[COL_PRICE] # Add target back
    
    # --- NEW: Add original numeric columns back with a suffix ---
    print("Adding original, unscaled numeric columns back to training data...")
    original_train_cols = train_df[numerical_predictors].add_suffix('_original')
    train_df_processed = train_df_processed.join(original_train_cols)
    # --- END NEW ---

    print("Transforming test data...")
    missing_cols_test = [col for col in all_predictor_columns if col not in test_df.columns]
    if missing_cols_test:
         print(f"Error: Columns missing from test data before transform: {missing_cols_test}")
         return
    test_transformed_data = transformer.transform(test_df[all_predictor_columns])
    test_df_processed = pd.DataFrame(test_transformed_data, columns=feature_names, index=test_df.index)
    test_df_processed[COL_PRICE] = test_df[COL_PRICE] # Add target back
    
    # --- NEW: Add original numeric columns back with a suffix ---
    print("Adding original, unscaled numeric columns back to test data...")
    original_test_cols = test_df[numerical_predictors].add_suffix('_original')
    test_df_processed = test_df_processed.join(original_test_cols)
    # --- END NEW ---

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)

    train_output_path = os.path.join(OUTPUT_DIR, "training_processed.csv")
    test_output_path = os.path.join(OUTPUT_DIR, "test_processed.csv")

    train_df_processed.to_csv(train_output_path, index=False)
    test_df_processed.to_csv(test_output_path, index=False)

    print("-" * 50)
    print("Preprocessing complete!")
    print(f"Processed training data saved to: {train_output_path}")
    print(f"Processed test data saved to: {test_output_path}")
    print("-" * 50)

if __name__ == "__main__":
    main()