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
# 6. Filters data to only include zip codes with population density > 94.
# 7. Removes extreme price outliers using the IQR method.
# 8. Imputation for missing lot size based on median lot size.
# 9. Imputation for missing house size based on median house size
#    Imputation for missing bed/bath counts based on median bed/bath counts.
# 10. Imputation for missing prices based on bed/bath count.
# 11. Scaling and encoding using RobustScaler and OneHotEncoder and included
#    the original, unscaled numeric columns for reference/visualization later.
# 12. A random 80/20 split for training and testing.
# 13. Save the final, model-ready datasets to new CSV files.
############################################################

import os
import pandas as pd
import numpy as np
from scipy.spatial import cKDTree

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
# Year to filter previous sold date
YEAR_TO_KEEP = 2022
# Distance threshold to consider "near" a Trader Joe's
DISTANCE_THRESHOLD_MILES = 25
# IQR Multiplier for outlier detection
IQR_MULTIPLIER = 1.5
# Zip code population density threshold
DENSITY_THRESHOLD = 5000

def main():
    """Main function to execute the preprocessing workflow."""
    print("Starting data preprocessing...")

    # --- 1. Data Loading ---
    if not os.path.exists(HOUSING_DATA_FILE) or not os.path.exists(ZIP_COORDS_FILE) or not os.path.exists(TRADER_JOES_ZIPS_FILE):
        print(f"Error: Make sure all data files are in the correct directories.")
        return

    df_housing_raw = pd.read_csv(HOUSING_DATA_FILE)
    df_tj_zips_raw = pd.read_csv(TRADER_JOES_ZIPS_FILE)
    # --- MODIFIED: Load density column ---
    df_zip_coords_raw = pd.read_csv(ZIP_COORDS_FILE)
    print(f"Loaded {len(df_housing_raw)} total housing records.")
    print(f"Loaded {len(df_zip_coords_raw)} zip code records.")

    # --- 2. Preprocess Coordinates Data ---
    print(f"Preprocessing zip code data and filtering for density > {DENSITY_THRESHOLD}...")
    df_zip_coords = df_zip_coords_raw.copy()
    df_zip_coords['zip'] = df_zip_coords['zip'].astype(str).str.zfill(5)
    # Keep density column for filtering
    df_zip_coords = df_zip_coords[['zip', 'lat', 'lng', 'density']].copy()
    df_zip_coords.dropna(subset=['lat', 'lng', 'density'], inplace=True)

    # Apply density filter
    initial_zip_count = len(df_zip_coords)
    df_zip_coords = df_zip_coords[df_zip_coords['density'] > DENSITY_THRESHOLD].copy()
    print(f"Filtered zip codes by density. {len(df_zip_coords)} of {initial_zip_count} zip codes remaining.")

    # --- 3. Preprocess Trader Joe's Locations ---
    # Merge with the already density-filtered df_zip_coords
    print("Preprocessing Trader Joe's locations...")
    df_tj_zips_raw['zip'] = df_tj_zips_raw['zip'].astype(str).str.extract(r'(\d{5})', expand=False)
    df_tj_zips_raw.dropna(subset=['zip'], inplace=True)
    df_tj_locations = pd.merge(df_tj_zips_raw, df_zip_coords[['zip', 'lat', 'lng']], on='zip', how='left') # Only need lat/lng now
    df_tj_locations.dropna(subset=['lat', 'lng'], inplace=True)
    # Store the coordinates in a numpy array for KDTree
    tj_coords = df_tj_locations[['lat', 'lng']].values
    print(f"Found coordinates for {len(tj_coords)} Trader Joe's locations in filtered zip codes.")
    if len(tj_coords) == 0:
        print("Warning: Could not find coordinates for any Trader Joe's locations in the high-density zip codes.")

    # Build KDTree for efficient nearest neighbor search
    if len(tj_coords) > 0:
        tree = cKDTree(np.radians(tj_coords)) # Use radians for haversine distance
    else:
        tree = None # Handle case where no TJs remain after density filter
        print("Skipping KDTree creation as no TJ locations remain.")

    # --- 4. Preprocess Housing Data ---
    df_housing = df_housing_raw.copy()
    initial_housing_count = len(df_housing)

    # Filter for selected states first
    df_housing = df_housing[df_housing[COL_STATE].isin(STATES_TO_KEEP)].copy()
    print(f"Filtered for states. {len(df_housing)} records remaining.")

    # Add house coordinates & density by merging with filtered zip data
    df_housing[COL_ZIP] = df_housing[COL_ZIP].astype(str).str.extract(r'(\d{5})', expand=False)
    # Merge with density-filtered zip data ---
    df_housing = pd.merge(df_housing, df_zip_coords[['zip', 'lat', 'lng']], left_on=COL_ZIP, right_on='zip', how='left')
    df_housing.rename(columns={'lat': COL_LAT, 'lng': COL_LON}, inplace=True)

    # Drop rows where merge failed due to density filter (or other reasons) ---
    # This effectively applies the density filter to the housing data
    df_housing.dropna(subset=[COL_LAT, COL_LON], inplace=True)
    print(f"Applied zip code density filter to housing data via merge. {len(df_housing)} records remaining.")
    print("Added house coordinates.")


    # Calculate distance to nearest TJ (Handle empty tree case)
    if tree is not None:
        print("Calculating distance to nearest Trader Joe's for each house...")
        house_coords_rad = np.radians(df_housing[[COL_LAT, COL_LON]].values)
        distances_rad, indices = tree.query(house_coords_rad, k=1)

        nearest_tj_coords = tj_coords[indices]
        df_housing[COL_NEAREST_TJ_LAT] = nearest_tj_coords[:, 0]
        df_housing[COL_NEAREST_TJ_LON] = nearest_tj_coords[:, 1]
        print("Added nearest Trader Joe's coordinates.")

        earth_radius_miles = 3958.8
        df_housing[COL_MIN_DIST_TJ] = distances_rad * earth_radius_miles
        print("Distance calculation complete.")

        df_housing[COL_IS_NEAR_TJ] = (df_housing[COL_MIN_DIST_TJ] <= DISTANCE_THRESHOLD_MILES).astype(int)
        print(f"Identified {df_housing[COL_IS_NEAR_TJ].sum()} properties within {DISTANCE_THRESHOLD_MILES} miles of a Trader Joe's.")
    else:
        # If no TJs remained, set distance/proximity features to default values (e.g., NaN or 0/infinity)
        print("Skipping distance calculation as no TJ locations were found in high-density zip codes.")
        df_housing[COL_NEAREST_TJ_LAT] = 0
        df_housing[COL_NEAREST_TJ_LON] = 0
        df_housing[COL_MIN_DIST_TJ] = 0
        df_housing[COL_IS_NEAR_TJ] = 0

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
        print(f"Error: No data found for properties sold in {YEAR_TO_KEEP} in the selected states/densities.")
        return
    print(f"Filtered for year {YEAR_TO_KEEP}. {len(df_housing)} of {year_filter_count} records remaining.")


    # --- 6. Data Cleaning, Imputation & Splitting ---
    # Remove Price Outliers using IQR
    print("Removing price outliers...")
    Q1 = df_housing[COL_PRICE].quantile(0.25)
    Q3 = df_housing[COL_PRICE].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - IQR_MULTIPLIER * IQR
    upper_bound = Q3 + IQR_MULTIPLIER * IQR
    initial_count = len(df_housing)
    df_housing = df_housing[(df_housing[COL_PRICE] >= lower_bound) & (df_housing[COL_PRICE] <= upper_bound)]
    removed_count = initial_count - len(df_housing)
    print(f"Removed {removed_count} records identified as price outliers based on IQR (Bounds: {lower_bound:,.0f} - {upper_bound:,.0f}).")
    print(f"{len(df_housing)} records remaining after outlier removal.")

    # Impute missing or zero prices
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
        print("Price imputation complete.")

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

    # Check if the column exists and has non-finite values before imputing
    if COL_MIN_DIST_TJ in df_housing.columns and not np.all(np.isfinite(df_housing[COL_MIN_DIST_TJ])):
         print(f"Imputing non-finite values in {COL_MIN_DIST_TJ}...")
         dist_median = df_housing.loc[np.isfinite(df_housing[COL_MIN_DIST_TJ]), COL_MIN_DIST_TJ].median()
         df_housing[COL_MIN_DIST_TJ].replace([np.inf, -np.inf], np.nan, inplace=True)
         df_housing[COL_MIN_DIST_TJ].fillna(dist_median, inplace=True)

    print(f"Data cleaned. {len(df_housing)} records remaining.")

    train_df, test_df = train_test_split(df_housing, test_size=0.2, random_state=42)
    print(f"Split data randomly: {len(train_df)} training, {len(test_df)} test records.")


    # --- 7. Preprocessing Pipeline ---
    numerical_predictors = [
        COL_BED, COL_BATH, COL_ACRE_LOT, COL_HOUSE_SIZE,
        COL_LAT, COL_LON, COL_MIN_DIST_TJ,
        COL_NEAREST_TJ_LAT, COL_NEAREST_TJ_LON
    ]
    categorical_predictors = [COL_STATE]
    passthrough_predictors = [COL_IS_NEAR_TJ, 'month_sold'] # month_sold added here

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
    # Ensure all columns exist before fitting
    missing_cols_fit = [col for col in all_predictor_columns if col not in train_df.columns]
    if missing_cols_fit:
        print(f"Error: Required columns missing for fitting transformer: {missing_cols_fit}")
        # Attempt to impute medians for missing numeric columns as a fallback before failing
        for col in missing_cols_fit:
            if col in numerical_predictors or col in passthrough_predictors: # Check if numeric/passthrough
                 if pd.api.types.is_numeric_dtype(train_df[col]):
                      print(f"Attempting to impute missing predictor column '{col}' with median.")
                      median_val = train_df[col].median()
                      train_df[col].fillna(median_val, inplace=True)
                      if col in test_df.columns: # Also impute test set
                           test_df[col].fillna(median_val, inplace=True)
                 else:
                      print(f"Column '{col}' is not numeric, cannot impute median. Aborting.")
                      return
            else:
                 print(f"Column '{col}' not in expected predictor lists. Aborting.")
                 return
        # Re-check after imputation attempt
        missing_cols_fit = [col for col in all_predictor_columns if col not in train_df.columns]
        if missing_cols_fit:
            print(f"Error: Columns still missing after imputation attempt: {missing_cols_fit}")
            return
        else:
            print("Imputation successful for missing predictor columns.")


    # Check for NaNs *within* the selected columns before fitting
    cols_to_check = train_df[all_predictor_columns]
    if cols_to_check.isnull().any().any():
         print("Error: NaN values detected in columns used for fitting the transformer:")
         print(cols_to_check.isnull().sum()[cols_to_check.isnull().sum() > 0])
         # Add more robust imputation here if necessary, or return
         print("Attempting final median imputation before fitting...")
         for col in cols_to_check.columns[cols_to_check.isnull().any()]:
             if pd.api.types.is_numeric_dtype(train_df[col]):
                 median_val = train_df[col].median()
                 train_df[col].fillna(median_val, inplace=True)
                 if col in test_df.columns:
                     test_df[col].fillna(median_val, inplace=True)
                 print(f"Imputed NaNs in '{col}' with median {median_val}.")
             else:
                 print(f"Column '{col}' has NaNs but is not numeric. Cannot impute median.")
                 
         # Re-check after imputation
         cols_to_check = train_df[all_predictor_columns]
         if cols_to_check.isnull().any().any():
              print("Error: NaN values still present after imputation attempt. Aborting.")
              print(cols_to_check.isnull().sum()[cols_to_check.isnull().sum() > 0])
              return
         else:
              print("Final imputation successful.")


    transformer.fit(train_df[all_predictor_columns])

    feature_names = transformer.get_feature_names_out()


    # --- 8. Transform Data and Save ---
    print("Transforming training data...")
    # Ensure columns exist and no NaNs before transforming
    missing_cols_train = [col for col in all_predictor_columns if col not in train_df.columns]
    if missing_cols_train:
         print(f"Error: Columns missing from training data before transform: {missing_cols_train}")
         return
    if train_df[all_predictor_columns].isnull().any().any():
         print("Error: NaN values detected in training data columns before transform.")
         print(train_df[all_predictor_columns].isnull().sum()[train_df[all_predictor_columns].isnull().sum() > 0])
         return 

    train_transformed_data = transformer.transform(train_df[all_predictor_columns])
    train_df_processed = pd.DataFrame(train_transformed_data, columns=feature_names, index=train_df.index)
    train_df_processed[COL_PRICE] = train_df[COL_PRICE] # Add target back

    print("Adding original, unscaled numeric columns back to training data...")
    original_train_cols = train_df[[col for col in numerical_predictors if col in train_df.columns]].add_suffix('_original') # Check column existence
    train_df_processed = train_df_processed.join(original_train_cols)

    print("Transforming test data...")
    missing_cols_test = [col for col in all_predictor_columns if col not in test_df.columns]
    if missing_cols_test:
         print(f"Error: Columns missing from test data before transform: {missing_cols_test}")
         return
    if test_df[all_predictor_columns].isnull().any().any():
         print("Error: NaN values detected in test data columns before transform.")
         print(test_df[all_predictor_columns].isnull().sum()[test_df[all_predictor_columns].isnull().sum() > 0])
         return

    test_transformed_data = transformer.transform(test_df[all_predictor_columns])
    test_df_processed = pd.DataFrame(test_transformed_data, columns=feature_names, index=test_df.index)
    test_df_processed[COL_PRICE] = test_df[COL_PRICE] # Add target back

    print("Adding original, unscaled numeric columns back to test data...")
    original_test_cols = test_df[[col for col in numerical_predictors if col in test_df.columns]].add_suffix('_original') # Check column existence
    test_df_processed = test_df_processed.join(original_test_cols)

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