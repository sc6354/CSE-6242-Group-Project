############################################################
# preprocess_data.py
############################################################
# This script preprocesses housing data to prepare it for models
# to predict prices. It implements the following :
# 1. Load your housing data and the Trader Joe's zip code data.
# 2. Create the 'is_near_trader_joes' feature.
# 3. Creats 'is_Sold' feature -  A flag (1/0) indicating if a
#    property has a previous sale date.
# 4. All records are kept, and missing sale dates are handled.
# 5. Clean the price and prev_sold_date columns.
# 6. Imputation for missing prices based on bed/bath count.
# 7. Geospatial Feature Encoding: Converts city and state into
#    latitude and longitude to create powerful, low-dimensional features.
# 8. Imputation for missing prices based on bed/bath count.
# 9. A random 80/20 split for training and testing.
# 10. Save the final, model-ready datasets to new CSV files.
############################################################

import os
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

# Note: geopy is a new requirement. Install with: pip install geopy
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# --- Configuration ---

# Data files
HOUSING_DATA_FILE = "data/raw/realtor-data.zip.csv"
TRADER_JOES_ZIPS_FILE = "data/raw/tj-locations.csv"
ZIP_COORDS_FILE = "data/simplemaps_uszips_basicv1.911/uszips.csv"

# Output directory
OUTPUT_DIR = "data/processed_data"

# Define column names for clarity
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
# New feature column
COL_IS_SOLD = "is_Sold"


def main():
    """Main function to execute the preprocessing workflow."""
    print("Starting data preprocessing...")

    # --- 1. Data Loading ---
    if not os.path.exists(HOUSING_DATA_FILE) or not os.path.exists(ZIP_COORDS_FILE):
        print(f"Error: Make sure data files are in the correct directory.")
        return

    df_housing = pd.read_csv(HOUSING_DATA_FILE)
    df_tj_zips = pd.read_csv(TRADER_JOES_ZIPS_FILE)
    df_zip_coords = pd.read_csv(ZIP_COORDS_FILE)
    print(f"Loaded {len(df_housing)} housing records.")

    # --- 2. Feature Engineering ---

    # MODIFICATION: Create the 'is_Sold' feature first
    df_housing[COL_IS_SOLD] = df_housing[COL_SOLD_DATE].notnull().astype(int)
    print(f"Created 'is_Sold' feature. Found {df_housing[COL_IS_SOLD].sum()} sold properties.")

    # Create the 'is_near_trader_joes' feature
    tj_zips_set = set(df_tj_zips['zip'].unique())
    df_housing[COL_IS_NEAR_TJ] = df_housing[COL_ZIP].isin(tj_zips_set).astype(int)
    
    # Fast Geospatial Encoding via Merge
    print("Performing fast geospatial encoding...")
    df_zip_coords = df_zip_coords[['zip', 'lat', 'lng']]
    df_housing = pd.merge(df_housing, df_zip_coords, left_on=COL_ZIP, right_on='zip', how='left')
    df_housing.rename(columns={'lat': COL_LAT, 'lng': COL_LON}, inplace=True)
    df_housing.drop(columns='zip', inplace=True)
    print("Geospatial encoding complete.")

    # Clean the price column
    if df_housing[COL_PRICE].dtype == 'object':
        df_housing[COL_PRICE] = df_housing[COL_PRICE].str.replace(r'[$,]', '', regex=True).astype(float)

    # Process the date column
    df_housing[COL_SOLD_DATE] = pd.to_datetime(df_housing[COL_SOLD_DATE], errors='coerce')
    df_housing['year_sold'] = df_housing[COL_SOLD_DATE].dt.year
    df_housing['month_sold'] = df_housing[COL_SOLD_DATE].dt.month

    # MODIFICATION: Fill missing date parts with 0 for unsold houses
    df_housing[['year_sold', 'month_sold']] = df_housing[['year_sold', 'month_sold']].fillna(0)


    # --- 3. Data Cleaning, Imputation & Splitting ---
    
    # Impute missing or zero prices
    print("Imputing missing or zero prices...")
    invalid_price_mask = (df_housing[COL_PRICE].isnull()) | (df_housing[COL_PRICE] <= 0)
    
    if invalid_price_mask.sum() > 0:
        df_housing['median_price_impute'] = df_housing.loc[~invalid_price_mask].groupby(
            [COL_BED, COL_BATH]
        )[COL_PRICE].transform('median')
        df_housing[COL_PRICE] = df_housing[COL_PRICE].mask(invalid_price_mask, df_housing['median_price_impute'])
        
        if df_housing[COL_PRICE].isnull().any():
            global_median = df_housing.loc[~invalid_price_mask, COL_PRICE].median()
            df_housing[COL_PRICE].fillna(global_median, inplace=True)
        df_housing.drop(columns=['median_price_impute'], inplace=True)
        print("Imputation complete.")

    # MODIFICATION: Changed dropna subset to keep unsold records
    df_housing.dropna(subset=[COL_HOUSE_SIZE, COL_LAT, COL_LON], inplace=True)
    median_lot_size = df_housing[df_housing[COL_IS_SOLD] == 1][COL_ACRE_LOT].median()
    df_housing[COL_ACRE_LOT].fillna(median_lot_size, inplace=True)
    
    print(f"Data cleaned. {len(df_housing)} records remaining.")

    # MODIFICATION: Using a random 80/20 split instead of time-based
    train_df, test_df = train_test_split(df_housing, test_size=0.2, random_state=42)
    print(f"Split data randomly: {len(train_df)} training, {len(test_df)} test records.")

    # --- 4. Preprocessing Pipeline ---
    # MODIFICATION: Added 'is_Sold' to the passthrough predictors
    numerical_predictors = [COL_BED, COL_BATH, COL_ACRE_LOT, COL_HOUSE_SIZE, COL_LAT, COL_LON]
    passthrough_predictors = [COL_IS_NEAR_TJ, COL_IS_SOLD, 'year_sold', 'month_sold']
    
    robust_scaler_pipeline = Pipeline(steps=[
        ("scaler", RobustScaler())
    ])

    transformer = ColumnTransformer(
        transformers=[
            ("robust", robust_scaler_pipeline, numerical_predictors),
        ],
        remainder="passthrough",
        verbose_feature_names_out=False
    )
    
    print("Fitting the transformer on the training data...")
    all_predictor_columns = numerical_predictors + passthrough_predictors
    transformer.fit(train_df[all_predictor_columns])
    
    feature_names = transformer.get_feature_names_out()

    # --- 5. Transform Data and Save ---
    
    print("Transforming training data...")
    train_transformed_data = transformer.transform(train_df[all_predictor_columns])
    train_df_processed = pd.DataFrame(train_transformed_data, columns=feature_names, index=train_df.index)
    train_df_processed[COL_PRICE] = train_df[COL_PRICE]

    print("Transforming test data...")
    test_transformed_data = transformer.transform(test_df[all_predictor_columns])
    test_df_processed = pd.DataFrame(test_transformed_data, columns=feature_names, index=test_df.index)
    test_df_processed[COL_PRICE] = test_df[COL_PRICE]
    
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

