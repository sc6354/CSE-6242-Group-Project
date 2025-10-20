############################################################
# glue_job_script.py
############################################################
# This PySpark script is designed to be run as an AWS Glue job.
# It preprocesses housing data, including geospatial encoding,
# creating an 'is_Sold' flag, and imputing missing prices.
############################################################

import sys
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from pyspark.sql.functions import col, when, year, month, to_date, regexp_replace, concat, lit, avg, first
from pyspark.sql.window import Window
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml import Pipeline

# --- 1. Initialization ---
args = getResolvedOptions(sys.argv, ['S3_SOURCE_BUCKET', 'S3_DESTINATION_BUCKET'])
source_bucket_name = args['S3_SOURCE_BUCKET']
destination_bucket_name = args['S3_DESTINATION_BUCKET']

sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
print("AWS Glue job started...")

# --- 2. Data Loading from S3 ---
housing_data_path = f"s3://{source_bucket_name}/raw_data/realtor-data.zip.csv"
tj_zips_path = f"s3://{source_bucket_name}/raw_data/tj-locations.csv"
coords_path = f"s3://{source_bucket_name}/raw_data/uszips.csv"

df_housing = spark.read.csv(housing_data_path, header=True, inferSchema=True)
df_tj_zips = spark.read.csv(tj_zips_path, header=True, inferSchema=True)
df_coords = spark.read.csv(coords_path, header=True, inferSchema=True)
print(f"Loaded {df_housing.count()} housing records.")

# --- 3. Feature Engineering ---
print("Performing feature engineering...")
# Create 'is_Sold' feature
df_housing = df_housing.withColumn("is_Sold", when(col("prev_sold_date").isNotNull(), 1).otherwise(0))

# Create 'is_near_trader_joes' feature
tj_zips_list = [row['zip'] for row in df_tj_zips.select('zip').distinct().collect()]
df_housing = df_housing.withColumn("is_near_trader_joes", when(col("zip_code").isin(tj_zips_list), 1).otherwise(0))

# Geospatial Encoding via Join
df_coords = df_coords.select(col("zip").alias("zip_code_join"), col("lat").alias("latitude"), col("lng").alias("longitude"))
df_housing = df_housing.join(df_coords, df_housing.zip_code == df_coords.zip_code_join, how="left").drop("zip_code_join")

# Clean price and date columns
df_housing = df_housing.withColumn("price_clean", regexp_replace(col("price"), "[$,]", "").cast("float"))
df_housing = df_housing.withColumn("sold_date_clean", to_date(col("prev_sold_date")))
df_housing = df_housing.withColumn("year_sold", year(col("sold_date_clean")))
df_housing = df_housing.withColumn("month_sold", month(col("sold_date_clean")))
df_housing = df_housing.fillna(0, subset=['year_sold', 'month_sold'])


# --- 4. Data Cleaning & Imputation ---
print("Cleaning data and imputing prices...")

# MODIFICATION: Impute missing or zero prices
invalid_price_mask = (col("price_clean").isNull()) | (col("price_clean") <= 0)

# Define a window partitioned by bed and bath to calculate median
window_spec = Window.partitionBy("bed", "bath")
# Calculate the median for each group
median_df = df_housing.filter(~invalid_price_mask).groupBy("bed", "bath").agg(first("price_clean").alias("median_price"))

# Join the medians back to the main dataframe
df_housing = df_housing.join(median_df, on=["bed", "bath"], how="left")

# Impute the prices
df_housing = df_housing.withColumn("price_imputed", when(invalid_price_mask, col("median_price")).otherwise(col("price_clean")))

# Fallback: fill any remaining nulls with the global median
global_median = df_housing.filter(~invalid_price_mask).selectExpr("percentile_approx(price_clean, 0.5)").first()[0]
df_housing = df_housing.fillna({"price_imputed": global_median})
print("Imputation complete.")

# Clean and drop rows with other critical missing info
df_housing = df_housing.dropna(subset=["house_size", "latitude", "longitude"])
median_acre_lot = df_housing.filter(col("is_Sold") == 1).selectExpr("percentile_approx(acre_lot, 0.5)").first()[0]
df_housing = df_housing.fillna({"acre_lot": median_acre_lot})

# --- 5. Data Splitting ---
# Random 80/20 split
train_df, test_df = df_housing.randomSplit([0.8, 0.2], seed=42)
print(f"Split data randomly: {train_df.count()} training, {test_df.count()} testing records.")


# --- 6. PySpark ML Preprocessing Pipeline ---
print("Building and fitting the PySpark ML pipeline...")
numerical_predictors = ["bed", "bath", "acre_lot", "house_size", "latitude", "longitude"]
passthrough_predictors = ["is_near_trader_joes", 'is_Sold', 'year_sold', 'month_sold']

stages = []
numeric_assembler = VectorAssembler(inputCols=numerical_predictors, outputCol="numeric_features")
scaler = StandardScaler(inputCol="numeric_features", outputCol="scaled_numeric_features")
stages += [numeric_assembler, scaler]

assembler_inputs = ["scaled_numeric_features"] + passthrough_predictors
final_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")
stages += [final_assembler]

pipeline = Pipeline(stages=stages)
pipeline_model = pipeline.fit(train_df)

# --- 7. Transform Data and Save to S3 ---
print("Transforming data and writing to destination S3 bucket...")
train_df_processed = pipeline_model.transform(train_df).select("features", col("price_imputed").alias("price"))
test_df_processed = pipeline_model.transform(test_df).select("features", col("price_imputed").alias("price"))

train_output_path = f"s3://{destination_bucket_name}/processed_data/training_imputed_csv"
test_output_path = f"s3://{destination_bucket_name}/processed_data/test_imputed_csv"

train_df_processed.coalesce(1).write.mode("overwrite").csv(train_output_path, header=True)
test_df_processed.coalesce(1).write.mode("overwrite").csv(test_output_path, header=True)

print(f"Successfully wrote processed training data to {train_output_path}")
print(f"Successfully wrote processed test data to {test_output_path}")
print("AWS Glue job finished.")

