# Housing Price Prediction Project

## Table of Contents

<br>

__Notice__: This README is for users running the notebook and Streamlit app locally and makes assumptions that the software can be installed on the hardware.


<br>

## 1. Background

This project aims to predict housing prices in the United States by leveraging a combination of standard real estate features and novel location-based data. Specifically, we investigate the "Trader Joe's effect"—the hypothesis that proximity to high-end grocery stores like Trader Joe's correlates with higher property values.

Our approach involves:
* **Data Collection:** Using real estate listings, Trader Joe's store locations, and zip code demographic/density data.
* **Feature Engineering:** Creating geospatial features such as distance to the nearest Trader Joe's, location density, and interaction terms between property characteristics.
* **Machine Learning Modeling:** Training and tuning various regression models, from linear baselines (Ridge, Lasso) to advanced tree-based ensembles (Random Forest, HistGradientBoosting), to accurately forecast housing prices.


## 2. Data Description

The `data` folder contains all datasets used for this Trader Joe's and Housing Price project. Below is a brief description of each file:

-   `data/raw/realtor-data.zip.csv`: [Contains housing prices](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset), beds, lot size, and zip codes.

-   `data/raw/tj-locations.csv`: Provides zip codes of each [Trader Joe's location](https://www.kaggle.com/datasets/saejinmahlauheinert/trader-joes-locations?phase=FinishSSORegistration&returnUrl=%2Fdatasets%2Fsaejinmahlauheinert%2Ftrader-joes-locations%2Fversions%2F33%3Fresource%3Ddownload&SSORegistrationToken=CfDJ8IaGWDgvvrBFtGGva9hUIY67e60_nY9Mf8ml79rMJZjCOHgInCOcGVQu5L4jNAtPBeWqD5A9muD6e7-EB6UhFvCtBg52rqWLStIu1omSD7Kyq6FwFOKg86J3etQgY_lZx_qst_Kq7LM4KzXTtFWgrNikVJcGISfX1sTTMTZXCIyEbJjQferZ4ptgrJ2sDetQ3f4R3tU88NcrcMcdGwcay2PJ7f0CDrhMQSCBj-30E8If9Z_RO-P-cubuEhGl2aHjsTV9d2pdz2ta_jcNLkZe2q9lLMIiBAtGcQmQuSo8jWEhUJPbN1rrmIqFg67cQ8sdjcOIrP_rJnk7AGYONwRD06eWNw&DisplayName=Sue).

-   `data/simplemaps_uszips_basicv1.911`: Provides longitude and latitude coordinates for US zip codes. Data is sourced from [SimpleMaps](https://simplemaps.com/data/us-zips).



## 3. Environment Setup & Running the Notebook

To reproduce the analysis and run the modeling notebook, follow these steps to set up your environment.


### Optional Steps to Acquire the Datasets:
To ensure reproducebility since Kaggle datasets can be taken down by dataset authors, we included the [raw dataset](https://github.com/sc6354/CSE-6242-Group-Project/raw/main/data/raw/realtor-data.zip.csv). From this raw dataset, 2 preprocessed datasets are created for modeling; one for [training](https://github.com/sc6354/CSE-6242-Group-Project/raw/main/data/processed_data/training_processed.csv) and another for [testing](https://github.com/sc6354/CSE-6242-Group-Project/raw/main/data/processed_data/test_processed.csv).

However, we also provided the commands to extract all datsets below.

1. Run this Kaggle API Command to download the housing dataset.
 ```
#!/bin/bash
kaggle datasets download ahmedshahriarsakib/usa-real-estate-dataset
 ```

2. Run this Kaggle API Command to download the Trader Joes Stores dataset. 
 ```
#!/bin/bash
kaggle datasets download saejinmahlauheinert/trader-joes-locations
 ```

3. Visit [https://simplemaps.com/data/us-zips](https://simplemaps.com/data/us-zips) and click on download to save zip code dataset.

<img src="data/simplemaps_uszips_basicv1.911/zipcode_dataset.png" alt="Click on download to save zip code dataset" width="500">


### Installation Steps
1.  **Clone or download this repository** to your local machine.
2.  **Create the Conda environment** using the provided `notebook_env.yaml` file. open your terminal, navigate to the project root directory, and run:
    ```bash
    conda env create -f conda/notebook_env.yaml
    ```
3.  **Activate the environment**:
    ```bash
    conda activate notebook_env
    ```

### Running the Analysis
1.  **Data Preprocessing:** Before running the notebook, ensure the data is preprocessed. Run the Python script from the project root:
    ```bash
    python src/preprocess_data.py
    ```
    This script will read raw data from `data/raw/`, perform cleaning and feature engineering, and save the processed training and test sets to `data/processed_data/`.

2.  **Launch Jupyter Notebook:**
    ```bash
    jupyter notebook
    ```
3.  **Open and Run the Notebook:** In the Jupyter interface, navigate to the `notebooks/` directory and open `housing_price_modeling.ipynb`. Run all cells to execute the modeling pipeline, evaluate performance, and generate visualizations. Secondly, navigate to `inter_modeling.ipynb` and run all cells to execute inference modeling.

## 4. Environment Setup & Running the Streamlit App

### Environment Setup
1.  **Create the Conda environment** using the provided `viz_env.yaml` file. open your terminal, navigate to the project root directory, and run:
    ```bash
    conda env create -f conda/viz_env.yaml
    ```
2.  **Activate the environment**:
    ```bash
    conda activate viz_env
    ```
### Run Data Preprocessing
1.  **Data Preprocessing:** Generate the required files for the app. Run the Python script from the project root:
    ```bash
    python src/process_map_data.py
    ```
    Note: ```process_map_data.py``` may take up to 10 minutes to run. It loads the full nationwide Census ZCTA shapefile, joins it with ZIP-code attributes, computes nearest ZIPs for every house in the prediction dataset using a BallTree nearest-neighbor search, aggregates predictions to the ZIP level, and then writes large GeoParquet/GeoJSON files for all 5 states. It will produce 4 output files:
    1. `data/processed_data/tx_zcta_with_prices.csv`
    2. `data/processed_data/tx_zcta_with_prices.parquet`
    3. `data/processed_data/states_zcta.geojson`
    4. `data/processed_data/states_zcta.parquet`

### Run Streamlit App
1.  **Run Streamlit App:**
    ```bash
    streamlit run map_app/app.py
    ```
- Please wait a few minutes for the app to load.
- To zoom in or out of a state, click the “+” or “-” icons about the color bar to the right.
- Hover over ZIP codes to view attributes. Note: Only ZIP codes with homes in our prediction dataset will have predicted value attributes.
- Red dots represent Trader Joe’s locations.
- Our recommended approach is to hover over ZIP codes near Trader Joe’s locations and ZIP codes further from Trader Joe’s locations to see how predicted attributes change by ZIP code.
