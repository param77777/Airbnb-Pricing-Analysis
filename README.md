# Airbnb Host & Listing Analysis Project

This repository contains a comprehensive data science project for analyzing Airbnb listings and host behavior. The analysis covers **Host Reliability**, **Listing Price Prediction**, and **Geospatial & Behavioral Clustering**.

---

## 💻 Code Functionality

The Python script (`main.py`) performs the following steps:

### 1. Data Loading and Cleaning

* **Imports:** Loads necessary data science libraries, including `pandas`, `sklearn`, `shap`, `folium`, and `statsmodels`.
* **Data Merge:** Loads `Listings.csv` and `Reviews.csv` and merges them on `listing_id`.
* **Imputation & Cleaning:** Cleans and fills missing values for key columns (host info, review scores, bedrooms).
* **Feature Engineering:** Creates new features like `host_active_days`, `days_since_last_review`, and converts `price` to a float.

### 2. Host Segmentation and Scoring

* **RFM (Recency, Frequency, Monetary):** Calculates R, F, and M scores for each host based on review activity and listing price, combining them into a final **RFM\_Score**.
* **Host Reliability Score:** Calculates a composite score based on normalized host features (e.g., `host_is_superhost`, identity verification, total listings).

### 3. Geospatial Clustering

* **K-Means:** Uses K-Means clustering on scaled `latitude` and `longitude` data to group listings into geographical regions.
* **Optimal K:** Determines the optimal number of clusters using the **KneeLocator** (Elbow method).
* **Output:** Generates an interactive HTML map (`cluster_map.html`) visualizing the geographical clusters, price, and host reliability.

### 4. Predictive Modeling and Explainability

* **Host Trust Prediction:**
    * Trains a **RidgeCV** model to predict the `host_reliability_score`.
    * Uses **SHAP** (SHapley Additive exPlanations) to explain the model's predictions and determine feature importance (`trust_score_shap.png`, `trust_feature_importance.png`).
* **Listing Price Prediction:**
    * Trains a **RidgeCV** model to predict the log-transformed `price`.
    * Performs detailed error analysis (RMSE, MAE, MAPE) and visualizes error by price band (`price_error_by_band.png`).

### 5. Behavioral Host Clustering

* **K-Means on Behavior:** Clusters hosts based on behavioral features (RFM, reliability, and all review scores).
* **Optimal K:** Selects the best number of clusters using the **Silhouette Score**.
* **Interpretation:** Assigns human-readable labels (e.g., "Trusted & Active," "Low Trust") to the clusters based on mean feature values and visualizes the results (`behavior_clusters.png`).

### 6. Time Series Analysis

* **STL Decomposition:** Performs Seasonal-Trend decomposition using Loess (STL) on the monthly review count for the most active host to analyze trend and seasonality (`time_series_decomposition.png`).

---

## ⚙️ How to Run

### Prerequisites
Install the required libraries:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn kneed folium shap statsmodels
