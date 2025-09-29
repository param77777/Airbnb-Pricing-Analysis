import shap
import folium
import numpy as np
import pandas as pd
import seaborn as sns
from kneed import KneeLocator
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from statsmodels.tsa.seasonal import STL
from folium.plugins import MarkerCluster
from sklearn.linear_model import RidgeCV
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

# ---------------------------- Load & Clean Data ----------------------------
df1 = pd.read_csv("Listings.csv", encoding="ISO-8859-1", low_memory=False)
df2 = pd.read_csv("Reviews.csv", encoding="ISO-8859-1", low_memory=False)
df = pd.merge(df1, df2, on="listing_id", how="left")

df.drop(columns=['host_response_time', 'host_response_rate', 'host_acceptance_rate', 'district'], inplace=True)
df['host_location'] = df['host_location'].fillna("Unknown")
df['host_is_superhost'] = df['host_is_superhost'].fillna("f")
df['host_has_profile_pic'] = df['host_has_profile_pic'].fillna("f")
df['host_identity_verified'] = df['host_identity_verified'].fillna("f")

top_props = df['property_type'].value_counts().nlargest(10).index
df['property_type'] = df['property_type'].apply(lambda x: x if x in top_props else 'Other')
df['property_type'] = df['property_type'].fillna(df['property_type'].mode()[0])
df['host_total_listings_count'] = df['host_total_listings_count'].fillna(df['host_total_listings_count'].median())
df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())

review_cols = [
    'review_scores_rating', 'review_scores_accuracy', 'review_scores_cleanliness',
    'review_scores_checkin', 'review_scores_communication', 'review_scores_location', 'review_scores_value'
]
for col in review_cols:
    df[col] = df[col].fillna(df[col].median())

df['host_since'] = pd.to_datetime(df['host_since'], errors='coerce').fillna(pd.to_datetime('2015-01-01'))
df['date'] = pd.to_datetime(df['date'], errors='coerce')
df['date'] = df['date'].fillna(df['date'].max())
df.dropna(subset=['review_id', 'reviewer_id', 'name'], inplace=True)

latest_date = df['date'].max()
df['host_active_days'] = (latest_date - df['host_since']).dt.days
df['days_since_last_review'] = (latest_date - df['date']).dt.days
df['price'] = df['price'].replace(r'[\$,]', '', regex=True).astype(float)
df['price_bin'] = pd.qcut(df['price'], q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
df['TotalSales'] = df['price'] * df['minimum_nights'] * df['accommodates']

# ---------------------------- RFM Features ----------------------------
df_rfm = df.groupby('host_id').agg({
    'date': lambda x: (latest_date - x.max()).days,
    'review_id': 'count',
    'TotalSales': 'sum'
}).reset_index()
df_rfm.columns = ['host_id', 'Recency', 'Frequency', 'Monetary']
df_rfm['R_Score'] = pd.qcut(df_rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
df_rfm['F_Score'] = pd.qcut(df_rfm['Frequency'].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])
df_rfm['M_Score'] = pd.qcut(df_rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
df_rfm['RFM_Score'] = df_rfm[['R_Score', 'F_Score', 'M_Score']].astype(int).sum(axis=1)
df = df.merge(df_rfm[['host_id', 'RFM_Score']], on='host_id', how='left')
df['RFM_Score'] = df['RFM_Score'].fillna(df['RFM_Score'].median()).astype('int8')

# ---------------------------- Host Reliability ----------------------------
df['host_is_superhost'] = df['host_is_superhost'].map({'t': 1, 'f': 0}).astype('int8')
df['host_identity_verified'] = df['host_identity_verified'].map({'t': 1, 'f': 0}).astype('int8')
df['host_has_profile_pic'] = df['host_has_profile_pic'].map({'t': 1, 'f': 0}).astype('int8')
df['host_total_listings_log'] = np.log1p(df['host_total_listings_count'])

scaler = MinMaxScaler()
df[['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic', 'host_total_listings_log']] = scaler.fit_transform(
    df[['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic', 'host_total_listings_log']]
)

df['host_reliability_score'] = (
    0.3 * df['host_is_superhost'] +
    0.25 * df['host_identity_verified'] +
    0.25 * df['host_has_profile_pic'] +
    0.2 * df['host_total_listings_log']
)

# ---------------------------- Geospatial Clustering ----------------------------
df = df[(df['latitude'].between(-90, 90)) & (df['longitude'].between(-180, 180))]
df = df[~((df['latitude'].abs() < 1) & (df['longitude'].abs() < 1))]

coords_sample = df[['latitude', 'longitude']].sample(n=100000, random_state=42)
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords_sample)

wcss = []
for k in range(1, 15):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(coords_scaled)
    wcss.append(kmeans.inertia_)

knee = KneeLocator(range(1, 15), wcss, curve="convex", direction="decreasing")
optimal_k = knee.knee or 3

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
kmeans.fit(coords_scaled)

coords_sample['location_cluster'] = kmeans.predict(coords_scaled)
coords_sample['price'] = df.loc[coords_sample.index, 'price'].values
coords_sample['host_reliability_score'] = df.loc[coords_sample.index, 'host_reliability_score'].values

centroids_scaled = scaler.inverse_transform(kmeans.cluster_centers_)
centroids_df = pd.DataFrame(centroids_scaled, columns=['latitude', 'longitude'])
centroids_df['cluster'] = range(optimal_k)

coords_full_scaled = scaler.transform(df[['latitude', 'longitude']])
df['location_cluster'] = kmeans.predict(coords_full_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(data=coords_sample, x='longitude', y='latitude', hue='location_cluster', palette='tab10', alpha=0.7)
plt.title("Geospatial Clustering of Listings (100k Sample)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

map_center = [coords_sample['latitude'].mean(), coords_sample['longitude'].mean()]
m = folium.Map(location=map_center, zoom_start=10)
marker_cluster = MarkerCluster().add_to(m)
colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred', 'cadetblue', 'darkgreen', 'black', 'gray']

for idx, row in coords_sample.iterrows():
    if pd.notnull(row['latitude']) and pd.notnull(row['longitude']) and pd.notnull(row['price']) and pd.notnull(row['host_reliability_score']):
        popup_text = (
            f"Cluster: {int(row['location_cluster'])}<br>"
            f"Price: ${row['price']:.2f}<br>"
            f"Reliability Score: {row['host_reliability_score']:.2f}"
        )
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=2,
            color=colors[int(row['location_cluster']) % len(colors)],
            fill=True,
            fill_opacity=0.6,
            popup=popup_text
        ).add_to(marker_cluster)

cluster_counts = coords_sample['location_cluster'].value_counts().sort_index()

summary_html = f"""
<div id="summary-box" style="
    position: absolute;
    top: 10px;
    right: 10px;
    z-index: 9999;
    font-family: Arial, sans-serif;
    font-size: 13px;
    background: white;
    padding: 12px 16px;
    border: 1px solid #ccc;
    border-radius: 8px;
    box-shadow: 2px 2px 6px rgba(0,0,0,0.2);
    max-width: 320px;
">
    <div style="font-size: 14px; font-weight: bold; margin-bottom: 6px;">
        Geospatial Cluster Summary
    </div>
    <div style="margin-bottom: 8px;">
        <b>Total Listings:</b> 100,000<br>
        <b>Total Clusters:</b> {len(cluster_counts)}
    </div>
    <div style="display: flex; flex-wrap: wrap; column-gap: 12px; row-gap: 4px;">
        {"".join([
            f"<div style='width: 48%; color:{colors[i % len(colors)]};'>Cluster {i}: {count}</div>"
            for i, count in cluster_counts.items()
        ])}
    </div>
</div>
"""
m.save("cluster_map.html")
with open("cluster_map.html", "r", encoding="utf-8") as f:
    html = f.read()
html = html.replace("</body>", summary_html + "</body>")
with open("cluster_map.html", "w", encoding="utf-8") as f:
    f.write(html)

def predict_host_trust(df):
    """Predict host trust score using Ridge regression with SHAP explainability"""
    trust_features = [
        'review_scores_rating', 'review_scores_cleanliness', 'review_scores_checkin',
        'review_scores_communication', 'review_scores_location', 'review_scores_value',
        'RFM_Score', 'price', 'accommodates', 'bedrooms'
    ]
    
    df_trust = df[trust_features + ['host_reliability_score']].dropna()
    df_trust = df_trust.replace([np.inf, -np.inf], np.nan).dropna()
    
    X = df_trust[trust_features]
    y = df_trust['host_reliability_score']
    
    model = RidgeCV(alphas=np.logspace(-3, 3, 10), cv=5)
    model.fit(X, y)
    
    predict_features = df[trust_features].fillna(0).replace([np.inf, -np.inf], 0)
    df['predicted_trust'] = model.predict(predict_features)
    
    print("\nCalculating SHAP values for trust score explainability...")
    explainer = shap.KernelExplainer(model.predict, X.iloc[:100])
    shap_values = explainer.shap_values(X.iloc[:100])
    
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X.iloc[:100], show=False)
    plt.title("Feature Importance for Host Trust Score")
    plt.tight_layout()
    plt.savefig("trust_score_shap.png")
    plt.close()
    
    global trust_feature_importance
    trust_feature_importance = dict(zip(trust_features, np.abs(shap_values).mean(0)))
    
    return df, model

def predict_price(df):
    """Predict listing prices using Ridge regression with detailed error analysis"""
    if 'predicted_trust' not in df.columns:
        df, _ = predict_host_trust(df)
    
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    if 'price' in numeric_cols:
        numeric_cols.remove('price')
    
    missing_pct = df[numeric_cols].isna().sum() / len(df)
    numeric_cols = [col for col in numeric_cols if missing_pct[col] < 0.5]
    
    global price_prediction_features
    price_prediction_features = numeric_cols
    
    X = df[numeric_cols].fillna(df[numeric_cols].mean())
    y = np.log1p(df['price'].fillna(df['price'].mean()))
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=0.2, 
            random_state=42, 
            stratify=pd.qcut(y, q=5, duplicates='drop')
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, 
            test_size=0.2, 
            random_state=42
        )
    
    model = RidgeCV(alphas=[0.1, 1.0, 10.0, 100.0, 1000.0], cv=3)
    model.fit(X_train, y_train)
    
    y_pred_log = model.predict(X_test)
    y_pred = np.expm1(y_pred_log)
    y_test = np.expm1(y_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    
    error_df = pd.DataFrame({'y_true': y_test, 'y_pred': y_pred})
    error_df = error_df[error_df['y_true'] > 1e-6]
    mape = mean_absolute_percentage_error(error_df['y_true'], error_df['y_pred'])
    
    print("\nPrice Prediction Error Analysis:")
    print(f"RMSE: ${rmse:.2f}")
    print(f"MAE: ${mae:.2f}")
    print(f"MAPE: {mape:.2%}")
    
    price_bands = pd.qcut(y_test, q=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])
    error_by_band = pd.DataFrame({
        'Actual': y_test,
        'Predicted': y_pred,
        'Error': np.abs(y_test - y_pred),
        'Price_Band': price_bands
    })
    
    print("\nError Analysis by Price Band:")
    print(error_by_band.groupby('Price_Band').agg({
        'Error': ['mean', 'std', 'count']
    }).round(2))
    
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Price_Band', y='Error', data=error_by_band)
    plt.title("Prediction Error Distribution by Price Band")
    plt.xlabel("Price Band")
    plt.ylabel("Absolute Error ($)")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig("price_error_by_band.png")
    plt.close()

    return model, rmse
price_prediction_features = None

def behavior_cluster(df):
    """Cluster hosts based on behavior patterns with optimal K selection"""
    print("Preparing data for clustering...")
    
    sample_size = min(100000, len(df))
    df_sample = df.sample(n=sample_size, random_state=42)
    print(f"Using a sample of {sample_size} records for clustering")
    
    behavior_features = [
        'host_reliability_score', 'RFM_Score',
        'review_scores_rating', 'review_scores_cleanliness', 'review_scores_checkin',
        'review_scores_communication', 'review_scores_location', 'review_scores_value'
    ]
    
    if 'RFM_Score' not in df_sample.columns:
        df_sample['RFM_Score'] = 0
        if 'review_scores_rating' in df_sample.columns:
            df_sample['RFM_Score'] += df_sample['review_scores_rating'].fillna(0)
        if 'days_since_last_review' in df_sample.columns:
            max_days = df_sample['days_since_last_review'].max()
            if max_days > 0:
                df_sample['RFM_Score'] += (1 - df_sample['days_since_last_review'] / max_days) * 10
    
    df_behavior = df_sample[behavior_features].copy()

    for col in behavior_features:
        if col in df_behavior.columns:
            df_behavior[col] = df_behavior[col].fillna(df_behavior[col].mean())
    
    print("Scaling data...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(df_behavior)
    
    print("Finding optimal number of clusters...")
    silhouette_scores = []
    k_values = [2, 3, 4, 5] 
    
    for k in k_values:
        print(f"Trying k={k}...")
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=5)
        labels = kmeans.fit_predict(X_scaled)
        try:
            score = silhouette_score(X_scaled, labels)
            silhouette_scores.append(score)
            print(f"Silhouette score for k={k}: {score:.4f}")
        except:
            silhouette_scores.append(0)
    
    optimal_k = k_values[np.argmax(silhouette_scores)]
    print(f"Optimal number of clusters: {optimal_k}")
    
    print("Performing final clustering...")
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=5)
    df_sample['behavior_cluster'] = kmeans.fit_predict(X_scaled)

    print("Mapping clusters back to full dataset...")
    cluster_map = df_sample[['host_id', 'behavior_cluster']].drop_duplicates()
    
    df = df.merge(cluster_map, on='host_id', how='left')

    df['behavior_cluster'] = df['behavior_cluster'].fillna(-1).astype(int)
    
    print(f"Clustering complete. Found {optimal_k} main clusters plus noise cluster (-1).")
    return df, kmeans

def interpret_behavior_clusters(df):
    """Add human-readable labels to behavior clusters"""
    if 'behavior_cluster' not in df.columns:
        df, _ = behavior_cluster(df)
    
    if 'host_reliability_score' not in df.columns:
        df['host_reliability_score'] = 0.5 
    if 'RFM_Score' not in df.columns:
        df['RFM_Score'] = 0 
    
    means = df.groupby('behavior_cluster')[
        ['host_reliability_score', 'RFM_Score']
    ].mean()
    
    labels = []
    for _, row in means.iterrows():
        if row['host_reliability_score'] > 0.7 and row['RFM_Score'] > 12:
            labels.append("🟢 Trusted & Active")
        elif row['host_reliability_score'] < 0.4:
            labels.append("🔴 Low Trust")
        else:
            labels.append("🟡 Mid-Tier")
    
    means['label'] = labels
    return means

if __name__ == "__main__":
    try:
        print("Loading data...")
        df1 = pd.read_csv(r"C:\Users\tyt36\OneDrive\Desktop\projects\DS\Listings.csv", encoding="ISO-8859-1", low_memory=False)
        print(f"Listings data loaded. Shape: {df1.shape}")
        df2 = pd.read_csv(r"C:\Users\tyt36\OneDrive\Desktop\projects\DS\Reviews.csv", encoding="ISO-8859-1", low_memory=False)
        print(f"Reviews data loaded. Shape: {df2.shape}")
        
        df = pd.merge(df1, df2, on="listing_id", how="left")
        print(f"Data merged. Final shape: {df.shape}")

        print("\nCalculating RFM Score...")
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'], errors='coerce')

        latest_date = df['date'].max()
        df['days_since_last_review'] = (latest_date - df['date']).dt.days
        
        rfm = df.groupby('host_id').agg({
            'days_since_last_review': 'min', 
            'review_id': 'count',             
            'price': 'mean'                   
        }).reset_index()
        
        rfm.columns = ['host_id', 'recency', 'frequency', 'monetary']

        rfm['recency'] = rfm['recency'].fillna(rfm['recency'].max())
        rfm['frequency'] = rfm['frequency'].fillna(0)
        rfm['monetary'] = rfm['monetary'].fillna(0)
        
        try:
            rfm['R_Score'] = pd.qcut(rfm['recency'], q=5, labels=[5, 4, 3, 2, 1])
        except ValueError:
            rfm['R_Score'] = 1 
            rfm.loc[rfm['recency'] < rfm['recency'].median(), 'R_Score'] = 3
            rfm.loc[rfm['recency'] < rfm['recency'].quantile(0.25), 'R_Score'] = 5
        
        try:
            rfm['F_Score'] = pd.qcut(rfm['frequency'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        except ValueError:
            rfm['F_Score'] = 1 
            rfm.loc[rfm['frequency'] > rfm['frequency'].median(), 'F_Score'] = 3
            rfm.loc[rfm['frequency'] > rfm['frequency'].quantile(0.75), 'F_Score'] = 5
        
        try:
            rfm['M_Score'] = pd.qcut(rfm['monetary'].rank(method='first'), q=5, labels=[1, 2, 3, 4, 5])
        except ValueError:
            rfm['M_Score'] = 1 
            rfm.loc[rfm['monetary'] > rfm['monetary'].median(), 'M_Score'] = 3
            rfm.loc[rfm['monetary'] > rfm['monetary'].quantile(0.75), 'M_Score'] = 5
        
        rfm['RFM_Score'] = rfm['R_Score'].fillna(1).astype(int) + rfm['F_Score'].fillna(1).astype(int) + rfm['M_Score'].fillna(1).astype(int)
        
        df = df.merge(rfm[['host_id', 'RFM_Score']], on='host_id', how='left')
        print(f"RFM Score calculated for {df['RFM_Score'].notna().sum()} hosts")
        
        print("\nCalculating host reliability score...")
        bool_cols = ['host_is_superhost', 'host_identity_verified', 'host_has_profile_pic']
        for col in bool_cols:
            if col in df.columns:
                df[col] = df[col].map({'t': 1, 'f': 0}).fillna(0)
        
        df['host_reliability_score'] = 0.0
        
        if 'host_is_superhost' in df.columns:
            df['host_reliability_score'] += df['host_is_superhost'] * 0.4
        if 'host_identity_verified' in df.columns:
            df['host_reliability_score'] += df['host_identity_verified'] * 0.3
        if 'host_has_profile_pic' in df.columns:
            df['host_reliability_score'] += df['host_has_profile_pic'] * 0.3
        
        review_cols = ['review_scores_rating', 'review_scores_cleanliness', 'review_scores_checkin',
                      'review_scores_communication', 'review_scores_location', 'review_scores_value']
        
        for col in review_cols:
            if col in df.columns:
                df['host_reliability_score'] += df[col].fillna(0) / 100 * 0.1
        
        df['host_reliability_score'] = df['host_reliability_score'].clip(0, 1)
        print(f"Host reliability score calculated for {df['host_reliability_score'].notna().sum()} hosts")
        
        print("\nPredicting host trust...")
        df, trust_model = predict_host_trust(df)
        print("Host trust prediction completed.")
        print(f"Number of hosts with trust scores: {df['host_reliability_score'].notna().sum()}")
        
        print("\nPredicting prices...")
        price_model, price_rmse = predict_price(df)
        print(f"Price prediction completed. RMSE: {price_rmse:.2f}")
        
        print("\nClustering hosts by behavior...")
        df, cluster_model = behavior_cluster(df)
        print(f"Behavior clustering completed. Number of clusters: {len(df['behavior_cluster'].unique())}")
        
        print("\nInterpreting behavior clusters...")
        cluster_labels = interpret_behavior_clusters(df)
        print("\nBehavior cluster interpretations:")
        print(cluster_labels)
        
        # ===== VISUALIZATION SECTION =====
        print("\nGenerating visualizations...")
        
        plt.figure(figsize=(10, 6))
        sns.histplot(df['predicted_trust'], kde=True, bins=50)
        plt.title("Distribution of Predicted Host Trust")
        plt.xlabel("Trust Score")
        plt.ylabel("Count")
        plt.grid(True, alpha=0.3)
        plt.savefig("host_trust_distribution.png")
        plt.close()
        
        low_trust_hosts = df[df['predicted_trust'] < 0.4]
        print(f"🚨 Low-trust hosts detected: {len(low_trust_hosts)} ({len(low_trust_hosts)/len(df)*100:.2f}%)")
        
        df_price_sample = df.sample(n=min(10000, len(df)), random_state=42)
        
        if price_prediction_features is None:
            print("Warning: Price prediction features not set. Skipping price prediction visualization.")
        else:
            X_price = df_price_sample[price_prediction_features].fillna(df_price_sample[price_prediction_features].mean())
            scaler = StandardScaler()
            X_price_scaled = scaler.fit_transform(X_price)
            y_price_pred = price_model.predict(X_price_scaled)

            plt.figure(figsize=(10, 6))
            plt.scatter(df_price_sample['price'], y_price_pred, alpha=0.3)
            plt.plot([0, df_price_sample['price'].max()], [0, df_price_sample['price'].max()], 'r--')
            plt.xlabel("Actual Price")
            plt.ylabel("Predicted Price")
            plt.title("Actual vs Predicted Listing Prices")
            plt.grid(True, alpha=0.3)
            plt.savefig("price_prediction.png")
            plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x='behavior_cluster', y='host_reliability_score', data=df.sample(n=min(100000, len(df)), random_state=42))
        plt.title("Host Reliability by Behavior Cluster")
        plt.xlabel("Behavior Cluster")
        plt.ylabel("Host Reliability Score")
        plt.grid(True, alpha=0.3)
        plt.savefig("behavior_clusters.png")
        plt.close()
        
        cluster_counts = df['behavior_cluster'].value_counts().sort_index()
        print("\nHost distribution across behavior clusters:")
        for cluster, count in cluster_counts.items():
            print(f"Cluster {cluster}: {count} hosts ({count/len(df)*100:.2f}%)")
        
        if 'date' in df.columns:
            try:
                print("\nGenerating time-series analysis...")
                
                df['year_month'] = df['date'].dt.to_period('M')
                monthly_reviews = df.groupby(['host_id', 'year_month'])['review_id'].count().reset_index()
                
                top_host_id = df['host_id'].value_counts().index[0]
                print(f"Analyzing time-series for top host: {top_host_id}")
                
                ts = monthly_reviews[monthly_reviews['host_id'] == top_host_id]
                
                if len(ts) > 12:
                    ts = ts.set_index('year_month')['review_id']
                    
                    date_range = pd.period_range(start=ts.index.min(), end=ts.index.max(), freq='M')
                    ts = ts.reindex(date_range)
                    
                    ts = ts.fillna(method='ffill').fillna(method='bfill')
                    
                    fig = plt.figure(figsize=(15, 12))
                    gs = fig.add_gridspec(3, 2)
                    
                    ax1 = fig.add_subplot(gs[0, :])
                    ts.plot(ax=ax1, marker='o', linestyle='-', markersize=4)
                    ax1.set_title('Monthly Review Activity')
                    ax1.set_xlabel('Date')
                    ax1.set_ylabel('Number of Reviews')
                    ax1.grid(True, alpha=0.3)
                    
                    try:
                        stl = STL(ts, 
                                 seasonal=13,  
                                 robust=True,  
                                 period=12)  
                        result = stl.fit()
                        
                        ax2 = fig.add_subplot(gs[1, 0])
                        result.trend.plot(ax=ax2, color='blue')
                        ax2.set_title('Trend Component')
                        ax2.grid(True, alpha=0.3)
                        
                        ax3 = fig.add_subplot(gs[1, 1])
                        result.seasonal.plot(ax=ax3, color='green')
                        ax3.set_title('Seasonal Component')
                        ax3.grid(True, alpha=0.3)
                        
                        ax4 = fig.add_subplot(gs[2, 0])
                        result.resid.plot(ax=ax4, color='red')
                        ax4.set_title('Residual Component')
                        ax4.grid(True, alpha=0.3)
                        
                        ax5 = fig.add_subplot(gs[2, 1])
                        seasonal_data = pd.DataFrame({
                            'Month': ts.index.month,
                            'Reviews': ts.values
                        })
                        sns.boxplot(data=seasonal_data, x='Month', y='Reviews', ax=ax5)
                        ax5.set_title('Seasonal Distribution by Month')
                        ax5.set_xlabel('Month')
                        ax5.set_ylabel('Number of Reviews')
                        ax5.grid(True, alpha=0.3)
                        
                        plt.tight_layout()
                        plt.suptitle(f"Time Series Analysis for Host {top_host_id}", y=1.02, fontsize=14)
                        plt.savefig("time_series_decomposition.png", dpi=300, bbox_inches='tight')
                        plt.close()
                        
                        seasonal_means = ts.groupby(ts.index.month).mean()
                        peak_month = seasonal_means.idxmax()
                        low_month = seasonal_means.idxmin()
                        print(f"\nSeasonal Insights:")
                        print(f"Peak activity in month {peak_month} with {seasonal_means[peak_month]:.1f} reviews")
                        print(f"Lowest activity in month {low_month} with {seasonal_means[low_month]:.1f} reviews")
                        print(f"Seasonal strength: {np.abs(result.seasonal).mean():.2f}")
                        
                    except Exception as e:
                        print(f"Error in STL decomposition: {str(e)}")
                        plt.figure(figsize=(12, 6))
                        ts.plot(marker='o', linestyle='-', markersize=4)
                        plt.title(f"Monthly Review Activity for Host {top_host_id}")
                        plt.xlabel("Date")
                        plt.ylabel("Number of Reviews")
                        plt.grid(True, alpha=0.3)
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        plt.savefig("time_series_simple.png", dpi=300, bbox_inches='tight')
                        plt.close()
                        print("Simple time series plot created instead.")
                else:
                    print(f"Not enough data points for host {top_host_id} to perform time series analysis.")
            except Exception as e:
                print(f"Error in time-series analysis: {str(e)}")
                print("Skipping time-series visualization.")
        
        print("\nGenerating model explainability visualizations...")

        if 'trust_feature_importance' in globals():
            plt.figure(figsize=(10, 6))
            features = list(trust_feature_importance.keys())
            importance = list(trust_feature_importance.values())
            plt.barh(range(len(features)), importance)
            plt.yticks(range(len(features)), features)
            plt.title("Feature Importance for Host Trust Score")
            plt.xlabel("Absolute SHAP Value")
            plt.tight_layout()
            plt.savefig("trust_feature_importance.png")
            plt.close()
        
        print("\nAnalysis complete! Visualizations saved to disk.")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        import traceback
        traceback.print_exc()