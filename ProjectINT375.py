
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
file_path = r"C:\Users\Suraj\Downloads\mortality.csv"
df = pd.read_csv(file_path)

# Ensure reproducibility
np.random.seed(42)

# ==========================================
# Objective 1: Data Preprocessing & Cleaning
# ==========================================


def objective_1_preprocessing(data):
    # Create a copy to avoid modifying original data
    data = data.copy()
    
    # Replace 'None', 'NaN', and empty strings with np.nan
    data = data.replace(['None', 'NaN', ''], np.nan)
    
    # Identify numeric columns safely
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Convert possible numeric-like columns to numeric
    for col in data.columns:
        if col not in num_cols:
            data[col] = pd.to_numeric(data[col], errors='ignore')
    
    # Recalculate numeric columns after conversion
    num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    
    # Fill missing values with median
    for col in num_cols:
        median_val = data[col].median()
        data[col] = data[col].fillna(median_val)
    
    print("Objective 1: Data cleaning and imputation complete.")
    
    return data, num_cols


# Call function
df_clean, features = objective_1_preprocessing(df)
#this code for make changes in the actual data (replace all the missing values)
#Replace invalid strings with NaN
   # data = data.replace(['None', 'NaN', ''], np.nan)
    
    # Select numeric columns safely
   # num_cols = data.select_dtypes(include=[np.number]).columns
    
    # Convert columns to numeric
    #data[num_cols] = data[num_cols].apply(pd.to_numeric, errors='coerce')
    
    # Fill missing values with median
    #data[num_cols] = data[num_cols].apply(lambda x: x.fillna(x.median()))
    
   # print("Objective 1: Data cleaning and imputation complete.")
    
    #return data, num_cols


#df_clean, features = objective_1_preprocessing(df.copy())

# ==========================================
# Objective 2: Exploratory Data Analysis (EDA)
# ==========================================
def objective_2_eda(data):
    plt.figure(figsize=(10, 6))
    sns.histplot(data.iloc[:, 5], bins=30, kde=True, color='blue')
    plt.title('Distribution of Infant Mortality Rate')
    plt.xlabel('Infant Mortality Rate (IMR)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.savefig('objective_2_eda.png')
    print("Objective 2: EDA completed. Plot saved.")
    plt.close()
    
objective_2_eda(df_clean)

# ==========================================
# Objective 3: Correlation Analysis
# ==========================================
def objective_3_correlation(data, num_cols):
    plt.figure(figsize=(12, 10))
    corr_matrix = data[num_cols].corr()
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title('Correlation Matrix of Socio-Economic and Health Factors')
    plt.savefig('objective_3_correlation.png')
    print("Objective 3: Correlation matrix computed. Plot saved.")
    plt.close()
    return corr_matrix

corr_matrix = objective_3_correlation(df_clean, features)

# ==========================================
# Objective 4: Maternal Health Impact Analysis
# ==========================================
def objective_4_maternal_impact(data):
    # Analyzing impact of Anemia on Infant Mortality
    imr_col = data.columns[5]
    anemia_col = data.columns[11] # Women Age 15-49 Years Who Are Anemic
    
    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=data[anemia_col], y=data[imr_col], alpha=0.6, color='red')
    sns.regplot(x=data[anemia_col], y=data[imr_col], scatter=False, color='darkred')
    plt.title('Impact of Maternal Anemia on Infant Mortality')
    plt.xlabel('Women Anemic (%)')
    plt.ylabel('Infant Mortality Rate')
    plt.savefig('objective_4_maternal_impact.png')
    print("Objective 4: Maternal health impact analysis completed. Plot saved.")
    plt.close()
    
objective_4_maternal_impact(df_clean)

# ==========================================
# Objective 5: Economic Disparity Analysis
# ==========================================
def objective_5_economic_disparity(data):
    imr_col = data.columns[5]
    income_col = data.columns[15] # Per Capita Income
    
    # Aggregate by District
    district_data = data.groupby('District')[[imr_col, income_col]].mean()
    
    plt.figure(figsize=(12, 6))
    sns.barplot(x=district_data.index, y=district_data[income_col], color='skyblue')
    plt.xticks(rotation=90)
    plt.title('Average Per Capita Income by District')
    plt.ylabel('Income (INR)')
    plt.tight_layout()
    plt.savefig('objective_5_economic_disparity.png')
    print("Objective 5: Economic Disparity bar plot generated. Plot saved.")
    plt.close()
    return district_data

district_stats = objective_5_economic_disparity(df_clean)

# ==========================================
# Objective 6: Vulnerability Clustering
# ==========================================
def objective_6_clustering(data, num_cols):
    # Select features for clustering (excluding identifiers and target IMR if preferred)
    cluster_features = num_cols[1:] # Exclude IMR
    X = data[cluster_features]
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    data['Vulnerability_Cluster'] = kmeans.fit_predict(X_scaled)
    
    imr_col = data.columns[5]
    avg_imr = data.groupby('Vulnerability_Cluster')[imr_col].mean()
    print("Objective 6: Sub-districts clustered into 3 risk tiers. Average IMR per cluster:")
    print(avg_imr)
    return data

df_clustered = objective_6_clustering(df_clean, features)

# ==========================================
# Objective 7: Predictive Modeling for IMR
# ==========================================
def objective_7_predictive_modeling(data, num_cols):
    imr_col = num_cols[0]
    X = data[num_cols[1:]] # Predictors
    y = data[imr_col]      # Target
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    predictions = rf_model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    # Feature Importance
    importance = pd.Series(rf_model.feature_importances_, index=X.columns).sort_values(ascending=False)
    
    plt.figure(figsize=(10, 6))
    importance[:10].plot(kind='bar', color='green')
    plt.title('Top 10 Feature Importances for Predicting IMR')
    plt.ylabel('Importance Score')
    plt.tight_layout()
    plt.savefig('objective_7_predictive_modeling.png')
    
    print(f"Objective 7: Predictive modeling complete. Model MSE: {mse:.2f}, R2 Score: {r2:.2f}. Plot saved.")
    plt.close()
    return rf_model, importance

model, feature_importance = objective_7_predictive_modeling(df_clean, features)
print("All objectives executed successfully.")




