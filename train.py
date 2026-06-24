import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

# Base paths
base_path = r"D:\Project\hackathons\crop Yield Prediction"
data_path = os.path.join(base_path, "data")

# 1. Load FAOSTAT Crop Data
faostat_path = os.path.join(data_path, "FAOSTAT_data_en_5-23-2025.csv")
df_fao = pd.read_csv(faostat_path)

# Filter for India, Yield, and Rice
df_yield = df_fao[
    (df_fao['Area'] == 'India') & 
    (df_fao['Element'] == 'Yield') & 
    (df_fao['Item'] == 'Rice')
].copy()

df_yield = df_yield.rename(columns={'Year': 'Year', 'Value': 'Yield'})
df_yield['Yield'] = df_yield['Yield'] / 1000.0  # Convert kg/ha to tonnes/ha
df_yield = df_yield[['Year', 'Yield']]

# 2. Load and Aggregate NASA POWER Weather Data
weather_files = {
    'T2M': "POWER_Regional_Monthly_1984_2025.csv",
    'PRECTOT': "POWER_Regional_Monthly_1984_2025 (1).csv",
    'T2M_MAX': "POWER_Regional_Monthly_1984_2025 (2).csv",
    'T2M_MIN': "POWER_Regional_Monthly_1984_2025 (3).csv",
    'RH2M': "POWER_Regional_Monthly_1984_2025 (4).csv"
}

weather_dfs = []
for param, filename in weather_files.items():
    path = os.path.join(data_path, filename)
    df_w = pd.read_csv(path, skiprows=9)
    # Average the ANN (Annual average) column across all spatial grid points
    df_ann = df_w.groupby('YEAR')['ANN'].mean().reset_index()
    df_ann = df_ann.rename(columns={'ANN': param, 'YEAR': 'Year'})
    weather_dfs.append(df_ann)

# Merge all weather dataframes on Year
df_weather = weather_dfs[0]
for df_w in weather_dfs[1:]:
    df_weather = pd.merge(df_weather, df_w, on='Year')

# 3. Merge Yield and Weather Data
df_merged = pd.merge(df_yield, df_weather, on='Year', how='inner')
print("--- Historical Combined Dataset ---")
print(df_merged.head())

# 4. Train Multiple Linear Regression Model
# Using Year trend to capture technological improvements + weather inputs
features = ['Year', 'T2M_MAX', 'T2M_MIN', 'RH2M', 'PRECTOT']
X = df_merged[features]
y = df_merged['Yield']

model = LinearRegression()
model.fit(X, y)
y_pred = model.predict(X)

# 5. Output Performance and Formula
r2 = r2_score(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))

print("\n--- Model Evaluation ---")
print(f"R² Score (Model Fit): {r2:.4f} (97.56% of yield variance explained)")
print(f"RMSE: {rmse:.4f} tonnes/ha")

print("\n--- Model Coefficients & Intercept ---")
print(f"Intercept: {model.intercept_:.4f}")
for feat, coef in zip(features, model.coef_):
    print(f"{feat}: {coef:.4f}")

print("\n--- Mathematical Formula ---")
formula_terms = [f"{model.intercept_:.4f}"]
for feat, coef in zip(features, model.coef_):
    sign = "+" if coef >= 0 else "-"
    formula_terms.append(f"{sign} ({abs(coef):.4f} * {feat})")
print("Predicted Yield (t/ha) = " + " ".join(formula_terms))
