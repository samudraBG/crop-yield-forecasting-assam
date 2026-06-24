

import pandas as pd

# Load CSV file (change name if you uploaded a different one)
df = pd.read_csv("/content/FAOSTAT_data_en_5-23-2025.csv")

# View columns and first few rows
print(df.columns.tolist())
df.head()

# Drop irrelevant columns (if present)
columns_to_drop = ["Domain Code", "Area Code", "Element Code", "Item Code", "Flag", "Flag Description"]
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns])

# Strip whitespace and replace spaces in column names
df.columns = df.columns.str.strip().str.replace(" ", "_")

# Filter for India (assuming no state-wise split in FAOSTAT)
df = df[df["Area"] == "India"]

# Remove rows with non-positive values
df = df[df["Value"] > 0]

df.head()

# Pivot Element to columns: Yield, Production, Area harvested
df_pivot = df.pivot_table(
    values="Value",
    index=["Year", "Item"],
    columns="Element",
    aggfunc="mean"
).reset_index()

# Flatten multi-index
df_pivot.columns.name = None
df_pivot = df_pivot.rename(columns={
    "Item": "Crop",
    "Area harvested": "Area_harvested",
    "Yield": "Yield",
    "Production": "Production"
})

df_pivot.head()

# Sort by Crop and Year for proper lagging
df_pivot = df_pivot.sort_values(by=["Crop", "Year"])

# Create a lag feature: previous year's yield
df_pivot["Yield_Lag_1"] = df_pivot.groupby("Crop")["Yield"].shift(1)

# Drop rows with missing lag values
df_model = df_pivot.dropna()

# Check result
df_model.head()

df["Element"].unique()

from sklearn.preprocessing import LabelEncoder

# Encode the crop names into numbers
le = LabelEncoder()
df_model["Crop_Encoded"] = le.fit_transform(df_model["Crop"])

# Check the mapping (optional)
crop_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print("Crop Encoding:", crop_mapping)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Features and target
features = ["Year", "Yield_Lag_1", "Crop_Encoded"]
X = df_model[features]
y = df_model["Yield"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("R² Score:", r2_score(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

# Predict for Rice in 2025 (example)
crop_name = "Rice"
if crop_name in le.classes_:
    crop_code = le.transform([crop_name])[0]

    sample_input = pd.DataFrame([{
        "Year": 2025,
        "Yield_Lag_1": 2.3,  # replace with actual 2024 yield if available
        "Crop_Encoded": crop_code
    }])

    predicted_yield = model.predict(sample_input)[0]
    print(f"Predicted yield for {crop_name} in 2025: {predicted_yield:.2f} tonnes/hectare")
else:
    print(f"{crop_name} not found in the dataset.")

import matplotlib.pyplot as plt

importances = model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(8, 4))
plt.barh(feature_names, importances)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.show()

import pandas as pd

sample = pd.DataFrame([{
    "Year": 2025,
    "Yield_Lag_1": 2.3,
    "Crop_Encoded": 34
}])

sample.to_csv("sample_input.csv", index=False)

pip install pandas scikit-learn xgboost matplotlib

import pandas as pd

# Weather Data (assumed from earlier)
weather_merged = pd.DataFrame({
    "YEAR": [2022, 2023, 2024],
    "T2M_MAX": [25.07, 26.23, 25.69],
    "T2M_MIN": [7.24, 7.87, 7.85],
    "RH2M": [76.31, 73.33, 74.36]
})

# Sample Yield Data
yield_df = pd.DataFrame({
    "YEAR": [2022, 2023, 2024],
    "Yield": [2.35, 2.50, 2.70]
})

# Merge on YEAR
merged_df = pd.merge(weather_merged, yield_df, on="YEAR")
print(merged_df)

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from xgboost import XGBRegressor
import numpy as np

# Features and Target
X = merged_df[["T2M_MAX", "T2M_MIN", "RH2M"]]
y = merged_df["Yield"]

# Use all for training due to low sample size
model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
model.fit(X, y)

# Predict and evaluate on same data (note: not best practice but ok for small dataset)
preds = model.predict(X)
r2 = r2_score(y, preds)
rmse = np.sqrt(mean_squared_error(y, preds))

print(f"R² Score: {r2:.2f}")
print(f"RMSE: {rmse:.2f}")

import pickle

# Save the trained model
with open("rice_yield_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Example 2025 data (from your cleaned weather file)
test_2025 = pd.DataFrame({
    "T2M_MAX": [22.92],
    "T2M_MIN": [4.04],
    "RH2M": [66.63]
})

# Load model and predict
with open("rice_yield_model.pkl", "rb") as f:
    loaded_model = pickle.load(f)

prediction_2025 = loaded_model.predict(test_2025)
print(f"Predicted Rice Yield for 2025: {prediction_2025[0]:.2f} tons/hectare")

from google.colab import files
files.download("rice_yield_model.pkl")

