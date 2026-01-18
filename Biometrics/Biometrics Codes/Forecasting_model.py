import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import os


# GET SCRIPT DIRECTORY

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.dirname(BASE_DIR)  # Biometrics folder


# LOAD BIOMETRIC CSV FILES

file_paths = [
    os.path.join(DATA_DIR, "1.biometric.csv"),
    os.path.join(DATA_DIR, "2.biometric.csv"),
    os.path.join(DATA_DIR, "3.biometric.csv"),
    os.path.join(DATA_DIR, "4.biometric.csv"),
]

dfs = [pd.read_csv(p) for p in file_paths]
biometric_master = pd.concat(dfs, ignore_index=True)


# DATA PREPROCESSING

biometric_master["date"] = pd.to_datetime(
    biometric_master["date"], dayfirst=True
)

biometric_master["total_biometric_updates"] = (
    biometric_master["bio_age_5_17"] +
    biometric_master["bio_age_17_"]
)

biometric_master["year"] = biometric_master["date"].dt.year
biometric_master["month"] = biometric_master["date"].dt.month


# MONTHLY AGGREGATION

monthly_data = (
    biometric_master
    .groupby(["year", "month"])["total_biometric_updates"]
    .sum()
    .reset_index()
)

monthly_data["t"] = range(len(monthly_data))


# TRAIN FORECAST MODEL

X = monthly_data[["t"]]
y = monthly_data["total_biometric_updates"]

model = LinearRegression()
model.fit(X, y)


# FORECAST NEXT 6 MONTHS

future_t = np.arange(
    len(monthly_data),
    len(monthly_data) + 6
).reshape(-1, 1)

future_predictions = model.predict(future_t)

print("\nForecasted biometric updates (next 6 months):")
for i, val in enumerate(future_predictions, start=1):
    print(f"Month +{i}: {int(val)}")



#forecasting_model visualization
import matplotlib.pyplot as plt

# PREDICT ON TRAINING DATA (ACTUAL vs PREDICTED)

monthly_data["predicted"] = model.predict(X)


# VISUALIZATION: ACTUAL vs PREDICTED

plt.figure(figsize=(10, 5))
plt.plot(
    monthly_data["t"],
    monthly_data["total_biometric_updates"],
    label="Actual Updates",
    marker="o"
)
plt.plot(
    monthly_data["t"],
    monthly_data["predicted"],
    label="Predicted Updates",
    linestyle="--",
    marker="x"
)

plt.title("Biometric Updates: Actual vs Predicted")
plt.xlabel("Time (Months)")
plt.ylabel("Total Biometric Updates")
plt.legend()
plt.grid(True)


# SAVE

plt.savefig(
    os.path.join(DATA_DIR, "Biometric_Actual_vs_Predicted.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.show()
