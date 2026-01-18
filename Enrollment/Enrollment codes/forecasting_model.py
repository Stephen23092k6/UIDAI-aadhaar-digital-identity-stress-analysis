import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.linear_model import LinearRegression


# PATH SETUP

CODES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODES_DIR)
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)


# LOAD MASTER DATA

master_path = os.path.join(OUT_DIR, "enrolment_master.csv")
df = pd.read_csv(master_path)

df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])


# MONTHLY AGGREGATION

monthly = (
    df.groupby(df["date"].dt.to_period("M"))["total_enrolment"]
    .sum()
    .reset_index()
)

monthly["date"] = monthly["date"].dt.to_timestamp()
monthly["t"] = np.arange(len(monthly))


# TRAIN MODEL

X = monthly[["t"]]
y = monthly["total_enrolment"]

model = LinearRegression()
model.fit(X, y)


# FORECAST NEXT 6 MONTHS

future_t = np.arange(len(monthly), len(monthly) + 6).reshape(-1, 1)
future_pred = model.predict(future_t)

future_dates = pd.date_range(
    start=monthly["date"].iloc[-1] + pd.offsets.MonthBegin(1),
    periods=6,
    freq="MS"
)

forecast_df = pd.DataFrame({
    "date": future_dates,
    "predicted_enrolment": future_pred
})

forecast_df.to_csv(os.path.join(OUT_DIR, "enrolment_forecast.csv"), index=False)


# VISUAL: ACTUAL vs PREDICTED

plt.figure(figsize=(12, 5))
plt.plot(monthly["date"], y, label="Actual", marker="o")
plt.plot(future_dates, future_pred, label="Forecast", marker="o", linestyle="--")
plt.title("Enrolment Forecast (Next 6 Months)")
plt.xlabel("Month")
plt.ylabel("Total Enrolments")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "enrolment_forecast.png"), dpi=300)
plt.show()

print(" Enrolment forecasting completed")
print(forecast_df)
