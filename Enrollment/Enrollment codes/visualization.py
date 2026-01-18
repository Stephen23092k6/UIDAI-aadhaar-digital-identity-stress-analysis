import pandas as pd
import matplotlib.pyplot as plt
import os


# PATH

CODES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODES_DIR)          # Enrollment/
OUT_DIR = os.path.join(BASE_DIR, "outputs")    # Enrollment/outputs
os.makedirs(OUT_DIR, exist_ok=True)


# LOAD MASTER DATA (SAFE)

master_path = os.path.join(OUT_DIR, "enrolment_master.csv")

if not os.path.exists(master_path):
    raise FileNotFoundError(
        f" File not found: {master_path}\n"
        f" Run EDA.py first to generate enrolment_master.csv inside outputs/"
    )

df = pd.read_csv(master_path)

# Safe date parse
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df = df.dropna(subset=["date"])

print(" Loaded master:", master_path)
print(df.head())


# 1) AGE SHARE

age_totals = {
    "0–5": df["age_0_5"].sum(),
    "5–17": df["age_5_17"].sum(),
    "18+": df["age_18_greater"].sum()
}

plt.figure(figsize=(6, 6))
plt.pie(age_totals.values(), labels=age_totals.keys(), autopct="%1.1f%%")
plt.title("Enrolments: Age Share Distribution")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "enrol_age_share.png"), dpi=300)
plt.show()


# 2) MONTHLY TREND

monthly = df.groupby(df["date"].dt.to_period("M"))["total_enrolment"].sum()
monthly.index = monthly.index.to_timestamp()

plt.figure(figsize=(12, 5))
plt.plot(monthly.index, monthly.values, marker="o")
plt.title("Enrolments: Monthly Trend")
plt.xlabel("Month")
plt.ylabel("Total Enrolments")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "enrol_monthly_trend.png"), dpi=300)
plt.show()


# 3) TOP STATES PRESSURE (ROBUST READ)

pressure_path = os.path.join(OUT_DIR, "enrolment_state_pressure_index.csv")

if not os.path.exists(pressure_path):
    raise FileNotFoundError(
        f" File not found: {pressure_path}\n"
        f" Run EDA.py first to generate enrolment_state_pressure_index.csv"
    )

state_pressure = pd.read_csv(pressure_path)

# Handle if saved without headers
if state_pressure.shape[1] == 2 and "state" not in state_pressure.columns:
    state_pressure.columns = ["state", "avg_daily_enrolment"]
elif "state" in state_pressure.columns:
    # sometimes it saves value column name differently — select second column automatically
    value_col = [c for c in state_pressure.columns if c != "state"][0]
    state_pressure = state_pressure[["state", value_col]]
    state_pressure.columns = ["state", "avg_daily_enrolment"]
else:
    # fallback if weird format
    state_pressure = pd.read_csv(pressure_path, header=None)
    state_pressure.columns = ["state", "avg_daily_enrolment"]

top10 = state_pressure.sort_values("avg_daily_enrolment", ascending=False).head(10)

plt.figure(figsize=(10, 6))
plt.barh(top10["state"], top10["avg_daily_enrolment"])
plt.title("Top 10 States: Enrolment Pressure (Avg Daily)")
plt.xlabel("Avg Daily Enrolments")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "enrol_top10_pressure.png"), dpi=300)
plt.show()


# 4) SEASONALITY (SUPER ROBUST)

seasonality_path = os.path.join(OUT_DIR, "enrolment_seasonality.csv")

if not os.path.exists(seasonality_path):
    raise FileNotFoundError(
        f" File not found: {seasonality_path}\n"
        f" Run EDA.py first to generate enrolment_seasonality.csv"
    )

seasonality = pd.read_csv(seasonality_path)

# If "month" missing, treated as no-header file
if "month" not in seasonality.columns:
    seasonality = pd.read_csv(seasonality_path, header=None)
    seasonality.columns = ["month", "value"]

# Identifing numeric column automatically (not month)
value_col = [c for c in seasonality.columns if c != "month"][0]

seasonality["month"] = pd.to_numeric(seasonality["month"], errors="coerce")
seasonality = seasonality.dropna(subset=["month"])
seasonality["month"] = seasonality["month"].astype(int)

seasonality[value_col] = pd.to_numeric(seasonality[value_col], errors="coerce").fillna(0)

plt.figure(figsize=(10, 4))
plt.plot(seasonality["month"], seasonality[value_col], marker="o")
plt.title("Enrolments: Seasonality Pattern (Month-wise Avg)")
plt.xlabel("Month")
plt.ylabel("Average Enrolments")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "enrol_seasonality.png"), dpi=300)
plt.show()


# 5) PINCODE TIERS

tiers_path = os.path.join(OUT_DIR, "enrolment_pincode_tiers.csv")

if not os.path.exists(tiers_path):
    raise FileNotFoundError(
        f" File not found: {tiers_path}\n"
        f" Run EDA.py first to generate enrolment_pincode_tiers.csv"
    )

tiers = pd.read_csv(tiers_path)
tier_counts = tiers["pincode_tier"].value_counts()

plt.figure(figsize=(7, 4))
plt.bar(tier_counts.index, tier_counts.values)
plt.title("Enrolment Activity: Pincode Tier Distribution (Proxy)")
plt.xlabel("Tier")
plt.ylabel("Number of Pincodes")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "enrol_pincode_tiers.png"), dpi=300)
plt.show()


# 6) OUTLIERS

outliers_path = os.path.join(OUT_DIR, "enrolment_outliers.csv")

if not os.path.exists(outliers_path):
    raise FileNotFoundError(
        f" File not found: {outliers_path}\n"
        f" Run EDA.py first to generate enrolment_outliers.csv"
    )

outliers = pd.read_csv(outliers_path)
outliers["date"] = pd.to_datetime(outliers["date"], dayfirst=True, errors="coerce")
outliers = outliers.dropna(subset=["date"])

plt.figure(figsize=(12, 5))
plt.scatter(df["date"], df["total_enrolment"], s=2, alpha=0.3)
plt.scatter(outliers["date"], outliers["total_enrolment"], s=15)
plt.title("Enrolments: Outlier Spikes Highlighted")
plt.xlabel("Date")
plt.ylabel("Total Enrolments")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "enrol_outliers_spikes.png"), dpi=300)
plt.show()

print(" Enrolment visuals generated in:", OUT_DIR)
