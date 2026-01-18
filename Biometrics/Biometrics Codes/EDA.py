import pandas as pd
import os


# PATHS 

CODES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODES_DIR)  # Biometrics/
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

# FILES

file_paths = [
    os.path.join(BASE_DIR, "1.biometric.csv"),
    os.path.join(BASE_DIR, "2.biometric.csv"),
    os.path.join(BASE_DIR, "3.biometric.csv"),
    os.path.join(BASE_DIR, "4.biometric.csv"),
]


# LOAD + MERGE

dfs = [pd.read_csv(p) for p in file_paths]
df = pd.concat(dfs, ignore_index=True)


# STANDARDIZE COLUMNS

df.columns = df.columns.str.lower().str.strip()

required_cols = ["date", "state", "district", "pincode", "bio_age_5_17", "bio_age_17_"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(" BIOMETRICS: Missing required columns:", missing)
    print(" Found columns:", df.columns.tolist())
    raise KeyError("Schema mismatch in biometric dataset.")

# DATE CLEANING (robust + debug)

df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
invalid_dates = df["date"].isna().sum()
print(f" BIOMETRICS: Invalid date rows detected: {invalid_dates}")
df = df.dropna(subset=["date"])
print(f" BIOMETRICS: Rows after date cleaning: {df.shape[0]}")

# NUMERIC CLEANING

for c in ["bio_age_5_17", "bio_age_17_"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)


# FEATURES

df["total_biometric_updates"] = df["bio_age_5_17"] + df["bio_age_17_"]
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# life-stage ratio signal
df["youth_to_adult_update_ratio"] = df["bio_age_5_17"] / (df["bio_age_17_"] + 1)


# PREVIEW

print("\n=== BIOMETRIC MASTER PREVIEW ===")
print(df.head(10))
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())


# SAVE MASTER

master_path = os.path.join(OUT_DIR, "biometric_master.csv")
df.to_csv(master_path, index=False)
print(f" Saved: {master_path}")


#Reports



# Coverage
print("\n=== BIOMETRICS: DATA COVERAGE ===")
print("Date Range:", df["date"].min(), "to", df["date"].max())
print("States:", df["state"].nunique(), "| Districts:", df["district"].nunique(), "| Pincodes:", df["pincode"].nunique())
print("\nMissing Values Summary:\n", df.isna().sum())

# State Pressure Index (avg daily biometric updates)
state_pressure = (
    df.groupby("state")["total_biometric_updates"]
    .mean()
    .sort_values(ascending=False)
)
state_pressure_path = os.path.join(OUT_DIR, "biometric_state_pressure_index.csv")
state_pressure.to_csv(state_pressure_path)
print("\nTop High-Pressure States (avg daily biometric updates):")
print(state_pressure.head(10))
print(f" Saved: {state_pressure_path}")

# Seasonality (month-wise avg)
seasonality = df.groupby("month")["total_biometric_updates"].mean().sort_index()
seasonality_path = os.path.join(OUT_DIR, "biometric_seasonality.csv")
seasonality.to_csv(seasonality_path)
print("\nMonthly Seasonality (avg biometric updates):")
print(seasonality)
print(f" Saved: {seasonality_path}")

# Outliers (spikes)
threshold = df["total_biometric_updates"].mean() + 3 * df["total_biometric_updates"].std()
outliers = df[df["total_biometric_updates"] > threshold].copy()
outliers_path = os.path.join(OUT_DIR, "biometric_outliers.csv")
outliers.to_csv(outliers_path, index=False)
print("\n=== BIOMETRICS: OUTLIERS ===")
print("Threshold:", int(threshold), "| Outlier rows:", outliers.shape[0])
print(outliers[["date", "state", "district", "pincode", "total_biometric_updates"]].head(10))
print(f" Saved: {outliers_path}")


# PINCODE ACTIVITY TIERS (Urban/Semi-urban/Rural Proxy)

pincode_totals = df.groupby("pincode")["total_biometric_updates"].sum().reset_index()
pincode_totals["activity_percentile"] = pincode_totals["total_biometric_updates"].rank(pct=True)

def tier_from_pct(p):
    if p >= 0.80:
        return "URBAN_PROXY"
    elif p >= 0.30:
        return "SEMI_URBAN_PROXY"
    else:
        return "RURAL_PROXY"

pincode_totals["pincode_tier"] = pincode_totals["activity_percentile"].apply(tier_from_pct)

tiers_path = os.path.join(OUT_DIR, "biometric_pincode_tiers.csv")
pincode_totals.to_csv(tiers_path, index=False)

tier_summary = pincode_totals.groupby("pincode_tier")["total_biometric_updates"].sum().sort_values(ascending=False)
tier_summary_path = os.path.join(OUT_DIR, "biometric_tier_summary.csv")
tier_summary.to_csv(tier_summary_path)

print("\n=== BIOMETRICS: PINCODE TIER SUMMARY ===")
print(tier_summary)
print(f" Saved tiers: {tiers_path}")
print(f" Saved tier summary: {tier_summary_path}")

# Insight summary
print("\n=== KEY INSIGHTS SUMMARY (Biometrics) ===")
print("- Biometric demand reflects periodic identity maintenance and life-stage transitions.")
print("- State Pressure Index highlights operational load hotspots for UIDAI centers.")
print("- Pincode tiers provide an urban/semi-urban/rural proxy without external lookup.")
print("- Outlier spikes can indicate drives, backlog clearance, or localized surges.")

