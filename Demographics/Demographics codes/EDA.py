import pandas as pd
import os

CODES_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(CODES_DIR)  # Demographics/
OUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUT_DIR, exist_ok=True)

file_paths = [
    os.path.join(BASE_DIR, "1.demo.csv"),
    os.path.join(BASE_DIR, "2.demo.csv"),
    os.path.join(BASE_DIR, "3.demo.csv"),
    os.path.join(BASE_DIR, "4.demo.csv"),
    os.path.join(BASE_DIR, "5.demo.csv"),
]

dfs = [pd.read_csv(p) for p in file_paths]
df = pd.concat(dfs, ignore_index=True)
df.columns = df.columns.str.lower().str.strip()

required_cols = ["date", "state", "district", "pincode", "demo_age_5_17", "demo_age_17_"]
missing = [c for c in required_cols if c not in df.columns]
if missing:
    print(" DEMOGRAPHICS: Missing required columns:", missing)
    print(" Found columns:", df.columns.tolist())
    raise KeyError("Schema mismatch in demographic dataset.")

df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
invalid_dates = df["date"].isna().sum()
print(f" DEMOGRAPHICS: Invalid date rows detected: {invalid_dates}")
df = df.dropna(subset=["date"])
print(f" DEMOGRAPHICS: Rows after date cleaning: {df.shape[0]}")

for c in ["demo_age_5_17", "demo_age_17_"]:
    df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)

df["total_demographic_updates"] = df["demo_age_5_17"] + df["demo_age_17_"]
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

df["youth_to_adult_update_ratio"] = df["demo_age_5_17"] / (df["demo_age_17_"] + 1)

print("\n=== DEMOGRAPHIC MASTER PREVIEW ===")
print(df.head(10))
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())

master_path = os.path.join(OUT_DIR, "demographic_master.csv")
df.to_csv(master_path, index=False)
print(f" Saved: {master_path}")

print("\n=== DEMOGRAPHICS: DATA COVERAGE ===")
print("Date Range:", df["date"].min(), "to", df["date"].max())
print("States:", df["state"].nunique(), "| Districts:", df["district"].nunique(), "| Pincodes:", df["pincode"].nunique())
print("\nMissing Values Summary:\n", df.isna().sum())

state_pressure = (
    df.groupby("state")["total_demographic_updates"]
    .mean()
    .sort_values(ascending=False)
)
state_pressure_path = os.path.join(OUT_DIR, "demographic_state_pressure_index.csv")
state_pressure.to_csv(state_pressure_path)
print("\nTop High-Pressure States (avg daily demographic updates):")
print(state_pressure.head(10))
print(f" Saved: {state_pressure_path}")

seasonality = df.groupby("month")["total_demographic_updates"].mean().sort_index()
seasonality_path = os.path.join(OUT_DIR, "demographic_seasonality.csv")
seasonality.to_csv(seasonality_path)
print("\nMonthly Seasonality (avg demographic updates):")
print(seasonality)
print(f" Saved: {seasonality_path}")

threshold = df["total_demographic_updates"].mean() + 3 * df["total_demographic_updates"].std()
outliers = df[df["total_demographic_updates"] > threshold].copy()
outliers_path = os.path.join(OUT_DIR, "demographic_outliers.csv")
outliers.to_csv(outliers_path, index=False)

print("\n=== DEMOGRAPHICS: OUTLIERS ===")
print("Threshold:", int(threshold), "| Outlier rows:", outliers.shape[0])
print(outliers[["date", "state", "district", "pincode", "total_demographic_updates"]].head(10))
print(f" Saved: {outliers_path}")

# PINCODE TIERS
pincode_totals = df.groupby("pincode")["total_demographic_updates"].sum().reset_index()
pincode_totals["activity_percentile"] = pincode_totals["total_demographic_updates"].rank(pct=True)

def tier_from_pct(p):
    if p >= 0.80:
        return "URBAN_PROXY"
    elif p >= 0.30:
        return "SEMI_URBAN_PROXY"
    else:
        return "RURAL_PROXY"

pincode_totals["pincode_tier"] = pincode_totals["activity_percentile"].apply(tier_from_pct)

tiers_path = os.path.join(OUT_DIR, "demographic_pincode_tiers.csv")
pincode_totals.to_csv(tiers_path, index=False)

tier_summary = pincode_totals.groupby("pincode_tier")["total_demographic_updates"].sum().sort_values(ascending=False)
tier_summary_path = os.path.join(OUT_DIR, "demographic_tier_summary.csv")
tier_summary.to_csv(tier_summary_path)

print("\n=== DEMOGRAPHICS: PINCODE TIER SUMMARY ===")
print(tier_summary)
print(f" Saved tiers: {tiers_path}")
print(f" Saved tier summary: {tier_summary_path}")

print("\n=== KEY INSIGHTS SUMMARY (Demographics) ===")
print("- Demographic activity shows where identity updates are concentrated regionally.")
print("- State Pressure Index helps prioritize service capacity and verification resources.")
print("- Pincode tiers offer an explainable proxy for urban/semi-urban/rural activity.")
print("- Outliers may indicate correction drives, mobility-linked changes, or backlog events.")
