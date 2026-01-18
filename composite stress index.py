import pandas as pd
import os
import glob


# COMPOSITE DIGITAL IDENTITY STRESS INDEX
# (Enrolment + Biometric + Demographic)



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


BASE = SCRIPT_DIR
if not os.path.exists(os.path.join(BASE, "Enrollment")):
    parent = os.path.dirname(BASE)
    if os.path.exists(os.path.join(parent, "Enrollment")):
        BASE = parent

print(" Base folder detected:", BASE)


 
# Utility functions

def folder_name_case_insensitive(target_name: str):
    """Return actual folder name matching target_name ignoring case, else None."""
    for d in os.listdir(BASE):
        if os.path.isdir(os.path.join(BASE, d)) and d.lower() == target_name.lower():
            return d
    return None


def list_csvs(module_folder: str):
    """Print all csv files inside module outputs (debug-friendly)."""
    out_dir = os.path.join(BASE, module_folder, "outputs")
    print(f"\n Checking folder: {out_dir}")
    if not os.path.exists(out_dir):
        print(" outputs folder NOT found")
        return []

    files = glob.glob(os.path.join(out_dir, "*.csv"))
    if not files:
        print(" No CSV files found in outputs/")
        return []

    for f in files:
        print("correct", os.path.basename(f))
    return files


def find_pressure_file(module_folder: str):
    """
    Find best state-pressure index file inside outputs folder.
    It auto-detects even if filename is weird.
    """
    out_dir = os.path.join(BASE, module_folder, "outputs")
    if not os.path.exists(out_dir):
        return None

    files = glob.glob(os.path.join(out_dir, "*.csv"))

    
    candidates = [
        f for f in files
        if ("state" in os.path.basename(f).lower() and "pressure" in os.path.basename(f).lower())
    ]

    if not candidates:
       
        candidates = [f for f in files if "pressure" in os.path.basename(f).lower()]

    if not candidates:
        return None

    
    candidates.sort(key=lambda x: ("index" not in os.path.basename(x).lower(), len(os.path.basename(x))))
    return candidates[0]


def load_pressure(path: str, value_name: str):
    """
    Robust loader:
    - Works if CSV has headers OR no headers
    - Works if first row accidentally contains header text
    - Ensures output columns: ['state', value_name]
    """
    df = pd.read_csv(path)

    # Case A: column 'state' exists
    if "state" in df.columns:
        value_col = [c for c in df.columns if c != "state"][0]
        df = df[["state", value_col]]
        df.columns = ["state", value_name]
        df[value_name] = pd.to_numeric(df[value_name], errors="coerce").fillna(0)
        return df

    # Case B: no header
    df = pd.read_csv(path, header=None)

    # Ensure at least 2 columns
    if df.shape[1] < 2:
        raise ValueError(f" File {os.path.basename(path)} has <2 columns. Cannot interpret.")

    df = df.iloc[:, :2]
    df.columns = ["state", value_name]

    # Drop fake header rows inside data
    df = df[df["state"].astype(str).str.lower() != "state"]

    df[value_name] = pd.to_numeric(df[value_name], errors="coerce").fillna(0)
    return df


# Detect module folder names (case-insensitive)

ENROLL_FOLDER = folder_name_case_insensitive("Enrollment")
BIO_FOLDER = folder_name_case_insensitive("Biometrics")
DEMO_FOLDER = folder_name_case_insensitive("Demographics")

if not ENROLL_FOLDER or not BIO_FOLDER or not DEMO_FOLDER:
    raise FileNotFoundError(
        f"\n Required folders missing in BASE directory.\n"
        f"Found:\n"
        f"Enrollment  : {ENROLL_FOLDER}\n"
        f"Biometrics  : {BIO_FOLDER}\n"
        f"Demographics: {DEMO_FOLDER}\n"
        f"\n Ensure folders exist exactly like: Enrollment/, Biometrics/, Demographics/"
    )

print("\n Folder names resolved:")
print("Enrollment  :", ENROLL_FOLDER)
print("Biometrics  :", BIO_FOLDER)
print("Demographics:", DEMO_FOLDER)


#  Debug list 

_ = list_csvs(ENROLL_FOLDER)
_ = list_csvs(BIO_FOLDER)
_ = list_csvs(DEMO_FOLDER)


# Locate pressure files automatically

enr_path = find_pressure_file(ENROLL_FOLDER)
bio_path = find_pressure_file(BIO_FOLDER)
demo_path = find_pressure_file(DEMO_FOLDER)

print("\n Pressure files detected:")
print("Enrolment   :", enr_path)
print("Biometric   :", bio_path)
print("Demographic :", demo_path)

if not enr_path or not bio_path or not demo_path:
    raise FileNotFoundError(
        "\n One or more pressure index CSV files missing.\n"
        "\n Ensure these exist (file names can vary):\n"
        "  Enrollment/outputs/*pressure*.csv\n"
        "  Biometrics/outputs/*pressure*.csv\n"
        "  Demographics/outputs/*pressure*.csv\n"
        "\n If Enrollment pressure file missing, re-run Enrollment EDA.py."
    )



enr = load_pressure(enr_path, "enrolment_pressure")
bio = load_pressure(bio_path, "biometric_pressure")
demo = load_pressure(demo_path, "demographic_pressure")

df = enr.merge(bio, on="state", how="outer").merge(demo, on="state", how="outer")
df = df.fillna(0)

print("\n Merged shape:", df.shape)
print(df.head())



# Normalize using percentile ranks (0–1 scale)

for col in ["enrolment_pressure", "biometric_pressure", "demographic_pressure"]:
    df[col + "_pct"] = df[col].rank(pct=True)


#  Composite Index (Equal weights)

df["composite_stress_index"] = df[
    ["enrolment_pressure_pct", "biometric_pressure_pct", "demographic_pressure_pct"]
].mean(axis=1)

df = df.sort_values("composite_stress_index", ascending=False)


# Top 10 preview

out_csv = os.path.join(BASE, "composite_stress_index.csv")
df.to_csv(out_csv, index=False)

print("\n Composite Stress Index created successfully!")
print(" Saved:", out_csv)

print("\n Top 10 states by composite stress:")
print(df[["state", "composite_stress_index"]].head(10))



#top 15 Chart

try:
    import matplotlib.pyplot as plt

    top15 = df.head(15).sort_values("composite_stress_index", ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(top15["state"], top15["composite_stress_index"])
    plt.title("Top 15 States: Composite Digital Identity Stress Index")
    plt.xlabel("Stress Index (0–1)")
    plt.tight_layout()

    out_png = os.path.join(BASE, "composite_stress_top15.png")
    plt.savefig(out_png, dpi=300)
    plt.show()

    print(" Chart saved:", out_png)

except Exception as e:
    print(" Chart step skipped due to:", e)
