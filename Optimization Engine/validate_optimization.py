
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "outputs" / "optimization" / "all_recommendations.csv"

if not FILE.exists():
    raise FileNotFoundError(
        f"Recommendations file not found: {FILE}"
    )

df = pd.read_csv(FILE)

print("\n1. TOTAL RECOMMENDATIONS")
print(f"Total: {len(df)}")

print("\n2. PRIORITY DISTRIBUTION")
print(
    df["Priority"]
    .value_counts()
    .reindex(["High", "Medium", "Low"], fill_value=0)
    .to_string()
)

print("\n3. RECOMMENDED ACTIONS")
print(df["Recommended_Action"].value_counts().to_string())

print("\n4. SLA STATUS")
print(
    df["SLA_Compliance_Status"]
    .fillna("Missing")
    .value_counts()
    .to_string()
)

print("\n5. SLA EVIDENCE STATUS")
print(
    df["SLA_Evidence_Status"]
    .fillna("Missing")
    .value_counts()
    .to_string()
)

print("\n6. HIGH-PRIORITY RECOMMENDATIONS BY ACTION")
high = df[df["Priority"] == "High"]

if high.empty:
    print("No high-priority recommendations.")
else:
    print(
        high["Recommended_Action"]
        .value_counts()
        .to_string()
    )

print("\n7. SAVINGS SCENARIO")
savings = pd.to_numeric(
    df["Illustrative_10pct_Savings_INR"],
    errors="coerce"
).fillna(0)

print(f"Records with estimated savings: {(savings > 0).sum()}")
print(f"Total illustrative scenario: ₹{savings.sum():,.2f}")

print("\n8. SAFETY CHECKS")
print(
    "Manual approval required for every record:",
    df["Manual_Approval_Required"].astype(str)
    .str.lower().isin(["true", "1"]).all()
)
print(
    "Automatic modifications disabled for every record:",
    df["Automatic_Modification_Allowed"].astype(str)
    .str.lower().isin(["false", "0"]).all()
)

print("\nValidation report complete.")
