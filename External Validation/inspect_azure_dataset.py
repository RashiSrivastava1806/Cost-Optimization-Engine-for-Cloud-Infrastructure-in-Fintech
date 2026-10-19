from pathlib import Path
import pandas as pd

source = Path(
    "data/external_validation/azure_costs/extracted/anonymized_costs.csv"
)
out = Path("data/external_validation/azure_costs/diagnostics")
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source, low_memory=False)

df["Date"] = pd.to_datetime(
    df["Date"], format="%m/%d/%Y", errors="coerce"
)
df["CostInBillingCurrency"] = pd.to_numeric(
    df["CostInBillingCurrency"], errors="coerce"
)

print("\n=== DATE COVERAGE ===")
print("Invalid dates:", df["Date"].isna().sum())
print("Earliest date:", df["Date"].min())
print("Latest date:", df["Date"].max())
print("Unique dates:", df["Date"].nunique())

print("\n=== DUPLICATE IMPACT ===")
print("Original rows:", len(df))
print("Exact duplicate rows:", df.duplicated().sum())
print("Rows after deduplication:", len(df.drop_duplicates()))
print("Original cost total:", df["CostInBillingCurrency"].sum())
print("Deduplicated cost total:",
      df.drop_duplicates()["CostInBillingCurrency"].sum())

valid = df.dropna(subset=["Date", "CostInBillingCurrency"])

daily = (
    valid.groupby("Date")
    .agg(
        daily_cost=("CostInBillingCurrency", "sum"),
        billing_records=("CostInBillingCurrency", "size"),
        unique_resources=("ResourceName", "nunique"),
        unique_services=("ConsumedService", "nunique"),
    )
    .reset_index()
    .sort_values("Date")
)
daily.to_csv(out / "azure_daily_cost_summary.csv", index=False)

services = (
    valid.groupby("MeterCategory")["CostInBillingCurrency"]
    .agg(["sum", "count"])
    .sort_values("sum", ascending=False)
)
services.to_csv(out / "azure_service_cost_summary.csv")

print("\n=== DAILY SUMMARY ===")
print("Daily observations:", len(daily))
print("Days with zero cost:", (daily["daily_cost"] == 0).sum())
print(daily.head().to_string(index=False))

print("\n=== TOP 15 METER CATEGORIES ===")
print(services.head(15).round(4).to_string())

print("\nSaved diagnostic outputs to:", out)
print("Original dataset was not modified.")
