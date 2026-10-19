import json
import zipfile
from pathlib import Path
from collections import Counter

import pandas as pd

BASE = Path("data/external_validation")
ARCHIVE = BASE / "raw/dataset-examples.zip"
RESULTS = BASE / "results"
TARGET = "EA-Cost-FOCUS_1.0.csv"
CHUNK_SIZE = 100_000

RESULTS.mkdir(parents=True, exist_ok=True)

if not ARCHIVE.exists():
    raise FileNotFoundError(f"Archive not found: {ARCHIVE}")

required_columns = [
    "BilledCost",
    "BillingCurrency",
    "ChargePeriodStart",
    "ChargePeriodEnd",
    "ServiceName",
    "ChargeCategory",
    "ResourceId",
]

row_count = 0
invalid_cost_rows = 0
missing_cost_rows = 0
negative_cost_rows = 0
zero_cost_rows = 0
invalid_start_dates = 0
missing_counts = Counter()
currency_rows = Counter()
currency_cost_totals = Counter()
currency_negative_rows = Counter()
currency_invalid_cost_rows = Counter()
monthly_parts = []
service_parts = []
category_parts = []
date_min = None
date_max = None
columns = None
chunk_count = 0

def clean_group(series):
    return (
        series.astype("string")
        .str.strip()
        .replace("", pd.NA)
        .fillna("(missing)")
    )

print("Starting external FOCUS billing validation...")
print("Archive:", ARCHIVE)
print("Dataset:", TARGET)
print("Chunk size:", f"{CHUNK_SIZE:,}")
print("No canonical project outputs will be changed.")

with zipfile.ZipFile(ARCHIVE, "r") as archive:
    if TARGET not in archive.namelist():
        raise FileNotFoundError(f"{TARGET} not found in archive")

    with archive.open(TARGET) as raw:
        reader = pd.read_csv(
            raw,
            encoding="utf-8-sig",
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for chunk in reader:
            chunk_count += 1
            row_count += len(chunk)

            if columns is None:
                columns = list(chunk.columns)
                missing_required_columns = [
                    col for col in required_columns
                    if col not in columns
                ]
                if missing_required_columns:
                    raise ValueError(
                        "Missing expected columns: "
                        + ", ".join(missing_required_columns)
                    )

                print("Columns found:", len(columns))

            # Count missing/blank values for important fields.
            for col in required_columns:
                values = chunk[col]
                missing = (
                    values.isna()
                    | values.astype("string").str.strip().eq("").fillna(False)
                )
                missing_counts[col] += int(missing.sum())

            cost = pd.to_numeric(chunk["BilledCost"], errors="coerce")
            invalid = cost.isna()
            negative = cost.lt(0).fillna(False)
            zero = cost.eq(0).fillna(False)

            missing_cost_rows += int(chunk["BilledCost"].isna().sum())
            invalid_cost_rows += int(invalid.sum())
            negative_cost_rows += int(negative.sum())
            zero_cost_rows += int(zero.sum())

            currency = clean_group(chunk["BillingCurrency"])
            currency_rows.update(currency.tolist())

            for curr, value in (
                pd.DataFrame({"currency": currency, "cost": cost})
                .groupby("currency", dropna=False)["cost"]
                .sum()
                .items()
            ):
                currency_cost_totals[str(curr)] += float(value)

            negative_by_currency = currency[negative].value_counts()
            for curr, value in negative_by_currency.items():
                currency_negative_rows[str(curr)] += int(value)

            invalid_by_currency = currency[invalid].value_counts()
            for curr, value in invalid_by_currency.items():
                currency_invalid_cost_rows[str(curr)] += int(value)

            # Charge-period date coverage and monthly aggregation.
            start = pd.to_datetime(
                chunk["ChargePeriodStart"], errors="coerce", utc=True
            )
            invalid_start_dates += int(start.isna().sum())

            valid_dates = start.dropna()
            if not valid_dates.empty:
                current_min = valid_dates.min()
                current_max = valid_dates.max()
                date_min = current_min if date_min is None else min(date_min, current_min)
                date_max = current_max if date_max is None else max(date_max, current_max)

            month = start.dt.strftime("%Y-%m").fillna("(invalid date)")
            monthly = pd.DataFrame({
                "Month": month,
                "Currency": currency,
                "BilledCost": cost,
            })
            monthly_parts.append(
                monthly.groupby(["Month", "Currency"], as_index=False)
                .agg(Records=("BilledCost", "size"),
                     BilledCost=("BilledCost", "sum"))
            )

            # Service and charge-category summaries.
            service = clean_group(chunk["ServiceName"])
            category = clean_group(chunk["ChargeCategory"])

            service_parts.append(
                pd.DataFrame({
                    "ServiceName": service,
                    "Currency": currency,
                    "BilledCost": cost,
                })
                .groupby(["ServiceName", "Currency"], as_index=False)
                .agg(Records=("BilledCost", "size"),
                     BilledCost=("BilledCost", "sum"))
            )

            category_parts.append(
                pd.DataFrame({
                    "ChargeCategory": category,
                    "Currency": currency,
                    "BilledCost": cost,
                })
                .groupby(["ChargeCategory", "Currency"], as_index=False)
                .agg(Records=("BilledCost", "size"),
                     BilledCost=("BilledCost", "sum"))
            )

            print(
                f"Processed chunk {chunk_count}: "
                f"{row_count:,} total rows",
                flush=True,
            )

def combine_parts(parts, keys):
    combined = pd.concat(parts, ignore_index=True)
    return (
        combined.groupby(keys, as_index=False, dropna=False)
        .agg(
            Records=("Records", "sum"),
            BilledCost=("BilledCost", "sum"),
        )
    )

currency_summary = pd.DataFrame([
    {
        "Currency": curr,
        "Records": count,
        "BilledCostSum": currency_cost_totals[curr],
        "NegativeCostRows": currency_negative_rows[curr],
        "InvalidCostRows": currency_invalid_cost_rows[curr],
    }
    for curr, count in currency_rows.items()
]).sort_values("Currency")

service_summary = combine_parts(
    service_parts, ["ServiceName", "Currency"]
).sort_values(["Currency", "BilledCost"], ascending=[True, False])

category_summary = combine_parts(
    category_parts, ["ChargeCategory", "Currency"]
).sort_values(["Currency", "BilledCost"], ascending=[True, False])

monthly_summary = combine_parts(
    monthly_parts, ["Month", "Currency"]
).sort_values(["Currency", "Month"])

quality_summary = pd.DataFrame([
    {"Check": "Total rows processed", "Count": row_count},
    {"Check": "CSV columns", "Count": len(columns or [])},
    {"Check": "Missing cost values", "Count": missing_cost_rows},
    {"Check": "Invalid/unparseable cost values", "Count": invalid_cost_rows},
    {"Check": "Negative billed cost rows", "Count": negative_cost_rows},
    {"Check": "Zero billed cost rows", "Count": zero_cost_rows},
    {"Check": "Invalid charge-period start dates", "Count": invalid_start_dates},
] + [
    {"Check": f"Missing/blank {col}", "Count": count}
    for col, count in sorted(missing_counts.items())
])

currency_summary.to_csv(RESULTS / "currency_summary.csv", index=False)
service_summary.to_csv(RESULTS / "service_cost_summary.csv", index=False)
category_summary.to_csv(RESULTS / "charge_category_summary.csv", index=False)
monthly_summary.to_csv(RESULTS / "monthly_cost_summary.csv", index=False)
quality_summary.to_csv(RESULTS / "data_quality_summary.csv", index=False)

summary = {
    "dataset": TARGET,
    "source": "Microsoft FinOps Toolkit dataset examples",
    "validation_type": "External billing schema, data quality and cost aggregation",
    "rows_processed": row_count,
    "column_count": len(columns or []),
    "columns": columns,
    "chunk_count": chunk_count,
    "currency_record_counts": dict(currency_rows),
    "currency_billed_cost_sums": dict(currency_cost_totals),
    "missing_cost_rows": missing_cost_rows,
    "invalid_cost_rows": invalid_cost_rows,
    "negative_cost_rows": negative_cost_rows,
    "zero_cost_rows": zero_cost_rows,
    "invalid_charge_period_start_dates": invalid_start_dates,
    "missing_values_by_key_field": dict(missing_counts),
    "charge_period_start_min_utc": (
        date_min.isoformat() if date_min is not None else None
    ),
    "charge_period_start_max_utc": (
        date_max.isoformat() if date_max is not None else None
    ),
    "limitations": [
        "Demo billing data, not production customer evidence",
        "BilledCost totals are grouped by BillingCurrency",
        "No CPU or memory utilization validation unless telemetry is separately available",
        "No contractual SLA compliance conclusion from billing data alone",
        "Negative charges are retained and flagged, not discarded",
        "Exact duplicate rows were not globally tested",
    ],
}

with open(RESULTS / "external_validation_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

print("\n=== VALIDATION SUMMARY ===")
print("Rows:", f"{row_count:,}")
print("Columns:", len(columns or []))
print("Chunks processed:", chunk_count)
print("Currency summary:")
print(currency_summary.to_string(index=False))
print("\nKey data quality checks:")
print(quality_summary.to_string(index=False))
print("\nCharge period start range (UTC):", summary["charge_period_start_min_utc"],
      "to", summary["charge_period_start_max_utc"])
print("\nCSV and JSON summaries saved to:", RESULTS)

# Generate charts if matplotlib is available.
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Charts must not combine monetary amounts across currencies.
    currency_choice = (
        currency_summary.sort_values("Records", ascending=False)
        .iloc[0]["Currency"]
    )
    curr_services = service_summary[
        service_summary["Currency"] == currency_choice
    ].nlargest(10, "BilledCost").sort_values("BilledCost")
    curr_months = monthly_summary[
        monthly_summary["Currency"] == currency_choice
    ].sort_values("Month")
    curr_categories = category_summary[
        category_summary["Currency"] == currency_choice
    ].sort_values("BilledCost")

    plt.figure(figsize=(10, 6))
    plt.barh(curr_services["ServiceName"], curr_services["BilledCost"])
    plt.xlabel(f"BilledCost ({currency_choice})")
    plt.ylabel("Service")
    plt.title(f"Top 10 Services by Billed Cost ({currency_choice})")
    plt.tight_layout()
    plt.savefig(RESULTS / "figure_01_top_services.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(curr_months["Month"], curr_months["BilledCost"], marker="o")
    plt.xlabel("Charge-period month")
    plt.ylabel(f"BilledCost ({currency_choice})")
    plt.title(f"Monthly Billed Cost ({currency_choice})")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(RESULTS / "figure_02_monthly_cost.png", dpi=180)
    plt.close()

    plt.figure(figsize=(10, 6))
    plt.barh(curr_categories["ChargeCategory"], curr_categories["BilledCost"])
    plt.xlabel(f"BilledCost ({currency_choice})")
    plt.ylabel("Charge category")
    plt.title(f"Billed Cost by Charge Category ({currency_choice})")
    plt.tight_layout()
    plt.savefig(RESULTS / "figure_03_charge_categories.png", dpi=180)
    plt.close()

    print("\nCharts generated using currency:", currency_choice)
    print("figure_01_top_services.png")
    print("figure_02_monthly_cost.png")
    print("figure_03_charge_categories.png")
except ImportError:
    print("\nmatplotlib is not installed; summary tables were still generated.")
    print("Install it with: pip install matplotlib")
