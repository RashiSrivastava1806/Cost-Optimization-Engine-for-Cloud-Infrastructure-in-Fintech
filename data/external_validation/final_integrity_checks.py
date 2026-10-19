import hashlib
import io
import json
import zipfile
from pathlib import Path

import pandas as pd

archive = Path("data/external_validation/raw/dataset-examples.zip")
target = "EA-Cost-FOCUS_1.0.csv"
out = Path("data/external_validation/results")
out.mkdir(parents=True, exist_ok=True)

seen_hashes = set()
exact_duplicate_rows = 0
rows = 0
cost_columns = [
    "BilledCost",
    "EffectiveCost",
    "ListCost",
    "ContractedCost",
]

cost_totals = {col: 0.0 for col in cost_columns}
cost_invalid = {col: 0 for col in cost_columns}
cost_negative = {col: 0 for col in cost_columns}
currency_counts = {}
currency_totals = {}

with zipfile.ZipFile(archive) as z:
    with z.open(target) as raw:
        stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")

        for chunk in pd.read_csv(
            stream,
            chunksize=25_000,
            low_memory=False,
        ):
            rows += len(chunk)

            # Stable row hashes allow exact duplicate detection across chunks.
            row_hashes = pd.util.hash_pandas_object(
                chunk.astype("string").fillna("<NA>"),
                index=False,
            ).tolist()

            for value in row_hashes:
                key = int(value)
                if key in seen_hashes:
                    exact_duplicate_rows += 1
                else:
                    seen_hashes.add(key)

            currencies = (
                chunk["BillingCurrency"].astype("string").str.strip()
                .replace("", pd.NA).fillna("(missing)")
            )
            for currency, count in currencies.value_counts().items():
                currency_counts[str(currency)] = (
                    currency_counts.get(str(currency), 0) + int(count)
                )

            for col in cost_columns:
                if col not in chunk.columns:
                    continue

                values = pd.to_numeric(chunk[col], errors="coerce")
                cost_invalid[col] += int(values.isna().sum())
                cost_negative[col] += int(values.lt(0).fillna(False).sum())
                cost_totals[col] += float(values.sum())

                frame = pd.DataFrame({
                    "Currency": currencies,
                    "Cost": values,
                })
                grouped = frame.groupby("Currency")["Cost"].sum()
                for currency, total in grouped.items():
                    key = (col, str(currency))
                    currency_totals[key] = (
                        currency_totals.get(key, 0.0) + float(total)
                    )

summary = {
    "dataset": target,
    "rows_processed": rows,
    "exact_duplicate_rows": exact_duplicate_rows,
    "duplicate_check_method": (
        "64-bit pandas row hashes; theoretical hash collisions are possible"
    ),
    "currency_record_counts": currency_counts,
    "cost_measure_totals": cost_totals,
    "invalid_values_by_cost_measure": cost_invalid,
    "negative_values_by_cost_measure": cost_negative,
    "cost_totals_by_currency": {
        f"{measure}|{currency}": value
        for (measure, currency), value in currency_totals.items()
    },
}

with open(out / "final_integrity_summary.json", "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)

measure_table = pd.DataFrame([
    {
        "CostMeasure": col,
        "Sum": cost_totals[col],
        "InvalidRows": cost_invalid[col],
        "NegativeRows": cost_negative[col],
    }
    for col in cost_columns
    if col in cost_totals
])
measure_table.to_csv(out / "cost_measure_comparison.csv", index=False)

print("=== FINAL INTEGRITY CHECKS ===")
print("Rows processed:", f"{rows:,}")
print("Exact duplicate rows detected:", exact_duplicate_rows)
print("Currency record counts:", currency_counts)
print("\n=== COST MEASURE COMPARISON ===")
print(measure_table.to_string(index=False))
print("\n=== COST TOTALS BY CURRENCY ===")
for key, value in sorted(currency_totals.items()):
    print(f"{key[0]} / {key[1]}: {value:.6f}")
print("\nSaved final_integrity_summary.json")
print("Saved cost_measure_comparison.csv")
print("No source data or canonical outputs modified.")
