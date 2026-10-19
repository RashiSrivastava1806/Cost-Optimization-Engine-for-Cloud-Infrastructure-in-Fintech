import io
import zipfile
from pathlib import Path

import pandas as pd

archive = Path("data/external_validation/raw/dataset-examples.zip")
target = "EA-Cost-FOCUS_1.0.csv"
out = Path("data/external_validation/results")
out.mkdir(parents=True, exist_ok=True)

category_parts = []
class_parts = []
description_parts = []

negative_count = 0
positive_count = 0
zero_count = 0
negative_sum = 0.0
positive_sum = 0.0

with zipfile.ZipFile(archive) as z:
    with z.open(target) as raw:
        stream = io.TextIOWrapper(raw, encoding="utf-8-sig", newline="")

        for chunk in pd.read_csv(
            stream,
            chunksize=100_000,
            low_memory=False,
        ):
            cost = pd.to_numeric(chunk["BilledCost"], errors="coerce")
            negative = cost.lt(0).fillna(False)
            positive = cost.gt(0).fillna(False)
            zero = cost.eq(0).fillna(False)

            negative_count += int(negative.sum())
            positive_count += int(positive.sum())
            zero_count += int(zero.sum())
            negative_sum += float(cost[negative].sum())
            positive_sum += float(cost[positive].sum())

            for field, target_list in [
                ("ChargeCategory", category_parts),
                ("ChargeClass", class_parts),
                ("ChargeDescription", description_parts),
            ]:
                labels = (
                    chunk[field].astype("string").str.strip()
                    .replace("", pd.NA).fillna("(blank/missing)")
                )

                temp = pd.DataFrame({
                    "Label": labels,
                    "Cost": cost,
                    "Negative": negative.astype(int),
                })

                grouped = temp.groupby("Label", dropna=False).agg(
                    Records=("Cost", "size"),
                    NegativeRows=("Negative", "sum"),
                    NetBilledCost=("Cost", "sum"),
                ).reset_index()

                target_list.append(grouped)

def combine(parts):
    data = pd.concat(parts, ignore_index=True)
    return data.groupby("Label", as_index=False).agg(
        Records=("Records", "sum"),
        NegativeRows=("NegativeRows", "sum"),
        NetBilledCost=("NetBilledCost", "sum"),
    )

category = combine(category_parts)
charge_class = combine(class_parts)
description = combine(description_parts)

category["NegativeRowPercent"] = (
    100 * category["NegativeRows"] / category["Records"]
)
charge_class["NegativeRowPercent"] = (
    100 * charge_class["NegativeRows"] / charge_class["Records"]
)
description["NegativeRowPercent"] = (
    100 * description["NegativeRows"] / description["Records"]
)

category = category.sort_values("NegativeRows", ascending=False)
charge_class = charge_class.sort_values("NegativeRows", ascending=False)

# Focus on descriptions that actually contain negative cost entries.
description = description[description["NegativeRows"] > 0].sort_values(
    "NegativeRows", ascending=False
)

category.to_csv(out / "negative_cost_by_category.csv", index=False)
charge_class.to_csv(out / "negative_cost_by_charge_class.csv", index=False)
description.head(100).to_csv(
    out / "negative_cost_by_description_top100.csv", index=False
)

print("=== OVERALL COST SIGN ANALYSIS (USD) ===")
print("Positive-cost rows:", f"{positive_count:,}")
print("Negative-cost rows:", f"{negative_count:,}")
print("Zero-cost rows:", f"{zero_count:,}")
print("Sum of positive costs:", round(positive_sum, 6))
print("Sum of negative costs:", round(negative_sum, 6))
print("Net billed cost:", round(positive_sum + negative_sum, 6))
print("Absolute value of negative costs:", round(abs(negative_sum), 6))

print("\n=== NEGATIVE COSTS BY CHARGE CATEGORY ===")
print(category.to_string(index=False))

print("\n=== NEGATIVE COSTS BY CHARGE CLASS ===")
print(charge_class.to_string(index=False))

print("\n=== TOP 15 DESCRIPTIONS BY NEGATIVE-ROW COUNT ===")
print(description.head(15).to_string(index=False))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_data = category.nlargest(12, "NegativeRows").sort_values(
        "NegativeRows"
    )

    plt.figure(figsize=(10, 6))
    plt.barh(plot_data["Label"], plot_data["NegativeRows"])
    plt.xlabel("Records with negative BilledCost")
    plt.ylabel("Charge category")
    plt.title("Negative BilledCost Records by Charge Category")
    plt.tight_layout()
    plt.savefig(out / "figure_04_negative_cost_analysis.png", dpi=180)
    plt.close()

    print("\nGraph saved: figure_04_negative_cost_analysis.png")
except ImportError:
    print("\nmatplotlib unavailable; CSV summaries were still saved.")

print("\nSaved negative-cost analysis tables to:", out)
