from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "data" / "canonical_run" / "forecasting_artifacts"
TRAIN_FILE = ROOT / "data" / "canonical_run" / "engineered_train_cloud_billing_data.csv"
TEST_FILE = ROOT / "data" / "canonical_run" / "engineered_test_cloud_billing_data.csv"
PREDICTIONS_FILE = ARTIFACTS / "test_predictions.csv"

OUTPUT_DIR = ROOT / "outputs" / "research_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_FILE = ARTIFACTS / "baseline_comparison_results.csv"
PREDICTIONS_OUT = ARTIFACTS / "baseline_comparison_predictions.csv"


def daily_cost_from_resource_data(path):
    df = pd.read_csv(path)

    required = {"Usage_Start_Date", "Total_Cost_INR"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} missing columns: {sorted(missing)}")

    df["Usage_Start_Date"] = pd.to_datetime(
        df["Usage_Start_Date"], errors="raise"
    )
    df["Total_Cost_INR"] = pd.to_numeric(
        df["Total_Cost_INR"], errors="raise"
    )

    df["Date"] = df["Usage_Start_Date"].dt.normalize()
    return (
        df.groupby("Date", as_index=False)["Total_Cost_INR"]
        .sum()
        .rename(columns={"Total_Cost_INR": "Actual_Cost_INR"})
        .sort_values("Date")
    )


train = daily_cost_from_resource_data(TRAIN_FILE)
test = daily_cost_from_resource_data(TEST_FILE)
saved = pd.read_csv(PREDICTIONS_FILE)

saved["Date"] = pd.to_datetime(saved["Date"], errors="raise").dt.normalize()
saved["Actual_Cost_INR"] = pd.to_numeric(
    saved["Actual_Cost_INR"], errors="raise"
)
saved["Predicted_Cost_INR"] = pd.to_numeric(
    saved["Predicted_Cost_INR"], errors="raise"
)

train = train.sort_values("Date").reset_index(drop=True)
test = test.sort_values("Date").reset_index(drop=True)
saved = saved.sort_values("Date").reset_index(drop=True)

# Verify saved actuals agree with the corrected resource-level test data.
check = test.merge(
    saved[["Date", "Actual_Cost_INR"]],
    on="Date",
    how="outer",
    suffixes=("_resource_data", "_saved"),
    indicator=True,
)

if not (check["_merge"] == "both").all():
    raise ValueError("Saved predictions and test data have different dates.")

if not np.allclose(
    check["Actual_Cost_INR_resource_data"],
    check["Actual_Cost_INR_saved"],
    rtol=1e-8,
    atol=0.01,
):
    raise ValueError(
        "Saved test actuals do not match the corrected resource-level test data."
    )

# Confirm daily aggregation and saved model predictions cover identical dates.
if not test["Date"].reset_index(drop=True).equals(
    saved["Date"].reset_index(drop=True)
):
    raise ValueError("Test dates are not aligned across input files.")

if train.empty or test.empty:
    raise ValueError("Training or test data is empty.")

# Baseline 1: previous observed day's actual expenditure.
# For the first test day, use the last training day's actual expenditure.
combined_actuals = pd.concat(
    [
        train[["Date", "Actual_Cost_INR"]],
        test[["Date", "Actual_Cost_INR"]],
    ],
    ignore_index=True,
).sort_values("Date").reset_index(drop=True)

combined_actuals["Naive_Prediction_INR"] = (
    combined_actuals["Actual_Cost_INR"].shift(1)
)

test_dates = set(test["Date"])
naive = combined_actuals[
    combined_actuals["Date"].isin(test_dates)
][["Date", "Naive_Prediction_INR"]]

# Baseline 2: fixed mean of training-period daily expenditure.
historical_mean = train["Actual_Cost_INR"].mean()

results_df = saved[
    ["Date", "Actual_Cost_INR", "Predicted_Cost_INR"]
].merge(naive, on="Date", how="left")

results_df["Historical_Average_Prediction_INR"] = historical_mean

if results_df["Naive_Prediction_INR"].isna().any():
    raise ValueError("Naïve baseline contains missing predictions.")

model_columns = {
    "Naïve previous-day baseline": "Naive_Prediction_INR",
    "Training-period historical average": "Historical_Average_Prediction_INR",
    "Linear trend with weekday effects": "Predicted_Cost_INR",
}

metric_rows = []

for model_name, prediction_col in model_columns.items():
    actual = results_df["Actual_Cost_INR"].to_numpy(dtype=float)
    predicted = results_df[prediction_col].to_numpy(dtype=float)
    errors = actual - predicted

    metric_rows.append(
        {
            "Model": model_name,
            "Test_Observations": len(errors),
            "MAE_INR": float(np.mean(np.abs(errors))),
            "RMSE_INR": float(np.sqrt(np.mean(errors ** 2))),
            "WAPE_Percent": float(
                np.sum(np.abs(errors)) / np.sum(np.abs(actual)) * 100
            ),
        }
    )

metrics = pd.DataFrame(metric_rows)
metrics.to_csv(RESULTS_FILE, index=False)
results_df.to_csv(PREDICTIONS_OUT, index=False)

print("\nBaseline comparison (lower is better)")
print(metrics.to_string(index=False, formatters={
    "MAE_INR": lambda x: f"₹{x:,.2f}",
    "RMSE_INR": lambda x: f"₹{x:,.2f}",
    "WAPE_Percent": lambda x: f"{x:.2f}%",
}))

print(f"\nSaved metrics: {RESULTS_FILE}")
print(f"Saved aligned predictions: {PREDICTIONS_OUT}")

# Grouped bars, with separate panels for MAE and RMSE.
fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
labels = ["Naïve\nbaseline", "Historical\naverage", "Linear trend +\nweekday effects"]
x = np.arange(len(metrics))

for ax, metric_col, title in [
    (axes[0], "MAE_INR", "Mean Absolute Error (MAE)"),
    (axes[1], "RMSE_INR", "Root Mean Squared Error (RMSE)"),
]:
    values = metrics[metric_col].to_numpy() / 100000
    bars = ax.bar(x, values, width=0.65)

    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Error (₹ lakh)")
    ax.set_xticks(x, labels)
    ax.grid(axis="y", linestyle="--", alpha=0.35)

    for bar, value in zip(bars, values):
        ax.annotate(
            f"{value:.2f}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

fig.suptitle(
    "Forecasting Model Comparison on the Held-Out Test Period",
    fontsize=14,
)
fig.tight_layout()

for ext in ("png", "pdf", "svg"):
    path = OUTPUT_DIR / f"figure_08_forecasting_model_comparison.{ext}"
    if ext == "png":
        fig.savefig(path, dpi=300, bbox_inches="tight")
    else:
        fig.savefig(path, bbox_inches="tight")
    print(f"Saved figure: {path}")

plt.close(fig)
