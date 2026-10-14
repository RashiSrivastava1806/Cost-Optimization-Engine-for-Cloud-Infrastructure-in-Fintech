from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    ROOT / "data" / "canonical_run" /
    "forecasting_artifacts" / "test_predictions.csv"
)

OUTPUT_DIR = ROOT / "outputs" / "research_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load the existing held-out test predictions.
df = pd.read_csv(DATA_FILE)

required = {"Date", "Actual_Cost_INR", "Predicted_Cost_INR"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["Date"] = pd.to_datetime(df["Date"], errors="raise")
df["Actual_Cost_INR"] = pd.to_numeric(
    df["Actual_Cost_INR"], errors="raise"
)
df["Predicted_Cost_INR"] = pd.to_numeric(
    df["Predicted_Cost_INR"], errors="raise"
)

df = df.dropna(
    subset=["Date", "Actual_Cost_INR", "Predicted_Cost_INR"]
).sort_values("Date")

if df.empty:
    raise ValueError("No valid test predictions available.")

# Positive residual = underprediction.
# Negative residual = overprediction.
df["Residual_INR"] = (
    df["Actual_Cost_INR"] - df["Predicted_Cost_INR"]
)

fig, ax = plt.subplots(figsize=(10, 6))

ax.scatter(
    df["Predicted_Cost_INR"],
    df["Residual_INR"],
    s=65,
    alpha=0.8,
    edgecolors="black",
    linewidths=0.5,
)

# Zero-error reference line.
ax.axhline(
    0,
    linestyle="--",
    linewidth=1.5,
    label="Zero error",
)

ax.set_title(
    "Forecast Residuals on the Held-Out Test Set",
    fontsize=14,
    pad=12,
)
ax.set_xlabel("Predicted daily expenditure (INR)", fontsize=11)
ax.set_ylabel("Residual: actual − predicted (INR)", fontsize=11)

ax.xaxis.set_major_formatter(
    FuncFormatter(lambda value, _: f"₹{value / 100000:.0f}L")
)
ax.yaxis.set_major_formatter(
    FuncFormatter(lambda value, _: f"₹{value / 100000:.0f}L")
)

ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
ax.legend(frameon=False)
fig.tight_layout()

for ext in ("png", "pdf", "svg"):
    output_file = OUTPUT_DIR / f"figure_07_forecast_residuals.{ext}"

    if ext == "png":
        fig.savefig(output_file, dpi=300, bbox_inches="tight")
    else:
        fig.savefig(output_file, bbox_inches="tight")

    print(f"Saved: {output_file}")

print("\nResidual summary")
print(f"Test observations: {len(df)}")
print(f"Mean residual: ₹{df['Residual_INR'].mean():,.2f}")
print(f"Median residual: ₹{df['Residual_INR'].median():,.2f}")
print(f"Underpredictions: {(df['Residual_INR'] > 0).sum()}")
print(f"Overpredictions: {(df['Residual_INR'] < 0).sum()}")
print(f"Exact predictions: {(df['Residual_INR'] == 0).sum()}")
print(f"Residual standard deviation: ₹{df['Residual_INR'].std():,.2f}")

plt.close(fig)
