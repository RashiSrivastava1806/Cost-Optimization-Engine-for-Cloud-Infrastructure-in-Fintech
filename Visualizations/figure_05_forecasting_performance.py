from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = (
    ROOT / "data" / "canonical_run" /
    "forecasting_artifacts" / "test_predictions.csv"
)
OUTPUT_DIR = ROOT / "outputs" / "research_figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_FILE)

required = {
    "Date", "Actual_Cost_INR", "Predicted_Cost_INR"
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {sorted(missing)}")

df["Date"] = pd.to_datetime(
    df["Date"], format="%Y-%m-%d", errors="raise"
)
df["Actual_Cost_INR"] = pd.to_numeric(
    df["Actual_Cost_INR"], errors="raise"
)
df["Predicted_Cost_INR"] = pd.to_numeric(
    df["Predicted_Cost_INR"], errors="raise"
)
df = df.sort_values("Date")

if df.empty:
    raise ValueError("The test predictions file is empty.")

fig, ax = plt.subplots(figsize=(11, 5.8))

ax.plot(
    df["Date"],
    df["Actual_Cost_INR"] / 100000,
    marker="o",
    markersize=4,
    linewidth=2,
    label="Actual expenditure",
)
ax.plot(
    df["Date"],
    df["Predicted_Cost_INR"] / 100000,
    marker="s",
    markersize=4,
    linewidth=2,
    linestyle="--",
    label="Predicted expenditure",
)

ax.set_title(
    "Actual vs. Predicted Daily Cloud Expenditure",
    fontsize=15,
    pad=14,
)
ax.set_xlabel("Billing date", fontsize=11)
ax.set_ylabel("Daily expenditure (₹ lakh)", fontsize=11)
ax.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
ax.legend(frameon=False)
ax.tick_params(axis="x", rotation=35)

fig.tight_layout()

for extension in ("png", "pdf", "svg"):
    path = OUTPUT_DIR / f"figure_05_forecasting_performance.{extension}"
    if extension == "png":
        fig.savefig(path, dpi=300, bbox_inches="tight")
    else:
        fig.savefig(path, bbox_inches="tight")
    print(f"Saved: {path}")

mae = (df["Actual_Cost_INR"] - df["Predicted_Cost_INR"]).abs().mean()
print(f"\nTest dates: {df['Date'].min():%Y-%m-%d} to {df['Date'].max():%Y-%m-%d}")
print(f"Test observations: {len(df)}")
print(f"Mean absolute error: ₹{mae:,.2f}")
print("Note: This figure uses held-out test predictions, not future forecasts.")

plt.close(fig)
