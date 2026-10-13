from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

root = Path(__file__).resolve().parents[1]
source = root / "data/canonical_run/forecasting_artifacts/future_30_day_forecast.csv"
out = root / "outputs/research_figures"
out.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(source)
df["Forecast_Date"] = pd.to_datetime(df["Forecast_Date"], errors="raise")
df["Predicted_Cost_INR"] = pd.to_numeric(df["Predicted_Cost_INR"], errors="raise")
df = df.sort_values("Forecast_Date")

if len(df) != 30 or df["Predicted_Cost_INR"].isna().any():
    raise ValueError("Expected 30 valid forecast records.")

fig, ax = plt.subplots(figsize=(11, 5.8))
dates = df["Forecast_Date"]
costs = df["Predicted_Cost_INR"] / 100000

ax.plot(dates, costs, marker="o", markersize=3.5, linewidth=2,
        label="Forecast expenditure")
ax.fill_between(dates, costs, alpha=0.12)

ax.set_title("Projected Daily Cloud Expenditure: 30-Day Forecast", fontsize=15, pad=14)
ax.set_xlabel("Forecast date")
ax.set_ylabel("Predicted expenditure (₹ lakh)")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.45)
ax.legend(frameon=False)
ax.tick_params(axis="x", rotation=35)
fig.tight_layout()

for ext in ("png", "pdf", "svg"):
    path = out / f"figure_06_future_cost_forecast.{ext}"
    fig.savefig(path, dpi=300 if ext == "png" else None, bbox_inches="tight")
    print(f"Saved: {path}")

print(f"Forecast period: {dates.min():%Y-%m-%d} to {dates.max():%Y-%m-%d}")
print(f"Observations: {len(df)}")
print(f"Total projected expenditure: ₹{df['Predicted_Cost_INR'].sum():,.2f}")
print(f"Average per day: ₹{df['Predicted_Cost_INR'].mean():,.2f}")
print("These are model projections, not guaranteed future costs.")

plt.close(fig)
