
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "cleaned_cloud_billing_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "research_figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and validate the billing dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=None, engine="python")

required_columns = ["Usage_Start_Date", "Total_Cost_INR"]
missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}\n"
        f"Available columns: {list(df.columns)}"
    )

df["Usage_Start_Date"] = pd.to_datetime(
    df["Usage_Start_Date"],
    errors="coerce"
)

# Handle numeric values even if they contain currency symbols or commas.
df["Total_Cost_INR"] = pd.to_numeric(
    df["Total_Cost_INR"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("₹", "", regex=False)
    .str.replace("$", "", regex=False)
    .str.strip(),
    errors="coerce"
)

# Exclude records that cannot be used for this analysis.
df = df.dropna(
    subset=["Usage_Start_Date", "Total_Cost_INR"]
).copy()

if df.empty:
    raise ValueError("No valid dates and costs remain after cleaning.")

if (df["Total_Cost_INR"] < 0).any():
    print(
        "Warning: Negative costs exist. "
        "Check whether they represent credits or billing adjustments."
    )


# --------------------------------------------------
# 3. Aggregate expenditure by calendar day
# --------------------------------------------------

df["Billing_Date"] = df["Usage_Start_Date"].dt.normalize()

daily_cost = (
    df.groupby("Billing_Date")["Total_Cost_INR"]
    .sum()
    .sort_index()
)

# Fill missing calendar dates with zero expenditure.
# This assumes the dataset represents all billing records
# for the observation period.
full_date_range = pd.date_range(
    start=daily_cost.index.min(),
    end=daily_cost.index.max(),
    freq="D"
)

daily_cost = daily_cost.reindex(
    full_date_range,
    fill_value=0
)

daily_cost.index.name = "Billing_Date"

# Smooth short-term fluctuations using a 7-calendar-day average.
rolling_cost = daily_cost.rolling(
    window=7,
    min_periods=1
).mean()

# Running expenditure across the observation period.
cumulative_cost = daily_cost.cumsum()


# --------------------------------------------------
# 4. Create the research figure
# --------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 15
})

fig, axes = plt.subplots(
    nrows=2,
    ncols=1,
    figsize=(12, 8),
    sharex=True,
    constrained_layout=True
)

# Panel A: Daily spending and smoothed trend
axes[0].plot(
    daily_cost.index,
    daily_cost.values,
    label="Daily expenditure",
    color="#7A91AC",
    linewidth=1.2,
    alpha=0.75
)

axes[0].plot(
    rolling_cost.index,
    rolling_cost.values,
    label="7-day rolling average",
    color="#B14A4A",
    linewidth=2.2
)

axes[0].set_title(
    "(a) Daily cloud expenditure and rolling average",
    loc="left",
    fontweight="bold"
)

axes[0].set_ylabel("Expenditure (INR)")
axes[0].legend(frameon=False, loc="upper left")
axes[0].grid(True, alpha=0.25)


# Panel B: Cumulative spending
axes[1].plot(
    cumulative_cost.index,
    cumulative_cost.values,
    color="#357A69",
    linewidth=2.2
)

axes[1].fill_between(
    cumulative_cost.index,
    cumulative_cost.values,
    alpha=0.12,
    color="#357A69"
)

axes[1].set_title(
    "(b) Cumulative cloud expenditure",
    loc="left",
    fontweight="bold"
)

axes[1].set_ylabel("Cumulative expenditure (INR)")
axes[1].set_xlabel("Billing date")
axes[1].grid(True, alpha=0.25)

# Display dates clearly without overcrowding the x-axis.
axes[1].xaxis.set_major_locator(
    mdates.WeekdayLocator(byweekday=mdates.MO, interval=1)
)

axes[1].xaxis.set_major_formatter(
    mdates.DateFormatter("%d %b")
)

plt.setp(
    axes[1].get_xticklabels(),
    rotation=35,
    ha="right"
)

fig.suptitle(
    "Historical Cloud Expenditure Analysis",
    fontweight="bold"
)


# --------------------------------------------------
# 5. Export publication-quality outputs
# --------------------------------------------------

png_path = OUTPUT_DIR / "figure_01_spending_over_time.png"
pdf_path = OUTPUT_DIR / "figure_01_spending_over_time.pdf"
svg_path = OUTPUT_DIR / "figure_01_spending_over_time.svg"

fig.savefig(
    png_path,
    dpi=300,
    bbox_inches="tight"
)

fig.savefig(
    pdf_path,
    bbox_inches="tight"
)

fig.savefig(
    svg_path,
    bbox_inches="tight"
)

plt.close(fig)


# --------------------------------------------------
# 6. Print a data summary for verification
# --------------------------------------------------

print("\nFIGURE 1 GENERATED SUCCESSFULLY")
print("-" * 45)
print(f"Valid billing records: {len(df):,}")
print(f"Observation start:     {daily_cost.index.min().date()}")
print(f"Observation end:       {daily_cost.index.max().date()}")
print(f"Calendar days:         {len(daily_cost):,}")
print(f"Total expenditure:     INR {daily_cost.sum():,.2f}")
print(f"Average daily cost:    INR {daily_cost.mean():,.2f}")
print(f"Peak daily cost:       INR {daily_cost.max():,.2f}")
print(
    f"Peak spending date:    "
    f"{daily_cost.idxmax().date()}"
)

print("\nSaved files:")
print(png_path)
print(pdf_path)
print(svg_path)
