
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "cleaned_cloud_billing_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "research_figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and validate data
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=None, engine="python")

required_columns = [
    "Service_Name",
    "Resource_ID",
    "Total_Cost_INR"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

df["Total_Cost_INR"] = pd.to_numeric(
    df["Total_Cost_INR"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .str.replace("₹", "", regex=False)
    .str.replace("$", "", regex=False)
    .str.strip(),
    errors="coerce"
)

df = df.dropna(subset=required_columns).copy()

if df.empty:
    raise ValueError("No valid billing records remain.")

# Aggregate records for each service and resource.
service_cost = (
    df.groupby("Service_Name")["Total_Cost_INR"]
    .sum()
    .sort_values(ascending=True)
)

resource_cost = (
    df.groupby("Resource_ID")["Total_Cost_INR"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .sort_values(ascending=True)
)

total_cost = df["Total_Cost_INR"].sum()

if total_cost <= 0:
    raise ValueError("Total expenditure must be positive for share analysis.")


# --------------------------------------------------
# 3. Create the figure
# --------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "figure.titlesize": 15
})

fig, axes = plt.subplots(
    1, 2,
    figsize=(15, max(6, 0.38 * max(len(service_cost), len(resource_cost)))),
    gridspec_kw={"width_ratios": [1, 1.15]},
    constrained_layout=True
)

currency_formatter = FuncFormatter(
    lambda value, _: f"₹{value:,.0f}"
)

# Panel A: Service-level expenditure
axes[0].barh(
    service_cost.index.astype(str),
    service_cost.values,
    color="#587DA5"
)

axes[0].set_title(
    "(a) Total expenditure by service",
    loc="left",
    fontweight="bold"
)

axes[0].set_xlabel("Total expenditure (INR)")
axes[0].set_ylabel("Cloud service")
axes[0].xaxis.set_major_formatter(currency_formatter)
axes[0].grid(axis="x", alpha=0.25)
axes[0].set_axisbelow(True)

# Panel B: Highest-cost resources
axes[1].barh(
    resource_cost.index.astype(str),
    resource_cost.values,
    color="#B66A54"
)

axes[1].set_title(
    "(b) Ten highest-cost resources",
    loc="left",
    fontweight="bold"
)

axes[1].set_xlabel("Total expenditure (INR)")
axes[1].set_ylabel("Resource ID")
axes[1].xaxis.set_major_formatter(currency_formatter)
axes[1].grid(axis="x", alpha=0.25)
axes[1].set_axisbelow(True)

fig.suptitle(
    "Cloud Expenditure Concentration",
    fontweight="bold"
)


# --------------------------------------------------
# 4. Save PNG, PDF, and SVG
# --------------------------------------------------

png_path = OUTPUT_DIR / "figure_02_cost_concentration.png"
pdf_path = OUTPUT_DIR / "figure_02_cost_concentration.pdf"
svg_path = OUTPUT_DIR / "figure_02_cost_concentration.svg"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")

plt.close(fig)


# --------------------------------------------------
# 5. Print useful research statistics
# --------------------------------------------------

service_ranked = service_cost.sort_values(ascending=False)
resource_ranked = resource_cost.sort_values(ascending=False)

top_service = service_ranked.index[0]
top_service_cost = service_ranked.iloc[0]

top_resource = resource_ranked.index[0]
top_resource_cost = resource_ranked.iloc[0]

top_10_resource_share = (
    resource_ranked.sum() / total_cost * 100
)

print("\nFIGURE 2 GENERATED SUCCESSFULLY")
print("-" * 50)
print(f"Valid billing records: {len(df):,}")
print(f"Unique services:       {df['Service_Name'].nunique():,}")
print(f"Unique resources:      {df['Resource_ID'].nunique():,}")
print(f"Total expenditure:     INR {total_cost:,.2f}")

print("\nHighest-cost service:")
print(f"{top_service}: INR {top_service_cost:,.2f}")
print(f"Share of total: {top_service_cost / total_cost * 100:.2f}%")

print("\nHighest-cost resource:")
print(f"{top_resource}: INR {top_resource_cost:,.2f}")
print(f"Share of total: {top_resource_cost / total_cost * 100:.2f}%")

print(
    f"\nTop 10 resources' share of total expenditure: "
    f"{top_10_resource_share:.2f}%"
)

print("\nSaved files:")
print(png_path)
print(pdf_path)
print(svg_path)
