
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
    "CPU_Utilization",
    "Memory_Utilization",
    "Total_Cost_INR",
    "Resource_ID"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

for column in ["CPU_Utilization", "Memory_Utilization", "Total_Cost_INR"]:
    df[column] = pd.to_numeric(
        df[column]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce"
    )

# Exclude missing and impossible utilization values.
df = df.dropna(
    subset=["CPU_Utilization", "Memory_Utilization", "Total_Cost_INR"]
).copy()

valid_utilization = (
    df["CPU_Utilization"].between(0, 100)
    & df["Memory_Utilization"].between(0, 100)
)

invalid_count = int((~valid_utilization).sum())
df = df.loc[valid_utilization].copy()

if df.empty:
    raise ValueError("No valid utilization and cost records remain.")


# --------------------------------------------------
# 3. Identify low-utilization records for investigation
# --------------------------------------------------

LOW_UTILIZATION_THRESHOLD = 30

df["Low_CPU_Utilization"] = (
    df["CPU_Utilization"] < LOW_UTILIZATION_THRESHOLD
)

df["Low_Memory_Utilization"] = (
    df["Memory_Utilization"] < LOW_UTILIZATION_THRESHOLD
)

df["Low_CPU_and_Memory"] = (
    df["Low_CPU_Utilization"]
    & df["Low_Memory_Utilization"]
)


# --------------------------------------------------
# 4. Prepare the figure
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
    figsize=(14, 6),
    constrained_layout=True
)

currency_formatter = FuncFormatter(
    lambda value, _: f"₹{value:,.0f}"
)

# Use the same color convention in both panels:
# red = low utilization; blue = other valid records.
colors = df["Low_CPU_Utilization"].map(
    {True: "#B85C5C", False: "#547FA3"}
)

# Panel A: Cost vs CPU utilization
axes[0].scatter(
    df["CPU_Utilization"],
    df["Total_Cost_INR"],
    c=colors,
    alpha=0.65,
    s=30,
    edgecolors="none"
)

axes[0].axvline(
    LOW_UTILIZATION_THRESHOLD,
    color="#555555",
    linestyle="--",
    linewidth=1.2,
    label="30% investigation threshold"
)

axes[0].set_title(
    "(a) Expenditure vs. CPU utilization",
    loc="left",
    fontweight="bold"
)

axes[0].set_xlabel("CPU utilization (%)")
axes[0].set_ylabel("Expenditure (INR)")
axes[0].yaxis.set_major_formatter(currency_formatter)
axes[0].set_xlim(0, 100)
axes[0].grid(alpha=0.2)
axes[0].legend(frameon=False, fontsize=8)


# Panel B: Cost vs memory utilization
memory_colors = df["Low_Memory_Utilization"].map(
    {True: "#B85C5C", False: "#547FA3"}
)

axes[1].scatter(
    df["Memory_Utilization"],
    df["Total_Cost_INR"],
    c=memory_colors,
    alpha=0.65,
    s=30,
    edgecolors="none"
)

axes[1].axvline(
    LOW_UTILIZATION_THRESHOLD,
    color="#555555",
    linestyle="--",
    linewidth=1.2,
    label="30% investigation threshold"
)

axes[1].set_title(
    "(b) Expenditure vs. memory utilization",
    loc="left",
    fontweight="bold"
)

axes[1].set_xlabel("Memory utilization (%)")
axes[1].set_ylabel("Expenditure (INR)")
axes[1].yaxis.set_major_formatter(currency_formatter)
axes[1].set_xlim(0, 100)
axes[1].grid(alpha=0.2)
axes[1].legend(frameon=False, fontsize=8)

fig.suptitle(
    "Cloud Cost and Resource Utilization",
    fontweight="bold"
)


# --------------------------------------------------
# 5. Export research figures
# --------------------------------------------------

png_path = OUTPUT_DIR / "figure_03_cost_vs_utilization.png"
pdf_path = OUTPUT_DIR / "figure_03_cost_vs_utilization.pdf"
svg_path = OUTPUT_DIR / "figure_03_cost_vs_utilization.svg"

fig.savefig(png_path, dpi=300, bbox_inches="tight")
fig.savefig(pdf_path, bbox_inches="tight")
fig.savefig(svg_path, bbox_inches="tight")

plt.close(fig)


# --------------------------------------------------
# 6. Print research statistics
# --------------------------------------------------

low_cpu = df["Low_CPU_Utilization"]
low_memory = df["Low_Memory_Utilization"]
low_both = df["Low_CPU_and_Memory"]

print("\nFIGURE 3 GENERATED SUCCESSFULLY")
print("-" * 50)
print(f"Valid records: {len(df):,}")
print(f"Records excluded for invalid utilization: {invalid_count:,}")
print(f"Low CPU utilization (<30%): {low_cpu.sum():,} ({low_cpu.mean() * 100:.2f}%)")
print(
    f"Low memory utilization (<30%): "
    f"{low_memory.sum():,} ({low_memory.mean() * 100:.2f}%)"
)
print(
    f"Low CPU AND memory utilization: "
    f"{low_both.sum():,} ({low_both.mean() * 100:.2f}%)"
)

if low_both.any():
    low_both_cost = df.loc[low_both, "Total_Cost_INR"].sum()
    all_cost = df["Total_Cost_INR"].sum()

    print(
        f"Cost associated with records low in both: "
        f"INR {low_both_cost:,.2f}"
    )
    print(
        f"Share of valid-record expenditure: "
        f"{low_both_cost / all_cost * 100:.2f}%"
    )

print("\nSaved files:")
print(png_path)
print(pdf_path)
print(svg_path)
