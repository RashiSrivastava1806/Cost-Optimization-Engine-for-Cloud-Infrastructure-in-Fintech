
from pathlib import Path

import pandas as pd
import plotly.express as px


# --------------------------------------------------
# 1. Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = PROJECT_ROOT / "data" / "cleaned_cloud_billing_data.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "research_figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# 2. Load and validate dataset
# --------------------------------------------------

if not DATA_PATH.exists():
    raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH, sep=None, engine="python")

required_columns = [
    "Resource_ID",
    "Service_Name",
    "CPU_Utilization",
    "Memory_Utilization",
    "Total_Cost_INR",
    "Usage_Start_Date",
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"Missing columns: {missing}\n"
        f"Available columns: {list(df.columns)}"
    )

# Convert numeric fields safely.
numeric_columns = [
    "CPU_Utilization",
    "Memory_Utilization",
    "Total_Cost_INR",
]

for col in numeric_columns:
    df[col] = pd.to_numeric(
        df[col]
        .astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("₹", "", regex=False)
        .str.replace("$", "", regex=False)
        .str.strip(),
        errors="coerce",
    )

df["Usage_Start_Date"] = pd.to_datetime(
    df["Usage_Start_Date"],
    errors="coerce",
)

df = df.dropna(
    subset=required_columns
).copy()

# Utilization is expected to be expressed as a percentage.
valid_utilization = (
    df["CPU_Utilization"].between(0, 100)
    & df["Memory_Utilization"].between(0, 100)
)

df = df.loc[valid_utilization].copy()

if df.empty:
    raise ValueError("No valid records remain after data validation.")

# These are screening thresholds, not SLA limits.
LOW_UTILIZATION_THRESHOLD = 30

df["CPU_Status"] = df["CPU_Utilization"].apply(
    lambda value: (
        "Low CPU utilization"
        if value < LOW_UTILIZATION_THRESHOLD
        else "CPU utilization ≥ 30%"
    )
)

df["Memory_Status"] = df["Memory_Utilization"].apply(
    lambda value: (
        "Low memory utilization"
        if value < LOW_UTILIZATION_THRESHOLD
        else "Memory utilization ≥ 30%"
    )
)

# Format date for tooltip display.
df["Billing_Date"] = df["Usage_Start_Date"].dt.strftime("%Y-%m-%d")


# --------------------------------------------------
# 3. Interactive graph: Cost vs CPU utilization
# --------------------------------------------------

cpu_fig = px.scatter(
    df,
    x="CPU_Utilization",
    y="Total_Cost_INR",
    color="CPU_Status",
    hover_name="Resource_ID",
    hover_data={
        "Service_Name": True,
        "Billing_Date": True,
        "CPU_Utilization": ":.2f",
        "Memory_Utilization": ":.2f",
        "Total_Cost_INR": ":,.2f",
        "CPU_Status": False,
    },
    color_discrete_map={
        "Low CPU utilization": "#D95F59",
        "CPU utilization ≥ 30%": "#477DB3",
    },
    labels={
        "CPU_Utilization": "CPU Utilization (%)",
        "Total_Cost_INR": "Expenditure (INR)",
        "CPU_Status": "CPU screening",
        "Service_Name": "Cloud Service",
        "Billing_Date": "Billing Date",
        "Memory_Utilization": "Memory Utilization (%)",
    },
    title="Cloud Expenditure vs CPU Utilization",
    template="plotly_white",
)

cpu_fig.add_vline(
    x=LOW_UTILIZATION_THRESHOLD,
    line_dash="dash",
    line_color="gray",
    annotation_text="30% screening threshold",
)

cpu_fig.update_traces(
    marker={"size": 9, "opacity": 0.75}
)

cpu_fig.update_layout(
    height=650,
    hoverlabel={"namelength": -1},
    legend_title_text="CPU screening",
)

cpu_fig.update_xaxes(range=[0, 100])


# --------------------------------------------------
# 4. Interactive graph: Cost vs memory utilization
# --------------------------------------------------

memory_fig = px.scatter(
    df,
    x="Memory_Utilization",
    y="Total_Cost_INR",
    color="Memory_Status",
    hover_name="Resource_ID",
    hover_data={
        "Service_Name": True,
        "Billing_Date": True,
        "CPU_Utilization": ":.2f",
        "Memory_Utilization": ":.2f",
        "Total_Cost_INR": ":,.2f",
        "Memory_Status": False,
    },
    color_discrete_map={
        "Low memory utilization": "#D95F59",
        "Memory utilization ≥ 30%": "#477DB3",
    },
    labels={
        "Memory_Utilization": "Memory Utilization (%)",
        "Total_Cost_INR": "Expenditure (INR)",
        "Memory_Status": "Memory screening",
        "Service_Name": "Cloud Service",
        "Billing_Date": "Billing Date",
        "CPU_Utilization": "CPU Utilization (%)",
    },
    title="Cloud Expenditure vs Memory Utilization",
    template="plotly_white",
)

memory_fig.add_vline(
    x=LOW_UTILIZATION_THRESHOLD,
    line_dash="dash",
    line_color="gray",
    annotation_text="30% screening threshold",
)

memory_fig.update_traces(
    marker={"size": 9, "opacity": 0.75}
)

memory_fig.update_layout(
    height=650,
    hoverlabel={"namelength": -1},
    legend_title_text="Memory screening",
)

memory_fig.update_xaxes(range=[0, 100])


# --------------------------------------------------
# 5. Save interactive HTML files
# --------------------------------------------------

cpu_path = OUTPUT_DIR / "figure_03a_interactive_cpu_cost.html"
memory_path = OUTPUT_DIR / "figure_03b_interactive_memory_cost.html"

cpu_fig.write_html(
    str(cpu_path),
    include_plotlyjs=True,
)

memory_fig.write_html(
    str(memory_path),
    include_plotlyjs=True,
)


# --------------------------------------------------
# 6. Print verification summary
# --------------------------------------------------

print("\nINTERACTIVE FIGURE 3 GENERATED SUCCESSFULLY")
print("-" * 55)
print(f"Records plotted: {len(df):,}")
print(f"Unique resources: {df['Resource_ID'].nunique():,}")
print(f"Unique services: {df['Service_Name'].nunique():,}")

print(
    "Low CPU utilization records: "
    f"{(df['CPU_Utilization'] < LOW_UTILIZATION_THRESHOLD).sum():,}"
)

print(
    "Low memory utilization records: "
    f"{(df['Memory_Utilization'] < LOW_UTILIZATION_THRESHOLD).sum():,}"
)

print("\nInteractive files:")
print(cpu_path)
print(memory_path)

print(
    "\nOpen either HTML file in a browser to hover over points, "
    "zoom, pan, and inspect resource details."
)
