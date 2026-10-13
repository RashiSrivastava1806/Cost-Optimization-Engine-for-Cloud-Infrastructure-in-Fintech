
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "research_figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_DIR / "anomaly_train_cloud_billing_data.csv"
TEST_PATH = DATA_DIR / "anomaly_test_cloud_billing_data.csv"


# --------------------------------------------------
# 2. Load anomaly detection results
# --------------------------------------------------

for path in [TRAIN_PATH, TEST_PATH]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df["Dataset_Split"] = "Training"
test_df["Dataset_Split"] = "Testing"

df = pd.concat([train_df, test_df], ignore_index=True)

required = ["Resource_ID", "Service_Name", "Total_Cost_INR"]
missing = [col for col in required if col not in df.columns]

if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["Total_Cost_INR"] = pd.to_numeric(
    df["Total_Cost_INR"], errors="coerce"
)

df = df.dropna(subset=["Total_Cost_INR"]).copy()


# --------------------------------------------------
# 3. Identify anomaly flags
# --------------------------------------------------

def to_boolean(series):
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "yes", "y"])
    )


if "Anomaly_Flag" in df.columns:
    df["Is_Anomaly"] = to_boolean(df["Anomaly_Flag"])
else:
    candidate_flags = [
        col for col in [
            "Contextual_Anomaly_Flag",
            "Statistical_Outlier_Flag",
        ]
        if col in df.columns
    ]

    if not candidate_flags:
        raise ValueError(
            "No overall or recognized component anomaly flags found. "
            "Inspect the anomaly CSV columns."
        )

    df["Is_Anomaly"] = False

    for col in candidate_flags:
        df["Is_Anomaly"] |= to_boolean(df[col])

df["Anomaly_Status"] = np.where(
    df["Is_Anomaly"],
    "Flagged anomaly",
    "Not flagged",
)


# --------------------------------------------------
# 4. Consistent publication style
# --------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Explicit colors distinguish detection groups.
COLORS = {
    "Flagged anomaly": "#C44E52",
    "Not flagged": "#4C78A8",
}


def save_figure(fig, filename):
    """Export the same figure as PNG, PDF, and SVG."""
    for extension in ["png", "pdf", "svg"]:
        path = OUTPUT_DIR / f"{filename}.{extension}"

        kwargs = {"bbox_inches": "tight"}

        if extension == "png":
            kwargs["dpi"] = 300

        fig.savefig(path, **kwargs)
        print(f"Saved: {path}")

    plt.close(fig)


# --------------------------------------------------
# 5. Figure 4(a): Cost distribution
# --------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 6))

statuses = ["Not flagged", "Flagged anomaly"]
box_data = [
    df.loc[
        df["Anomaly_Status"] == status, "Total_Cost_INR"
    ].values
    for status in statuses
]

valid_data = [
    (status, values)
    for status, values in zip(statuses, box_data)
    if len(values) > 0
]

positions = list(range(1, len(valid_data) + 1))

bp = ax.boxplot(
    [values for _, values in valid_data],
    positions=positions,
    widths=0.48,
    patch_artist=True,
    showfliers=False,
    medianprops={"color": "black", "linewidth": 1.5},
    whiskerprops={"linewidth": 1.1},
    capprops={"linewidth": 1.1},
)

for patch, (status, _) in zip(bp["boxes"], valid_data):
    patch.set_facecolor(COLORS[status])
    patch.set_alpha(0.75)

# Plot all individual observations with jitter for visibility.
rng = np.random.default_rng(42)

for position, (status, values) in zip(positions, valid_data):
    jitter = rng.uniform(-0.10, 0.10, size=len(values))

    ax.scatter(
        position + jitter,
        values,
        s=12,
        alpha=0.35,
        color=COLORS[status],
        edgecolors="none",
        rasterized=True,
    )

ax.set_xticks(positions)
ax.set_xticklabels([status for status, _ in valid_data])
ax.set_ylabel("Total expenditure (INR)")
ax.set_title(
    "Figure 4(a). Billing Cost Distribution by Anomaly Status",
    pad=14,
)
ax.grid(axis="y", linestyle="--", alpha=0.3)
ax.ticklabel_format(axis="y", style="plain")
fig.tight_layout()

save_figure(fig, "figure_04a_anomaly_cost_distribution")


# --------------------------------------------------
# 6. Figure 4(b): Cost per hour vs total cost
# --------------------------------------------------

if "Cost_per_Hour" in df.columns:
    plot_df = df.copy()
    plot_df["Cost_per_Hour"] = pd.to_numeric(
        plot_df["Cost_per_Hour"], errors="coerce"
    )
    plot_df = plot_df.dropna(
        subset=["Cost_per_Hour", "Total_Cost_INR"]
    )

    if not plot_df.empty:
        fig, ax = plt.subplots(figsize=(9, 6))

        for status in ["Not flagged", "Flagged anomaly"]:
            subset = plot_df[
                plot_df["Anomaly_Status"] == status
            ]

            if subset.empty:
                continue

            ax.scatter(
                subset["Cost_per_Hour"],
                subset["Total_Cost_INR"],
                s=25 if status == "Not flagged" else 34,
                alpha=0.55 if status == "Not flagged" else 0.85,
                color=COLORS[status],
                label=f"{status} (n={len(subset):,})",
                edgecolors="none",
                rasterized=True,
            )

        ax.set_xlabel("Cost per hour")
        ax.set_ylabel("Total expenditure (INR)")
        ax.set_title(
            "Figure 4(b). Cost-Based Anomaly Screening",
            pad=14,
        )
        ax.ticklabel_format(axis="both", style="plain")
        ax.grid(linestyle="--", alpha=0.3)
        ax.legend(frameon=False)
        fig.tight_layout()

        save_figure(fig, "figure_04b_anomaly_cost_scatter")
    else:
        print("Figure 4(b) skipped: no valid numeric observations.")
else:
    print("Figure 4(b) skipped: Cost_per_Hour column not found.")


# --------------------------------------------------
# 7. Report statistics for the research notes
# --------------------------------------------------

total = len(df)
flagged = int(df["Is_Anomaly"].sum())
unflagged = total - flagged

print("\nANOMALY ANALYSIS SUMMARY")
print("=" * 45)
print(f"Records analyzed: {total:,}")
print(f"Flagged records: {flagged:,}")
print(f"Unflagged records: {unflagged:,}")
print(
    f"Flagged percentage: {flagged / total * 100:.2f}%"
    if total else "Flagged percentage: N/A"
)

for status in statuses:
    subset = df[df["Anomaly_Status"] == status]

    if not subset.empty:
        print(
            f"{status} mean cost: "
            f"INR {subset['Total_Cost_INR'].mean():,.2f}"
        )

print("\nStatic research figures generated.")
