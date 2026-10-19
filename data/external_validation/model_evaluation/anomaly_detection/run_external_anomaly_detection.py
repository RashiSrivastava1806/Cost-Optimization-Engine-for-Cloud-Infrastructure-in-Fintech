from pathlib import Path
import json
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path.cwd()
EXTERNAL = ROOT / "data" / "external_validation"
OUT = EXTERNAL / "model_evaluation" / "anomaly_detection"
OUT.mkdir(parents=True, exist_ok=True)

ZIP_PATH = EXTERNAL / "raw" / "dataset-examples.zip"
CSV_NAME = "EA-Cost-FOCUS_1.0.csv"


def load_dataset():
    candidates = [
        EXTERNAL / CSV_NAME,
        EXTERNAL / "raw" / CSV_NAME,
        EXTERNAL / "EA-Cost-FOCUS_1.0" / CSV_NAME,
    ]

    for path in candidates:
        if path.exists():
            print(f"Reading dataset: {path}")
            return pd.read_csv(path, low_memory=False)

    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as archive:
            matches = [
                name for name in archive.namelist()
                if name.endswith(CSV_NAME)
            ]
            if not matches:
                raise FileNotFoundError(
                    f"{CSV_NAME} was not found inside {ZIP_PATH}"
                )
            print(f"Reading {matches[0]} from archive")
            with archive.open(matches[0]) as stream:
                return pd.read_csv(stream, low_memory=False)

    raise FileNotFoundError(
        "Could not find the FOCUS CSV or dataset-examples.zip."
    )


def iqr_upper_fence(series):
    values = pd.to_numeric(series, errors="coerce").dropna()
    if len(values) < 4:
        return np.inf
    q1 = values.quantile(0.25)
    q3 = values.quantile(0.75)
    return q3 + 1.5 * (q3 - q1)


def main():
    df = load_dataset()

    required = {"BilledCost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df["BilledCost"] = pd.to_numeric(df["BilledCost"], errors="coerce")
    df = df.dropna(subset=["BilledCost"]).copy()
    df = df.reset_index(drop=True)

    if df.empty:
        raise ValueError("No valid BilledCost records were found.")

    # Retain negative billing entries. Analyze their magnitude without
    # assuming that negative costs necessarily represent refunds or errors.
    df["Absolute_BilledCost"] = df["BilledCost"].abs()
    df["Log_Absolute_Cost"] = np.log1p(df["Absolute_BilledCost"])

    if "ServiceName" not in df.columns:
        df["ServiceName"] = "Unknown"
    df["ServiceName"] = (
        df["ServiceName"].fillna("Unknown").astype(str).str.strip()
    )
    df.loc[df["ServiceName"] == "", "ServiceName"] = "Unknown"

    # Method 1: global upper IQR fence on log absolute cost.
    global_fence = iqr_upper_fence(df["Log_Absolute_Cost"])
    df["Global_Cost_Outlier"] = df["Log_Absolute_Cost"] > global_fence

    # Method 2: service-relative IQR fence, only for services with enough
    # observations to estimate a within-service distribution.
    service_counts = df.groupby("ServiceName")["ServiceName"].transform("size")
    df["Service_Cost_Outlier"] = False
    service_fences = {}

    for service, indices in df.groupby("ServiceName").groups.items():
        values = df.loc[indices, "Log_Absolute_Cost"]
        if len(values) >= 30:
            fence = iqr_upper_fence(values)
            service_fences[str(service)] = (
                float(fence) if np.isfinite(fence) else None
            )
            df.loc[indices, "Service_Cost_Outlier"] = values > fence

    df["Service_Fence_Available"] = service_counts >= 30

    # Flag a record if either method detects unusually high cost magnitude.
    df["Cost_Outlier_Flag"] = (
        df["Global_Cost_Outlier"] | df["Service_Cost_Outlier"]
    )

    def flag_reason(row):
        reasons = []
        if row["Global_Cost_Outlier"]:
            reasons.append("Global IQR")
        if row["Service_Cost_Outlier"]:
            reasons.append("Within-service IQR")
        return "; ".join(reasons) if reasons else "Not flagged"

    df["Detection_Reason"] = df.apply(flag_reason, axis=1)

    # Preserve useful context fields when available.
    context_fields = [
        "ChargePeriodStart", "ChargePeriodEnd", "BillingCurrency",
        "ServiceName", "ServiceCategory", "ResourceId", "ResourceType",
        "RegionName", "ChargeCategory", "ChargeDescription",
        "PricingCategory", "ConsumedQuantity", "ConsumedUnit",
    ]
    selected = [c for c in context_fields if c in df.columns]
    output_columns = selected + [
        "BilledCost", "Absolute_BilledCost",
        "Global_Cost_Outlier", "Service_Cost_Outlier",
        "Service_Fence_Available", "Cost_Outlier_Flag",
        "Detection_Reason",
    ]

    flagged = df.loc[df["Cost_Outlier_Flag"], output_columns].copy()
    flagged = flagged.sort_values(
        "Absolute_BilledCost", ascending=False
    )
    flagged.to_csv(OUT / "flagged_billing_records.csv", index=False)

    df[output_columns].to_csv(
        OUT / "all_records_with_detection_flags.csv", index=False
    )

    # Method comparison and descriptive financial impact.
    total_records = len(df)
    flagged_count = int(df["Cost_Outlier_Flag"].sum())
    flagged_rows = df[df["Cost_Outlier_Flag"]]
    unflagged_rows = df[~df["Cost_Outlier_Flag"]]

    method_comparison = pd.DataFrame([
        {
            "Method": "Global IQR on log absolute BilledCost",
            "Records_Flagged": int(df["Global_Cost_Outlier"].sum()),
            "Share_of_Records_Percent":
                float(df["Global_Cost_Outlier"].mean() * 100),
            "Threshold_Log_Absolute_Cost":
                float(global_fence) if np.isfinite(global_fence) else None,
        },
        {
            "Method": "Within-service IQR on log absolute BilledCost",
            "Records_Flagged": int(df["Service_Cost_Outlier"].sum()),
            "Share_of_Records_Percent":
                float(df["Service_Cost_Outlier"].mean() * 100),
            "Threshold_Log_Absolute_Cost": None,
        },
        {
            "Method": "Combined flag (either method)",
            "Records_Flagged": flagged_count,
            "Share_of_Records_Percent":
                float(flagged_count / total_records * 100),
            "Threshold_Log_Absolute_Cost": None,
        },
    ])
    method_comparison.to_csv(
        OUT / "detection_method_comparison.csv", index=False
    )

    # Group-level cost impact. Net cost can be negative, so absolute amount
    # is also reported; neither measure is interpreted as confirmed loss.
    impact = pd.DataFrame([
        {
            "Group": "Flagged",
            "Records": int(len(flagged_rows)),
            "Net_BilledCost": float(flagged_rows["BilledCost"].sum()),
            "Absolute_BilledCost": float(
                flagged_rows["Absolute_BilledCost"].sum()
            ),
            "Mean_Absolute_BilledCost": float(
                flagged_rows["Absolute_BilledCost"].mean()
            ) if len(flagged_rows) else 0.0,
        },
        {
            "Group": "Not flagged",
            "Records": int(len(unflagged_rows)),
            "Net_BilledCost": float(unflagged_rows["BilledCost"].sum()),
            "Absolute_BilledCost": float(
                unflagged_rows["Absolute_BilledCost"].sum()
            ),
            "Mean_Absolute_BilledCost": float(
                unflagged_rows["Absolute_BilledCost"].mean()
            ) if len(unflagged_rows) else 0.0,
        },
        {
            "Group": "All records",
            "Records": int(total_records),
            "Net_BilledCost": float(df["BilledCost"].sum()),
            "Absolute_BilledCost": float(
                df["Absolute_BilledCost"].sum()
            ),
            "Mean_Absolute_BilledCost": float(
                df["Absolute_BilledCost"].mean()
            ),
        },
    ])
    impact.to_csv(OUT / "flagged_cost_impact.csv", index=False)

    # Top services by number of flagged billing records.
    service_summary = (
        df.groupby("ServiceName")
        .agg(
            Total_Records=("BilledCost", "size"),
            Flagged_Records=("Cost_Outlier_Flag", "sum"),
            Net_BilledCost=("BilledCost", "sum"),
            Absolute_BilledCost=("Absolute_BilledCost", "sum"),
        )
        .reset_index()
    )
    service_summary["Flagged_Percent"] = (
        service_summary["Flagged_Records"]
        / service_summary["Total_Records"] * 100
    )
    service_summary = service_summary.sort_values(
        ["Flagged_Records", "Absolute_BilledCost"],
        ascending=False,
    )
    service_summary.to_csv(
        OUT / "service_anomaly_summary.csv", index=False
    )

    # Figure 1: flagged versus unflagged record counts.
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = ["Flagged", "Not flagged"]
    counts = [flagged_count, total_records - flagged_count]
    ax.bar(labels, counts)
    ax.set_title("Billing Records Flagged by Cost-Outlier Screening")
    ax.set_ylabel("Number of records")
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(counts):
        ax.text(i, value, f"{value:,}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "figure_01_flagged_vs_unflagged.png", dpi=180)
    plt.close(fig)

    # Figure 2: largest absolute billed-cost records and flags.
    plot_df = df.nlargest(40, "Absolute_BilledCost").copy()
    plot_df = plot_df.sort_values("Absolute_BilledCost")
    fig, ax = plt.subplots(figsize=(11, 7))
    colors = np.where(plot_df["Cost_Outlier_Flag"], "tab:red", "tab:blue")
    ax.barh(
        np.arange(len(plot_df)),
        plot_df["Absolute_BilledCost"],
        color=colors,
    )
    ax.set_title("Largest Absolute Billed-Cost Records")
    ax.set_xlabel("Absolute BilledCost (USD)")
    ax.set_ylabel("Records ranked by cost magnitude")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "figure_02_largest_cost_records.png", dpi=180)
    plt.close(fig)

    # Figure 3: services with the most flagged records.
    top_services = service_summary.head(15).sort_values("Flagged_Records")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(
        top_services["ServiceName"],
        top_services["Flagged_Records"],
    )
    ax.set_title("Services with the Most Flagged Billing Records")
    ax.set_xlabel("Flagged records")
    ax.set_ylabel("Service")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "figure_03_flagged_by_service.png", dpi=180)
    plt.close(fig)

    # Figure 4: absolute-cost distribution before and after log transform.
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(
        np.log1p(df["Absolute_BilledCost"]),
        bins=60,
        alpha=0.75,
    )
    ax.axvline(
        global_fence, linestyle="--",
        label="Global IQR upper fence"
    )
    ax.set_title("Log Absolute Billed-Cost Distribution and IQR Threshold")
    ax.set_xlabel("log(1 + absolute BilledCost)")
    ax.set_ylabel("Record count")
    ax.legend()
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(OUT / "figure_04_cost_outlier_threshold.png", dpi=180)
    plt.close(fig)

    summary = {
        "dataset": "Microsoft FinOps Toolkit demo FOCUS dataset",
        "target_field": "BilledCost",
        "currency": "USD, based on prior dataset inspection; verify currency_summary.csv",
        "records_analyzed": int(total_records),
        "records_flagged": flagged_count,
        "flagged_share_percent": float(flagged_count / total_records * 100),
        "records_not_flagged": int(total_records - flagged_count),
        "global_log_absolute_cost_iqr_upper_fence": (
            float(global_fence) if np.isfinite(global_fence) else None
        ),
        "service_groups_with_at_least_30_records": int(
            (df.groupby("ServiceName").size() >= 30).sum()
        ),
        "flagged_net_billed_cost_usd": float(
            flagged_rows["BilledCost"].sum()
        ),
        "flagged_absolute_billed_cost_usd": float(
            flagged_rows["Absolute_BilledCost"].sum()
        ),
        "all_records_net_billed_cost_usd": float(df["BilledCost"].sum()),
        "all_records_absolute_billed_cost_usd": float(
            df["Absolute_BilledCost"].sum()
        ),
        "method": (
            "Screening flags records with unusually high log absolute "
            "BilledCost using a global IQR fence or a within-service IQR "
            "fence. Service-relative thresholds are only calculated for "
            "services with at least 30 records."
        ),
        "limitations": [
            "No labelled incidents are available; precision, recall, "
            "false-positive rate and accuracy cannot be calculated.",
            "A flagged billing record is a cost outlier for investigation, "
            "not a confirmed incident, fraud event or billing error.",
            "CPU, memory and application telemetry are not present in this "
            "billing dataset, so utilization-based anomalies are not tested.",
            "Negative costs are retained and analyzed by magnitude; their "
            "business meaning is not inferred.",
            "The thresholds are statistical screening rules and may require "
            "service-specific review before operational use."
        ],
    }

    with open(OUT / "anomaly_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    index = pd.DataFrame([
        ["flagged_billing_records.csv", "Flagged records and reasons"],
        ["all_records_with_detection_flags.csv", "All records with detection flags"],
        ["detection_method_comparison.csv", "Counts from each screening method"],
        ["flagged_cost_impact.csv", "Net and absolute cost associated with flags"],
        ["service_anomaly_summary.csv", "Flagged records by service"],
        ["figure_01_flagged_vs_unflagged.png", "Flagged versus unflagged counts"],
        ["figure_02_largest_cost_records.png", "Largest cost-magnitude records"],
        ["figure_03_flagged_by_service.png", "Services with most flags"],
        ["figure_04_cost_outlier_threshold.png", "Cost distribution and threshold"],
        ["anomaly_summary.json", "Methods, results and limitations"],
    ], columns=["File", "Description"])
    index.to_csv(OUT / "output_index.csv", index=False)

    print("\n========== EXTERNAL COST-OUTLIER SCREENING COMPLETE ==========")
    print(f"Output folder: {OUT}")
    print(f"Records analyzed: {total_records:,}")
    print(f"Records flagged: {flagged_count:,}")
    print(f"Flagged share: {flagged_count / total_records * 100:.2f}%")
    print(
        "Net BilledCost associated with flagged rows: "
        f"${flagged_rows['BilledCost'].sum():,.2f}"
    )
    print(
        "Absolute BilledCost magnitude associated with flagged rows: "
        f"${flagged_rows['Absolute_BilledCost'].sum():,.2f}"
    )
    print("\nMethod comparison:")
    print(method_comparison.to_string(index=False))
    print("\nGenerated files:")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
