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
OUT = EXTERNAL / "model_evaluation" / "optimization"
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
        "Cannot find the FOCUS CSV or dataset-examples.zip."
    )


def clean_text(series, default="Unknown"):
    result = series.fillna(default).astype(str).str.strip()
    return result.mask(result.eq(""), default)


def summarize_cost(group, keys):
    summary = group.groupby(keys, dropna=False).agg(
        Record_Count=("BilledCost", "size"),
        Net_BilledCost=("BilledCost", "sum"),
        Positive_Cost=("Positive_Cost", "sum"),
        Negative_Cost=("Negative_Cost", "sum"),
        Absolute_Cost=("Absolute_Cost", "sum"),
    ).reset_index()

    summary["Positive_Spend_Share_Percent"] = np.where(
        summary["Positive_Cost"].sum() > 0,
        summary["Positive_Cost"] / summary["Positive_Cost"].sum() * 100,
        0.0,
    )
    return summary


def main():
    df = load_dataset()

    if "BilledCost" not in df.columns:
        raise ValueError("Required column BilledCost is missing.")

    df["BilledCost"] = pd.to_numeric(df["BilledCost"], errors="coerce")
    df = df.dropna(subset=["BilledCost"]).copy()

    if df.empty:
        raise ValueError("No valid billing costs were found.")

    # Normalize grouping fields where available.
    for field in [
        "ResourceId", "ServiceName", "ServiceCategory",
        "RegionName", "ResourceType", "BillingCurrency",
        "ChargeCategory",
    ]:
        if field not in df.columns:
            df[field] = "Unknown"
        df[field] = clean_text(df[field])

    df["Positive_Cost"] = df["BilledCost"].clip(lower=0)
    df["Negative_Cost"] = df["BilledCost"].clip(upper=0)
    df["Absolute_Cost"] = df["BilledCost"].abs()

    # A resource-level grouping is possible only when an identifier is
    # available. Unknown IDs are kept distinct only as an "Unknown" group,
    # so interpret that aggregate cautiously.
    resource_summary = summarize_cost(
        df, ["ResourceId", "ServiceName", "ServiceCategory",
             "RegionName", "ResourceType", "BillingCurrency"]
    )

    # Define review candidates using high positive billed spend only.
    # This is a prioritization heuristic, not a waste or utilization test.
    identified = resource_summary[
        resource_summary["ResourceId"].ne("Unknown")
        & resource_summary["Positive_Cost"].gt(0)
    ].copy()

    if identified.empty:
        raise ValueError(
            "No identifiable resources with positive billed costs were found."
        )

    spend_threshold = float(identified["Positive_Cost"].quantile(0.90))
    identified["High_Spend_Review_Candidate"] = (
        identified["Positive_Cost"] >= spend_threshold
    )

    identified["Review_Priority"] = np.select(
        [
            identified["Positive_Cost"] >= identified["Positive_Cost"].quantile(0.99),
            identified["Positive_Cost"] >= spend_threshold,
        ],
        ["Priority 1 - Highest spend", "Priority 2 - High spend"],
        default="Not in top-decile review group",
    )

    identified["Review_Reason"] = np.where(
        identified["High_Spend_Review_Candidate"],
        "High positive billed spend; verify usage, utilization, business criticality and SLA before any change",
        "Below high-spend review threshold",
    )

    identified = identified.sort_values(
        "Positive_Cost", ascending=False
    )
    identified.to_csv(
        OUT / "resource_cost_review_ranking.csv", index=False
    )

    candidates = identified[
        identified["High_Spend_Review_Candidate"]
    ].copy()

    # Service and region tables help reviewers identify where spending
    # is concentrated without declaring the spending unnecessary.
    service_summary = summarize_cost(df, ["ServiceName", "ServiceCategory"])
    service_summary = service_summary.sort_values(
        "Positive_Cost", ascending=False
    )
    service_summary.to_csv(
        OUT / "service_cost_review_summary.csv", index=False
    )

    region_summary = summarize_cost(df, ["RegionName"])
    region_summary = region_summary.sort_values(
        "Positive_Cost", ascending=False
    )
    region_summary.to_csv(
        OUT / "region_cost_review_summary.csv", index=False
    )

    resource_type_summary = summarize_cost(df, ["ResourceType"])
    resource_type_summary = resource_type_summary.sort_values(
        "Positive_Cost", ascending=False
    )
    resource_type_summary.to_csv(
        OUT / "resource_type_cost_summary.csv", index=False
    )

    # Scenarios apply only to positive billed spend in the review group.
    # They are sensitivity scenarios, not realized or verified savings.
    candidate_positive_spend = float(candidates["Positive_Cost"].sum())
    scenario_rows = []

    for rate in [0.05, 0.10, 0.15]:
        scenario_rows.append({
            "Scenario": f"{int(rate * 100)}% reduction scenario",
            "Assumed_Reduction_Percent": rate * 100,
            "Review_Candidate_Resources": int(len(candidates)),
            "Candidate_Positive_BilledCost_USD": candidate_positive_spend,
            "Illustrative_Impact_USD": candidate_positive_spend * rate,
            "Interpretation": (
                "Hypothetical only; assumes a reduction across candidate "
                "positive billed spend and does not establish feasibility, "
                "SLA safety, or realized savings."
            ),
        })

    scenarios = pd.DataFrame(scenario_rows)
    scenarios.to_csv(
        OUT / "illustrative_savings_scenarios.csv", index=False
    )

    # Concentration metrics based on positive billed spend.
    total_positive = float(df["Positive_Cost"].sum())
    candidate_share = (
        candidate_positive_spend / total_positive * 100
        if total_positive > 0 else 0.0
    )

    concentration = pd.DataFrame([
        {
            "Metric": "Total records analyzed",
            "Value": int(len(df)),
            "Unit": "records",
        },
        {
            "Metric": "Identified resource groups with positive spend",
            "Value": int(len(identified)),
            "Unit": "groups",
        },
        {
            "Metric": "High-spend review candidate groups",
            "Value": int(len(candidates)),
            "Unit": "groups",
        },
        {
            "Metric": "90th percentile resource positive spend threshold",
            "Value": spend_threshold,
            "Unit": "USD",
        },
        {
            "Metric": "Total positive BilledCost",
            "Value": total_positive,
            "Unit": "USD",
        },
        {
            "Metric": "Candidate positive BilledCost",
            "Value": candidate_positive_spend,
            "Unit": "USD",
        },
        {
            "Metric": "Candidate share of total positive BilledCost",
            "Value": candidate_share,
            "Unit": "percent",
        },
        {
            "Metric": "Total net BilledCost",
            "Value": float(df["BilledCost"].sum()),
            "Unit": "USD",
        },
        {
            "Metric": "Total absolute BilledCost",
            "Value": float(df["Absolute_Cost"].sum()),
            "Unit": "USD",
        },
    ])
    concentration.to_csv(
        OUT / "optimization_concentration_metrics.csv", index=False
    )

    # Figure 1: top 15 resources by positive billed cost.
    top_resources = identified.head(15).sort_values("Positive_Cost")
    labels = [
        (str(row.ResourceId)[:36] + "…")
        if len(str(row.ResourceId)) > 37 else str(row.ResourceId)
        for row in top_resources.itertuples()
    ]

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.barh(labels, top_resources["Positive_Cost"])
    ax.set_title("Top Resources by Positive Billed Cost")
    ax.set_xlabel("Positive BilledCost (USD)")
    ax.set_ylabel("Resource ID")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "figure_01_top_resource_spend.png", dpi=180)
    plt.close(fig)

    # Figure 2: top 12 services by positive spend.
    top_services = service_summary.head(12).sort_values("Positive_Cost")
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(top_services["ServiceName"], top_services["Positive_Cost"])
    ax.set_title("Service-Level Positive Billed Cost")
    ax.set_xlabel("Positive BilledCost (USD)")
    ax.set_ylabel("Service")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "figure_02_service_spend.png", dpi=180)
    plt.close(fig)

    # Figure 3: illustrative impact of three hypothetical reduction rates.
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        scenarios["Scenario"],
        scenarios["Illustrative_Impact_USD"],
    )
    ax.set_title("Illustrative Reduction Scenarios for High-Spend Candidates")
    ax.set_ylabel("Hypothetical impact (USD)")
    ax.set_xlabel("Assumed reduction scenario")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "figure_03_illustrative_scenarios.png", dpi=180)
    plt.close(fig)

    # Figure 4: positive billed-spend concentration in top resource groups.
    sorted_spend = identified["Positive_Cost"].sort_values(
        ascending=False
    ).reset_index(drop=True)
    cumulative = sorted_spend.cumsum() / sorted_spend.sum() * 100
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(
        np.arange(1, len(cumulative) + 1),
        cumulative,
        linewidth=2,
    )
    ax.axhline(80, linestyle="--", label="80% of positive spend")
    ax.set_title("Cumulative Positive Spend Across Ranked Resources")
    ax.set_xlabel("Resources ranked from highest to lowest spend")
    ax.set_ylabel("Cumulative positive spend (%)")
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "figure_04_spend_concentration.png", dpi=180)
    plt.close(fig)

    summary = {
        "dataset": "Microsoft FinOps Toolkit demo FOCUS dataset",
        "cost_field": "BilledCost",
        "currency": "USD, as identified in prior dataset inspection; verify BillingCurrency",
        "records_analyzed": int(len(df)),
        "identifiable_resource_groups_with_positive_spend": int(len(identified)),
        "review_candidate_groups": int(len(candidates)),
        "review_threshold_positive_cost_usd": spend_threshold,
        "candidate_positive_billed_cost_usd": candidate_positive_spend,
        "total_positive_billed_cost_usd": total_positive,
        "candidate_share_of_positive_billed_cost_percent": candidate_share,
        "scenario_estimates_usd": {
            str(int(row["Assumed_Reduction_Percent"])): float(
                row["Illustrative_Impact_USD"]
            )
            for row in scenario_rows
        },
        "candidate_rule": (
            "Identifiable resource groups with positive billed spend at or "
            "above the 90th percentile of positive spend across identifiable "
            "resource groups."
        ),
        "limitations": [
            "High spend alone does not prove waste or inefficiency.",
            "The dataset does not provide sufficient utilization, workload, "
            "criticality or SLA evidence to approve downsizing automatically.",
            "Savings scenarios are hypothetical and are not realized savings.",
            "Any proposed change requires owner review, usage evidence, "
            "business criticality checks and SLA/performance validation.",
            "Negative billing entries are reported separately and not "
            "assumed to be refunds or errors."
        ],
    }

    with open(OUT / "optimization_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    index = pd.DataFrame([
        ["resource_cost_review_ranking.csv", "Ranked resource review candidates"],
        ["service_cost_review_summary.csv", "Positive, negative and net cost by service"],
        ["region_cost_review_summary.csv", "Cost summaries by region"],
        ["resource_type_cost_summary.csv", "Cost summaries by resource type"],
        ["illustrative_savings_scenarios.csv", "Hypothetical 5%, 10% and 15% scenarios"],
        ["optimization_concentration_metrics.csv", "Spend concentration and candidate metrics"],
        ["figure_01_top_resource_spend.png", "Highest-spend resources"],
        ["figure_02_service_spend.png", "Service-level positive spend"],
        ["figure_03_illustrative_scenarios.png", "Hypothetical reduction scenarios"],
        ["figure_04_spend_concentration.png", "Cumulative resource spend curve"],
        ["optimization_summary.json", "Methods, results and limitations"],
    ], columns=["File", "Description"])
    index.to_csv(OUT / "output_index.csv", index=False)

    print("\n========== EXTERNAL OPTIMIZATION REVIEW COMPLETE ==========")
    print(f"Output folder: {OUT}")
    print(f"Records analyzed: {len(df):,}")
    print(f"Identifiable positive-spend resource groups: {len(identified):,}")
    print(f"High-spend review candidates: {len(candidates):,}")
    print(f"Review threshold: ${spend_threshold:,.2f}")
    print(f"Candidate positive spend: ${candidate_positive_spend:,.2f}")
    print(f"Share of positive spend: {candidate_share:.2f}%")
    print("\nIllustrative scenarios (not measured savings):")
    print(scenarios.to_string(index=False))
    print("\nGenerated files:")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
