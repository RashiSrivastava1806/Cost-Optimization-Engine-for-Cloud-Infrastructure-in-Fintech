from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path.cwd()
BASE = ROOT / "data" / "external_validation" / "model_evaluation"
OUT = BASE / "summary"
OUT.mkdir(parents=True, exist_ok=True)

FOLDERS = {
    "forecasting": BASE / "forecasting",
    "anomaly_detection": BASE / "anomaly_detection",
    "optimization": BASE / "optimization",
    "sla": BASE / "sla",
}

EXPECTED_JSON = {
    "forecasting": "run_summary.json",
    "anomaly_detection": "anomaly_summary.json",
    "optimization": "optimization_summary.json",
    "sla": "sla_summary.json",
}


def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def as_float(value):
    try:
        x = float(value)
        return x if np.isfinite(x) else None
    except (TypeError, ValueError):
        return None


def add_metric(rows, category, metric, value, unit, interpretation):
    rows.append({
        "Analysis": category,
        "Metric": metric,
        "Value": value,
        "Unit": unit,
        "Interpretation": interpretation,
    })


summaries = {
    category: read_json(folder / EXPECTED_JSON[category])
    for category, folder in FOLDERS.items()
}

# 1. Build a consolidated, machine-readable results table.
metrics = []

forecast = summaries["forecasting"]
best_metrics = forecast.get("best_model_metrics", {})

add_metric(
    metrics, "Forecasting", "Best model",
    forecast.get("best_model_by_MAE", "Unavailable"),
    "model",
    "Selected by holdout MAE in the external benchmark."
)
add_metric(
    metrics, "Forecasting", "Holdout MAE",
    as_float(best_metrics.get("MAE")), "USD",
    "Lower is better; evaluate alongside the short test window."
)
add_metric(
    metrics, "Forecasting", "Holdout RMSE",
    as_float(best_metrics.get("RMSE")), "USD",
    "Penalizes larger prediction errors."
)
add_metric(
    metrics, "Forecasting", "Holdout WAPE",
    as_float(best_metrics.get("WAPE_percent")), "percent",
    "Weighted absolute percentage error on the holdout period."
)
add_metric(
    metrics, "Forecasting", "Training observations",
    forecast.get("train_observations"), "daily observations",
    "Chronological training window."
)
add_metric(
    metrics, "Forecasting", "Test observations",
    forecast.get("test_observations"), "daily observations",
    "Chronological holdout; a small sample limits confidence."
)

anomaly = summaries["anomaly_detection"]
add_metric(
    metrics, "Anomaly Detection", "Records analyzed",
    anomaly.get("records_analyzed"), "records",
    "Billing records screened for cost outliers."
)
add_metric(
    metrics, "Anomaly Detection", "Records flagged",
    anomaly.get("records_flagged"), "records",
    "Statistical cost-outlier flags; not confirmed incidents."
)
add_metric(
    metrics, "Anomaly Detection", "Flagged share",
    as_float(anomaly.get("flagged_share_percent")), "percent",
    "Share flagged by the combined screening rule."
)
add_metric(
    metrics, "Anomaly Detection", "Flagged rows net billed cost",
    as_float(anomaly.get("flagged_net_billed_cost_usd")), "USD",
    "Associated billing amount, not proven loss or savings."
)
add_metric(
    metrics, "Anomaly Detection", "Ground-truth accuracy",
    "NOT_AVAILABLE", "status",
    "No labelled incidents are available to calculate accuracy."
)

optimization = summaries["optimization"]
add_metric(
    metrics, "Optimization", "Review candidate groups",
    optimization.get("review_candidate_groups"), "resource groups",
    "High-spend review candidates, not verified waste."
)
add_metric(
    metrics, "Optimization", "Review threshold",
    as_float(optimization.get("review_threshold_positive_cost_usd")), "USD",
    "90th percentile threshold of positive spend across identified resources."
)
add_metric(
    metrics, "Optimization", "Candidate positive billed cost",
    as_float(optimization.get("candidate_positive_billed_cost_usd")), "USD",
    "Spend associated with candidate groups, not potential savings."
)
add_metric(
    metrics, "Optimization", "Candidate share of positive spend",
    as_float(optimization.get("candidate_share_of_positive_billed_cost_percent")),
    "percent",
    "Share of positive billed cost associated with review candidates."
)

scenario_values = optimization.get("scenario_estimates_usd", {})
for rate, value in scenario_values.items():
    add_metric(
        metrics, "Optimization",
        f"Hypothetical {rate}% reduction scenario",
        as_float(value), "USD",
        "Illustrative scenario only; not realized or verified savings."
    )

sla = summaries["sla"]
add_metric(
    metrics, "SLA", "Overall compliance status",
    sla.get("overall_sla_compliance_status", "NOT_VERIFIABLE"),
    "status",
    "Billing data alone cannot establish SLA compliance."
)
add_metric(
    metrics, "SLA", "Evidence categories assessed",
    sla.get("evidence_categories_assessed"), "categories",
    "Required operational evidence categories examined."
)
add_metric(
    metrics, "SLA", "Categories missing",
    sla.get("categories_missing"), "categories",
    "No recognized matching field was found."
)
add_metric(
    metrics, "SLA", "Categories requiring validation",
    sla.get("categories_with_matching_fields_requiring_validation"),
    "categories",
    "A matching field name alone does not establish valid SLA evidence."
)
add_metric(
    metrics, "SLA", "Automatic optimization approval",
    sla.get("automatic_optimization_approval", "NOT_AUTHORIZED"),
    "status",
    "Do not automatically modify production resources without validated evidence."
)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(OUT / "master_results_table.csv", index=False)

# 2. Consolidate the forecasting model comparison if available.
forecast_comparison = FOLDERS["forecasting"] / "model_comparison.csv"
if forecast_comparison.exists():
    pd.read_csv(forecast_comparison).to_csv(
        OUT / "forecast_model_comparison.csv", index=False
    )

# 3. Build an inventory of every file in the model-evaluation folders.
inventory_rows = []
for category, folder in FOLDERS.items():
    if not folder.exists():
        inventory_rows.append({
            "Analysis": category,
            "File": "",
            "Type": "",
            "Status": "FOLDER_MISSING",
            "Description": "Expected output folder does not exist.",
        })
        continue

    files = sorted(p for p in folder.rglob("*") if p.is_file())
    if not files:
        inventory_rows.append({
            "Analysis": category,
            "File": "",
            "Type": "",
            "Status": "NO_OUTPUT_FILES",
            "Description": "Folder exists but contains no files.",
        })

    for path in files:
        inventory_rows.append({
            "Analysis": category,
            "File": str(path.relative_to(ROOT)),
            "Type": path.suffix.lower().lstrip(".") or "no extension",
            "Status": "CREATED",
            "Description": "",
        })

inventory_df = pd.DataFrame(inventory_rows)
inventory_df.to_csv(OUT / "output_inventory.csv", index=False)

# 4. Make an overall visual overview with honest status labels.
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Forecasting panel
ax = axes[0, 0]
comparison_path = FOLDERS["forecasting"] / "model_comparison.csv"
if comparison_path.exists():
    comp = pd.read_csv(comparison_path)
    if {"Model", "MAE"}.issubset(comp.columns):
        comp = comp.dropna(subset=["MAE"]).sort_values("MAE")
        ax.bar(comp["Model"].astype(str), comp["MAE"])
        ax.set_title("Forecasting: Holdout MAE (USD)")
        ax.set_ylabel("MAE; lower is better")
        ax.tick_params(axis="x", rotation=20)
        ax.grid(axis="y", alpha=0.25)
    else:
        ax.text(0.5, 0.5, "Model comparison columns unavailable",
                ha="center", va="center", wrap=True)
        ax.set_title("Forecasting")
else:
    ax.text(0.5, 0.5, "Forecast comparison not generated",
            ha="center", va="center", wrap=True)
    ax.set_title("Forecasting")

# Anomaly panel
ax = axes[0, 1]
flagged = as_float(anomaly.get("records_flagged"))
not_flagged = None
analyzed = as_float(anomaly.get("records_analyzed"))
if flagged is not None and analyzed is not None:
    not_flagged = max(0, analyzed - flagged)
    ax.bar(
        ["Flagged", "Not flagged"],
        [flagged, not_flagged]
    )
    ax.set_title("Billing Cost-Outlier Screening")
    ax.set_ylabel("Records")
    ax.grid(axis="y", alpha=0.25)
else:
    ax.text(0.5, 0.5, "Anomaly output not generated",
            ha="center", va="center", wrap=True)
    ax.set_title("Anomaly Detection")

# Optimization panel
ax = axes[1, 0]
candidate_spend = as_float(
    optimization.get("candidate_positive_billed_cost_usd")
)
scenario_10 = as_float(scenario_values.get("10"))
if candidate_spend is not None and scenario_10 is not None:
    ax.bar(
        ["Candidate positive spend", "Hypothetical 10% scenario"],
        [candidate_spend, scenario_10]
    )
    ax.set_title("Optimization: Spend vs Illustrative Scenario")
    ax.set_ylabel("USD")
    ax.tick_params(axis="x", rotation=12)
    ax.grid(axis="y", alpha=0.25)
else:
    ax.text(0.5, 0.5, "Optimization output not generated",
            ha="center", va="center", wrap=True)
    ax.set_title("Optimization")
    
# SLA panel
ax = axes[1, 1]
ax.axis("off")
ax.set_title("SLA Evidence Assessment")
sla_lines = [
    f"Status: {sla.get('overall_sla_compliance_status', 'NOT_VERIFIABLE')}",
    f"Evidence categories: {sla.get('evidence_categories_assessed', 'N/A')}",
    f"Missing categories: {sla.get('categories_missing', 'N/A')}",
    f"Require validation: {sla.get('categories_with_matching_fields_requiring_validation', 'N/A')}",
    "",
    "Billing records do not prove uptime, latency",
    "or contractual SLA compliance.",
]
ax.text(
    0.03, 0.82, "\n".join(sla_lines),
    transform=ax.transAxes,
    ha="left", va="top", fontsize=11,
    bbox={"boxstyle": "round,pad=0.6", "fill": False}
)

fig.suptitle(
    "External FOCUS Dataset: Model Evaluation Overview",
    fontsize=15
)
fig.tight_layout(rect=[0, 0, 1, 0.96])
fig.savefig(OUT / "external_validation_overview.png", dpi=180)
plt.close(fig)

# 5. Write a concise report for reviewing and presenting results.
report_lines = [
    "# External Dataset Model Evaluation Report",
    "",
    "## Dataset",
    "",
    "Microsoft FinOps Toolkit demo FOCUS billing dataset. The primary cost "
    "measure used in these analyses is `BilledCost` in USD.",
    "",
    "## Results",
    "",
    "### Forecasting",
    "",
    f"- Best model by holdout MAE: {forecast.get('best_model_by_MAE', 'Unavailable')}",
    f"- Holdout MAE: {best_metrics.get('MAE', 'Unavailable')} USD",
    f"- Holdout RMSE: {best_metrics.get('RMSE', 'Unavailable')} USD",
    f"- Holdout WAPE: {best_metrics.get('WAPE_percent', 'Unavailable')}%",
    f"- Test observations: {forecast.get('test_observations', 'Unavailable')}",
    "- The short daily history and holdout period limit confidence in generalization.",
    "- The standalone benchmark does not automatically validate the canonical project model.",
    "",
    "### Anomaly detection",
    "",
    f"- Records analyzed: {anomaly.get('records_analyzed', 'Unavailable')}",
    f"- Cost-outlier flags: {anomaly.get('records_flagged', 'Unavailable')}",
    f"- Flagged share: {anomaly.get('flagged_share_percent', 'Unavailable')}%",
    "- Flags indicate statistical billing-cost outliers, not confirmed incidents.",
    "- No labelled incidents are available to measure precision, recall, or accuracy.",
    "",
    "### Optimization",
    "",
    f"- High-spend review candidate groups: {optimization.get('review_candidate_groups', 'Unavailable')}",
    f"- Candidate positive billed cost: {optimization.get('candidate_positive_billed_cost_usd', 'Unavailable')} USD",
    "- Candidates require usage, criticality, owner, and SLA review before any changes.",
    "- Reduction scenarios are hypothetical and are not realized savings.",
    "",
    "### SLA evidence",
    "",
    f"- Overall status: {sla.get('overall_sla_compliance_status', 'NOT_VERIFIABLE')}",
    f"- Evidence categories assessed: {sla.get('evidence_categories_assessed', 'Unavailable')}",
    f"- Missing categories: {sla.get('categories_missing', 'Unavailable')}",
    "- Billing data alone cannot establish uptime, latency, error rates, or contractual SLA compliance.",
    "- Automatic production optimization must not be authorized from these billing records alone.",
    "",
    "## Research limitations",
    "",
    "1. The dataset is a demo sample rather than production customer evidence.",
    "2. Forecasting evidence is limited by the short time series and holdout size.",
    "3. Anomaly labels are unavailable, so detection accuracy cannot be established.",
    "4. High spend does not itself prove waste, inefficiency, or safe downsizing.",
    "5. SLA compliance remains not verifiable without validated operational measurements and targets.",
    "6. Hypothetical cost-reduction scenarios must not be described as measured savings.",
    "",
    "## Output folders",
    "",
    "- `forecasting/`: holdout predictions, metrics, and forecasting graphs.",
    "- `anomaly_detection/`: flagged records, method comparison, and cost-outlier graphs.",
    "- `optimization/`: cost-review ranking, spend summaries, and illustrative scenarios.",
    "- `sla/`: evidence matrix, verification statuses, and evidence graphs.",
    "- `summary/`: consolidated metrics, inventory, overview graph, and this report.",
    "",
    "## Overall conclusion",
    "",
    "This external evaluation demonstrates a repeatable analysis workflow on a "
    "second billing dataset. It provides preliminary evidence about forecasting "
    "error, statistical cost outliers, spend concentration, and available SLA "
    "evidence. It does not prove production forecasting reliability, confirmed "
    "incidents, realized savings, or SLA compliance.",
]

(OUT / "external_validation_report.md").write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8"
)

# Inventory also lists the main report and overview image.
extra_rows = [
    {
        "Analysis": "summary",
        "File": str((OUT / "master_results_table.csv").relative_to(ROOT)),
        "Type": "csv",
        "Status": "CREATED",
        "Description": "Consolidated metrics across all analyses.",
    },
    {
        "Analysis": "summary",
        "File": str((OUT / "output_inventory.csv").relative_to(ROOT)),
        "Type": "csv",
        "Status": "CREATED",
        "Description": "Index of analysis artifacts.",
    },
    {
        "Analysis": "summary",
        "File": str((OUT / "external_validation_overview.png").relative_to(ROOT)),
        "Type": "png",
        "Status": "CREATED",
        "Description": "Four-part external evaluation overview.",
    },
    {
        "Analysis": "summary",
        "File": str((OUT / "external_validation_report.md").relative_to(ROOT)),
        "Type": "md",
        "Status": "CREATED",
        "Description": "Consolidated findings and limitations.",
    },
]
inventory_df = pd.concat(
    [inventory_df, pd.DataFrame(extra_rows)],
    ignore_index=True
)
inventory_df.to_csv(OUT / "output_inventory.csv", index=False)

print("\n========== EXTERNAL VALIDATION SUMMARY COMPLETE ==========")
print(f"Summary folder: {OUT}")
print("\nMaster results:")
print(metrics_df.to_string(index=False))
print("\nOutput inventory status:")
print(inventory_df.groupby(["Analysis", "Status"]).size().to_string())
print("\nSummary artifacts:")
for name in [
    "master_results_table.csv",
    "forecast_model_comparison.csv",
    "output_inventory.csv",
    "external_validation_overview.png",
    "external_validation_report.md",
]:
    path = OUT / name
    if path.exists():
        print(f"  {path.relative_to(ROOT)}")
    elif name == "forecast_model_comparison.csv":
        print(f"  {path.relative_to(ROOT)} (not available: forecast comparison missing)")
