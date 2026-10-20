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
OUT = EXTERNAL / "model_evaluation" / "sla"
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


def normalize(name):
    return "".join(ch.lower() for ch in str(name) if ch.isalnum())


def main():
    df = load_dataset()
    columns = list(df.columns)
    normalized_to_original = {normalize(c): c for c in columns}

    # Required evidence categories and common possible column-name variants.
    # A matching column is evidence of field availability only; it does not
    # by itself establish valid measurement semantics or contractual compliance.
    requirements = [
        {
            "Evidence_Category": "Availability / uptime",
            "Required_Signal": "Measured service availability or uptime",
            "Aliases": [
                "availability", "availabilitypercent", "uptime",
                "uptimepercent", "serviceavailability", "availabilityratio",
            ],
            "Why_It_Matters": "Needed to compare measured availability with a defined SLA target.",
        },
        {
            "Evidence_Category": "Latency / response time",
            "Required_Signal": "Measured request latency or response time",
            "Aliases": [
                "latency", "latencyp95", "latencyp99", "responsetime",
                "responsetimems", "durationms", "requestduration",
            ],
            "Why_It_Matters": "Needed to evaluate whether performance meets response-time objectives.",
        },
        {
            "Evidence_Category": "Error rate",
            "Required_Signal": "Measured failed requests or error rate",
            "Aliases": [
                "errorrate", "errorratepercent", "faileddrequests",
                "failedrequests", "http5xx", "http5xxcount",
                "requesterrors", "failurecount",
            ],
            "Why_It_Matters": "Needed to evaluate service failures and reliability objectives.",
        },
        {
            "Evidence_Category": "SLA / SLO target",
            "Required_Signal": "Explicit contractual target or service objective",
            "Aliases": [
                "slatarget", "availabilitytarget", "uptimetarget",
                "latencytarget", "slo", "sloavailability",
                "targetavailability", "contractualtarget",
            ],
            "Why_It_Matters": "Measured service behavior must be compared against an explicit target.",
        },
        {
            "Evidence_Category": "Incident / outage",
            "Required_Signal": "Incident records, outage windows, or downtime duration",
            "Aliases": [
                "incidentid", "incidentcount", "outage", "outagecount",
                "downtime", "downtimeminutes", "incidentstart",
                "incidentend", "serviceinterruption",
            ],
            "Why_It_Matters": "Helps corroborate service interruptions and downtime calculations.",
        },
        {
            "Evidence_Category": "Workload / request volume",
            "Required_Signal": "Requests, transactions, or workload volume",
            "Aliases": [
                "requestcount", "requestspersecond", "transactioncount",
                "throughput", "throughputrps", "requestvolume",
                "operationscount",
            ],
            "Why_It_Matters": "Provides workload context for interpreting performance measurements.",
        },
        {
            "Evidence_Category": "CPU utilization",
            "Required_Signal": "Measured CPU usage or utilization",
            "Aliases": [
                "cpuutilization", "cpuusagepercent", "cpupercent",
                "cpuusage", "cpuutilizationpercent",
            ],
            "Why_It_Matters": "Helps assess capacity and whether a compute optimization could affect performance.",
        },
        {
            "Evidence_Category": "Memory utilization",
            "Required_Signal": "Measured memory usage or utilization",
            "Aliases": [
                "memoryutilization", "memoryusagepercent",
                "memorypercent", "memoryused", "memoryutilizationpercent",
            ],
            "Why_It_Matters": "Helps assess memory capacity and the safety of resource changes.",
        },
        {
            "Evidence_Category": "Network performance",
            "Required_Signal": "Network latency, packet loss, or throughput telemetry",
            "Aliases": [
                "networklatency", "packetloss", "packetlosspercent",
                "networkthroughput", "networkerrors", "networkavailability",
            ],
            "Why_It_Matters": "Needed to assess network-related performance and reliability.",
        },
        {
            "Evidence_Category": "Measurement timestamp",
            "Required_Signal": "Timestamped operational measurements",
            "Aliases": [
                "measurementtimestamp", "metricstimestamp", "observedat",
                "eventtimestamp", "telemetrytimestamp",
            ],
            "Why_It_Matters": "Operational measurements need a time basis to evaluate SLA windows.",
        },
    ]

    rows = []
    for item in requirements:
        matched = [
            normalized_to_original[a]
            for a in item["Aliases"]
            if a in normalized_to_original
        ]

        if not matched:
            status = "MISSING_FROM_DATASET"
            non_null_rows = 0
            details = "No recognized matching field found."
        else:
            non_null_rows = int(
                df[matched].notna().any(axis=1).sum()
            )
            if non_null_rows == 0:
                status = "FIELD_PRESENT_BUT_EMPTY"
                details = "Matching column exists, but all values are missing."
            else:
                status = "FIELD_PRESENT_REQUIRES_VALIDATION"
                details = (
                    "Matching field exists; meaning, units, source, sampling "
                    "and target semantics must be validated before use."
                )

        rows.append({
            "Evidence_Category": item["Evidence_Category"],
            "Required_Signal": item["Required_Signal"],
            "Matched_Columns": "; ".join(matched) if matched else "",
            "Rows_With_At_Least_One_NonNull_Matched_Value": non_null_rows,
            "Total_Dataset_Rows": int(len(df)),
            "Status": status,
            "Why_It_Matters": item["Why_It_Matters"],
            "Assessment_Notes": details,
        })

    evidence = pd.DataFrame(rows)
    evidence.to_csv(OUT / "sla_evidence_matrix.csv", index=False)

    # Identify basic billing and date fields separately. They are not SLA
    # evidence, but document what this dataset can actually support.
    basic_fields = [
        ("BilledCost", "Billing amount", "Billing data only"),
        ("ChargePeriodStart", "Charge period start", "Billing period timestamp, not service telemetry"),
        ("ChargePeriodEnd", "Charge period end", "Billing period timestamp, not service telemetry"),
        ("ResourceId", "Resource identifier", "Resource grouping, not proof of SLA performance"),
        ("ServiceName", "Service name", "Service grouping, not proof of SLA performance"),
    ]

    basic_rows = []
    for field, description, interpretation in basic_fields:
        present = field in df.columns
        non_null = int(df[field].notna().sum()) if present else 0
        basic_rows.append({
            "Field": field,
            "Description": description,
            "Present": bool(present),
            "NonNull_Rows": non_null,
            "Total_Rows": int(len(df)),
            "Interpretation": interpretation,
        })

    pd.DataFrame(basic_rows).to_csv(
        OUT / "available_billing_evidence.csv", index=False
    )

    # Service-level statuses: no service is certified compliant merely
    # because billing records exist.
    service_field = "ServiceName"
    if service_field in df.columns:
        services = (
            df.assign(
                ServiceName=df[service_field].fillna("Unknown").astype(str),
                BilledCost=pd.to_numeric(df["BilledCost"], errors="coerce")
                if "BilledCost" in df.columns else np.nan,
            )
            .groupby("ServiceName", dropna=False)
            .agg(
                Billing_Records=("ServiceName", "size"),
                Net_BilledCost=("BilledCost", "sum"),
            )
            .reset_index()
        )
        services["SLA_Assessment_Status"] = "NOT_VERIFIABLE"
        services["Reason"] = (
            "Billing data does not establish measured uptime, latency, "
            "error rate or a contractual SLA target."
        )
        services.to_csv(OUT / "service_sla_evidence_status.csv", index=False)

    # Explicit overall assessment.
    overall = pd.DataFrame([{
        "Dataset": "Microsoft FinOps Toolkit demo FOCUS dataset",
        "Rows_Assessed": int(len(df)),
        "SLA_Compliance_Status": "NOT_VERIFIABLE",
        "Operational_SLA_Metrics_Verified": 0,
        "SLA_Targets_Verified": 0,
        "Automatic_Optimization_Approval": "NOT_AUTHORIZED",
        "Reason": (
            "Billing data alone does not provide verified operational "
            "measurements and contractual targets needed to determine SLA compliance."
        ),
    }])
    overall.to_csv(OUT / "sla_assessment_summary.csv", index=False)

    status_counts = (
        evidence["Status"].value_counts()
        .rename_axis("Status")
        .reset_index(name="Evidence_Categories")
    )
    status_counts.to_csv(OUT / "sla_evidence_status_counts.csv", index=False)

    # Graph 1: how many evidence categories are missing versus present.
    plot_counts = (
        evidence["Status"].value_counts()
        .reindex([
            "MISSING_FROM_DATASET",
            "FIELD_PRESENT_BUT_EMPTY",
            "FIELD_PRESENT_REQUIRES_VALIDATION",
        ], fill_value=0)
    )

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(plot_counts.index, plot_counts.values)
    ax.set_title("External Dataset: SLA Evidence Availability")
    ax.set_ylabel("Number of evidence categories")
    ax.set_xlabel("Evidence status")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    for i, value in enumerate(plot_counts.values):
        ax.text(i, value, str(int(value)), ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(OUT / "figure_01_sla_evidence_availability.png", dpi=180)
    plt.close(fig)

    # Graph 2: each required evidence category and whether it is missing.
    plot_evidence = evidence.copy()
    plot_evidence["Available_Field"] = (
        plot_evidence["Status"] == "FIELD_PRESENT_REQUIRES_VALIDATION"
    )
    plot_evidence = plot_evidence.sort_values("Available_Field")
    fig, ax = plt.subplots(figsize=(11, 7))
    values = plot_evidence["Available_Field"].astype(int)
    ax.barh(plot_evidence["Evidence_Category"], values)
    ax.set_title("SLA and Operational Evidence by Category")
    ax.set_xlabel("1 = matching field found; 0 = no recognized field found")
    ax.set_ylabel("Evidence category")
    ax.set_xticks([0, 1], labels=["Missing", "Field found"])
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "figure_02_sla_evidence_by_category.png", dpi=180)
    plt.close(fig)

    summary = {
        "dataset": "Microsoft FinOps Toolkit demo FOCUS dataset",
        "records_assessed": int(len(df)),
        "overall_sla_compliance_status": "NOT_VERIFIABLE",
        "evidence_categories_assessed": int(len(evidence)),
        "categories_missing": int(
            (evidence["Status"] == "MISSING_FROM_DATASET").sum()
        ),
        "categories_with_empty_fields": int(
            (evidence["Status"] == "FIELD_PRESENT_BUT_EMPTY").sum()
        ),
        "categories_with_matching_fields_requiring_validation": int(
            (evidence["Status"] == "FIELD_PRESENT_REQUIRES_VALIDATION").sum()
        ),
        "automatic_optimization_approval": "NOT_AUTHORIZED",
        "interpretation": (
            "This is an evidence availability assessment, not an SLA compliance test. "
            "A matching field name is not sufficient to verify measurement meaning, "
            "units, provenance, contractual targets or compliance."
        ),
        "limitations": [
            "Billing charges and billing-period dates are not operational SLA measurements.",
            "No compliance percentage is calculated without validated observations and targets.",
            "Service-level statuses remain NOT_VERIFIABLE unless sufficient validated evidence is supplied.",
            "Before optimizing a production FinTech service, validate uptime, latency, error rates, "
            "workload, resource utilization, contractual targets and rollback procedures."
        ],
    }

    with open(OUT / "sla_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    index = pd.DataFrame([
        ["sla_evidence_matrix.csv", "Required evidence, matched fields and status"],
        ["available_billing_evidence.csv", "Billing fields available in the dataset"],
        ["service_sla_evidence_status.csv", "Per-service SLA verification status"],
        ["sla_assessment_summary.csv", "Overall SLA status and approval guardrail"],
        ["sla_evidence_status_counts.csv", "Count of evidence categories by status"],
        ["figure_01_sla_evidence_availability.png", "Evidence availability status graph"],
        ["figure_02_sla_evidence_by_category.png", "Evidence presence by category"],
        ["sla_summary.json", "Assessment methods and limitations"],
    ], columns=["File", "Description"])
    index.to_csv(OUT / "output_index.csv", index=False)

    print("\n========== EXTERNAL SLA EVIDENCE ASSESSMENT COMPLETE ==========")
    print(f"Output folder: {OUT}")
    print(f"Records assessed: {len(df):,}")
    print(f"Overall SLA status: NOT_VERIFIABLE")
    print("\nEvidence matrix:")
    print(evidence[
        ["Evidence_Category", "Status", "Matched_Columns"]
    ].to_string(index=False))
    print("\nGenerated files:")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name}")


if __name__ == "__main__":
    main()
