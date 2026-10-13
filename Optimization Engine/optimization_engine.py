
from pathlib import Path
import json
import pandas as pd

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
OUTPUT = ROOT / "outputs" / "optimization"
OUTPUT.mkdir(parents=True, exist_ok=True)

INPUT_FILES = {
    "train": DATA / "sla_train_compliance_results.csv",
    "test": DATA / "sla_test_compliance_results.csv",
}


def is_true(value):
    return str(value).strip().lower() in {
        "true", "1", "1.0", "yes"
    }


def generate_recommendations(file_path, split_name):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing input file: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [
        "Resource_ID",
        "Total_Cost_INR",
    ]

    for column in required_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required column missing: {column}"
            )

    recommendations = []

    for _, row in df.iterrows():
        cost = pd.to_numeric(
            row["Total_Cost_INR"],
            errors="coerce"
        )

        if pd.isna(cost) or cost <= 0:
            continue

        cpu = pd.to_numeric(
            row.get("CPU_Utilization"),
            errors="coerce"
        )

        memory = pd.to_numeric(
            row.get("Memory_Utilization"),
            errors="coerce"
        )

        low_utilization = (
            pd.notna(cpu)
            and pd.notna(memory)
            and cpu < 30
            and memory < 30
        )

        anomaly = is_true(
            row.get("Anomaly_Flag", False)
        )

        candidate = is_true(
            row.get(
                "Potential_Optimization_Candidate",
                False
            )
        )

        investigation = is_true(
            row.get(
                "SLA_Anomaly_Investigation_Required",
                False
            )
        )

        if not (
            low_utilization
            or anomaly
            or candidate
            or investigation
        ):
            continue

        # Decide the recommended action.
        if anomaly or investigation:
            action = "Investigate anomaly before optimization"
            reason = (
                "An unusual cost or usage signal requires "
                "investigation before any resource change."
            )
            priority = "High"
            savings = 0.0

        elif pd.isna(cpu) or pd.isna(memory):
            action = "Collect utilization telemetry"
            reason = (
                "CPU or memory data is missing. "
                "Rightsizing cannot yet be justified."
            )
            priority = "Medium"
            savings = 0.0

        elif low_utilization:
            action = "Review resource for possible rightsizing"
            reason = (
                f"CPU utilization: {cpu:.1f}%; "
                f"memory utilization: {memory:.1f}%. "
                "Both are below the 30% screening threshold."
            )
            priority = "Medium"

            # Illustrative scenario, not guaranteed savings.
            savings = round(float(cost) * 0.10, 2)

        else:
            action = "Review service and regional cost benchmarks"
            reason = (
                "An optimization candidate was flagged. "
                "Validate workload demand, pricing and "
                "performance requirements before acting."
            )
            priority = "Low"
            savings = 0.0

        sla_status = row.get(
            "SLA_Compliance_Status",
            "NOT_VERIFIABLE"
        )

        risk = row.get(
            "SLA_Risk_Level",
            row.get("Risk_Assessment_Band", "UNKNOWN")
        )

        evidence = row.get(
            "SLA_Evidence_Status",
            "UNKNOWN"
        )

        block_reason = row.get(
            "Optimization_Block_Reason",
            ""
        )

        # SLA compliance must not be assumed.
        sla_unverified = (
            "not_verifiable" in str(sla_status).lower()
            or "unknown" in str(evidence).lower()
            or "unavailable" in str(evidence).lower()
        )

        recommendations.append({
            "Split": split_name,
            "Resource_ID": row["Resource_ID"],
            "Service_Name": row.get(
                "Service_Name", "Unknown"
            ),
            "Region_Zone": row.get(
                "Region_Zone", "Unknown"
            ),
            "Current_Cost_INR": round(float(cost), 2),
            "CPU_Utilization": cpu,
            "Memory_Utilization": memory,
            "SLA_Compliance_Status": sla_status,
            "SLA_Evidence_Status": evidence,
            "SLA_Risk_Level": risk,
            "Recommended_Action": action,
            "Priority": priority,
            "Reason": reason,
            "Illustrative_10pct_Savings_INR": savings,
            "Savings_Guaranteed": False,
            "SLA_Evidence_Unverified": sla_unverified,
            "Existing_Block_Reason": block_reason,
            "Manual_Approval_Required": True,
            "Automatic_Modification_Allowed": False,
        })

    return pd.DataFrame(recommendations)


# Generate recommendations separately for train and test data.
all_recommendations = []

for split_name, file_path in INPUT_FILES.items():
    result = generate_recommendations(
        file_path,
        split_name
    )

    output_file = (
        OUTPUT / f"recommendations_{split_name}.csv"
    )

    result.to_csv(output_file, index=False)
    all_recommendations.append(result)

    print(
        f"{split_name}: {len(result)} recommendations "
        f"saved to {output_file}"
    )


# Combine both sets of recommendations.
combined = pd.concat(
    all_recommendations,
    ignore_index=True
)

# Explicit priority order: High, Medium, Low.
priority_order = {
    "High": 0,
    "Medium": 1,
    "Low": 2,
}

if not combined.empty:
    combined["_priority_order"] = (
        combined["Priority"].map(priority_order)
    )

    combined = combined.sort_values(
        ["_priority_order", "Current_Cost_INR"],
        ascending=[True, False]
    ).drop(columns="_priority_order")

combined.to_csv(
    OUTPUT / "all_recommendations.csv",
    index=False
)

# Summary metrics.
summary = {
    "total_recommendations": len(combined),
    "high_priority": (
        int((combined["Priority"] == "High").sum())
        if not combined.empty else 0
    ),
    "medium_priority": (
        int((combined["Priority"] == "Medium").sum())
        if not combined.empty else 0
    ),
    "low_priority": (
        int((combined["Priority"] == "Low").sum())
        if not combined.empty else 0
    ),
    "manual_approval_required": True,
    "automatic_modifications_allowed": False,
    "illustrative_savings_scenario_INR": round(
        float(
            combined["Illustrative_10pct_Savings_INR"].sum()
        ),
        2
    ) if not combined.empty else 0.0,
    "important_note": (
        "Savings are illustrative estimates, not guaranteed "
        "or realized. SLA compliance must be independently "
        "verified before any resource change."
    ),
}

summary_file = OUTPUT / "optimization_summary.json"

summary_file.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nOPTIMIZATION SUMMARY")
print(json.dumps(summary, indent=2))
print("\nAll recommendations saved successfully.")
