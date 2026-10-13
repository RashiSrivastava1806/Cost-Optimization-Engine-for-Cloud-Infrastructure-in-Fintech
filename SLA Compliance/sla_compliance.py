
"""
SLA Compliance Integration Module
---------------------------------
Integrates the existing SLA rule engine and risk assessment.

Outputs:
1. Consolidated training assessment CSV
2. Consolidated testing assessment CSV
3. Overall compliance and optimization summary CSV
4. Decision audit CSV

Important:
- Missing SLA evidence means NOT_VERIFIABLE.
- Risk scores are heuristic indicators, not probabilities.
- No cloud resources are modified automatically.
"""

import pandas as pd

from sla_config import (
    PROJECT_ROOT,
    DATA_DIR,
    TRAIN_INPUT_FILE,
    TEST_INPUT_FILE,
    MISSING_SLA_EVIDENCE_STATUS,
    AUTOMATIC_RESOURCE_MODIFICATION_ENABLED,
    REQUIRE_MANUAL_APPROVAL,
    REQUIRE_SLA_EVIDENCE_FOR_CERTIFICATION,
)

from sla_data_validation import validate_dataset
from sla_risk_assessment import (
    assess_resource_risk,
    create_summary,
)


# =========================================================
# 1. OUTPUT FILES
# =========================================================

TRAIN_OUTPUT_FILE = DATA_DIR / "sla_train_compliance_results.csv"
TEST_OUTPUT_FILE = DATA_DIR / "sla_test_compliance_results.csv"
SUMMARY_OUTPUT_FILE = DATA_DIR / "sla_compliance_summary.csv"
AUDIT_OUTPUT_FILE = DATA_DIR / "sla_decision_audit.csv"


# =========================================================
# 2. VALIDATION ADAPTER
# =========================================================

def validate_input_dataset(df, dataset_name):
    """
    Run the existing validation module when its public
    interface supports a DataFrame input.

    The integration also performs essential checks locally
    so that this module remains explicit about its requirements.
    """

    result = df.copy()

    required_columns = [
        "Resource_ID",
        "Service_Name",
        "CPU_Utilization",
        "Memory_Utilization",
        "Usage_Start_Date",
        "Usage_End_Date",
        "Total_Cost_INR",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: missing required columns: "
            f"{missing_columns}"
        )

    if result["Resource_ID"].isna().any():
        raise ValueError(
            f"{dataset_name}: Resource_ID contains missing values."
        )

    if result["Resource_ID"].astype(str).str.strip().eq("").any():
        raise ValueError(
            f"{dataset_name}: Resource_ID contains blank values."
        )

    # Each row represents a resource usage record. Do not silently
    # discard duplicates, because that would change the dataset.
    duplicate_mask = result["Resource_ID"].duplicated(keep=False)

    if duplicate_mask.any():
        print(
            f"WARNING: {dataset_name} contains "
            f"{int(duplicate_mask.sum())} rows with duplicate "
            "Resource_ID values. They will be retained."
        )

    print(
        f"{dataset_name} integration checks passed: "
        f"{len(result)} rows, {len(result.columns)} columns."
    )

    return result


# =========================================================
# 3. CONSOLIDATED ASSESSMENT
# =========================================================

def run_compliance_assessment(df, dataset_name):
    """Validate, assess risk, and prepare consolidated results."""

    validated_df = validate_input_dataset(df, dataset_name)

    # This function calls the rule engine internally.
    assessed_df = assess_resource_risk(validated_df)

    # Explicitly record the current evidence limitation.
    assessed_df["SLA_Evidence_Status"] = (
        MISSING_SLA_EVIDENCE_STATUS
    )

    assessed_df["SLA_Certification_Allowed"] = False

    assessed_df["Automatic_Modification_Allowed"] = (
        AUTOMATIC_RESOURCE_MODIFICATION_ENABLED
    )

    assessed_df["Manual_Approval_Required"] = (
        REQUIRE_MANUAL_APPROVAL
    )

    assessed_df["SLA_Evidence_Required_For_Certification"] = (
        REQUIRE_SLA_EVIDENCE_FOR_CERTIFICATION
    )

    assessed_df["Compliance_Interpretation"] = (
        "SLA compliance cannot be verified from available data"
    )

    # This field is deliberately not inferred from anomalies.
    assessed_df["Confirmed_SLA_Violation"] = "NOT_ESTABLISHED"

    return assessed_df


# =========================================================
# 4. DECISION AUDIT
# =========================================================

def create_decision_audit(result, dataset_name):
    """Create a compact, explainable record for each input row."""

    audit_columns = [
        "Resource_ID",
        "Service_Name",
        "Total_Cost_INR",
        "CPU_Utilization",
        "Memory_Utilization",
        "SLA_Compliance_Status",
        "SLA_Evidence_Status",
        "Confirmed_SLA_Violation",
        "Risk_Assessment_Score",
        "Risk_Assessment_Band",
        "Risk_Assessment_Reason",
        "Optimization_Decision",
        "Optimization_Block_Reason",
        "SLA_Risk_Reason",
        "Risk_Assessment_Priority",
        "Optimization_Opportunity_Flag",
        "Manual_Approval_Required",
        "Automatic_Modification_Allowed",
    ]

    available_columns = [
        column
        for column in audit_columns
        if column in result.columns
    ]

    audit = result[available_columns].copy()
    audit.insert(0, "Dataset", dataset_name)

    return audit


# =========================================================
# 5. COMPLIANCE SUMMARY
# =========================================================

def create_compliance_summary(result, dataset_name):
    """Summarize compliance status and operational decisions."""

    # Reuse the existing risk summary.
    summary = create_summary(result, dataset_name)

    status_counts = (
        result["SLA_Compliance_Status"]
        .value_counts()
        .to_dict()
    )

    summary.update({
        "Compliant": status_counts.get("COMPLIANT", 0),
        "Violation_Detected": status_counts.get(
            "VIOLATION_DETECTED", 0
        ),
        "Not_Verifiable": status_counts.get(
            "NOT_VERIFIABLE", 0
        ),
        "Confirmed_SLA_Violations": int(
            result["Confirmed_SLA_Violation"]
            .eq("CONFIRMED")
            .sum()
        ),
        "Certification_Allowed": int(
            result["SLA_Certification_Allowed"].sum()
        ),
        "Automatic_Modifications_Allowed": int(
            result["Automatic_Modification_Allowed"].sum()
        ),
        "Manual_Approval_Required": int(
            result["Manual_Approval_Required"].sum()
        ),
    })

    return summary


# =========================================================
# 6. MAIN EXECUTION
# =========================================================

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_path = PROJECT_ROOT / TRAIN_INPUT_FILE
    test_path = PROJECT_ROOT / TEST_INPUT_FILE

    for path in [train_path, test_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required input file not found: {path}"
            )

    print("=" * 70)
    print("SLA COMPLIANCE INTEGRATION WORKFLOW")
    print("=" * 70)

    all_summaries = []
    all_audits = []

    datasets = [
        ("TRAIN", train_path, TRAIN_OUTPUT_FILE),
        ("TEST", test_path, TEST_OUTPUT_FILE),
    ]

    for dataset_name, input_path, output_path in datasets:
        print(f"\nProcessing {dataset_name} dataset...")

        raw_df = pd.read_csv(input_path)

        result = run_compliance_assessment(
            raw_df,
            dataset_name,
        )

        result.to_csv(output_path, index=False)

        summary = create_compliance_summary(
            result,
            dataset_name,
        )
        all_summaries.append(summary)

        audit = create_decision_audit(
            result,
            dataset_name,
        )
        all_audits.append(audit)

        print(f"Input rows: {len(raw_df)}")
        print(f"Results saved to: {output_path}")

        print("\nSLA compliance status:")
        print(
            result["SLA_Compliance_Status"]
            .value_counts()
            .to_string()
        )

        print("\nOptimization decisions:")
        print(
            result["Optimization_Decision"]
            .value_counts()
            .to_string()
        )

        print("\nRisk bands:")
        print(
            result["Risk_Assessment_Band"]
            .value_counts()
            .to_string()
        )

        print("\nReview priorities:")
        print(
            result["Risk_Assessment_Priority"]
            .value_counts()
            .to_string()
        )

        print(
            "\nConfirmed SLA violations:",
            int(
                result["Confirmed_SLA_Violation"]
                .eq("CONFIRMED")
                .sum()
            ),
        )

    # Write combined summaries and audit records.
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(SUMMARY_OUTPUT_FILE, index=False)

    audit_df = pd.concat(
        all_audits,
        ignore_index=True,
    )
    audit_df.to_csv(AUDIT_OUTPUT_FILE, index=False)

    print("\n" + "=" * 70)
    print("FINAL COMPLIANCE SUMMARY")
    print("=" * 70)

    print(summary_df.to_string(index=False))

    print("\nGenerated files:")
    print(f"Training results: {TRAIN_OUTPUT_FILE}")
    print(f"Testing results:  {TEST_OUTPUT_FILE}")
    print(f"Summary:          {SUMMARY_OUTPUT_FILE}")
    print(f"Decision audit:   {AUDIT_OUTPUT_FILE}")

    print("\nWorkflow completed successfully.")
    print(
        "IMPORTANT: NOT_VERIFIABLE is not the same as compliant "
        "or non-compliant."
    )
    print(
        "No SLA violations have been established from the "
        "available evidence."
    )


if __name__ == "__main__":
    main()
