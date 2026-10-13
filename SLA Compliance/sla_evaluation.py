from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "sla_train_compliance_results.csv"
TEST_FILE = DATA_DIR / "sla_test_compliance_results.csv"

OUTPUT_FILE = DATA_DIR / "sla_evaluation_summary.csv"
ISSUES_FILE = DATA_DIR / "sla_evaluation_issues.csv"


# ============================================================
# EVALUATION CONFIGURATION
# ============================================================

REQUIRED_COLUMNS = [
    "Resource_ID",
    "CPU_Utilization",
    "Memory_Utilization",
    "SLA_Evidence_Status",
    "Risk_Assessment_Score",
    "Risk_Assessment_Band",
    "Optimization_Decision",
    "Risk_Assessment_Priority",
    "SLA_Certification_Allowed",
    "Automatic_Modification_Allowed",
    "Manual_Approval_Required",
    "Confirmed_SLA_Violation",
]

VALID_SLA_STATUSES = {
    "COMPLIANT",
    "VIOLATION_DETECTED",
    "NOT_VERIFIABLE",
}

VALID_RISK_BANDS = {
    "LOW",
    "MEDIUM",
    "HIGH",
    "UNKNOWN",
}

VALID_DECISIONS = {
    "BLOCKED",
    "REVIEW_REQUIRED",
    "PROVISIONAL_CANDIDATE",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def add_issue(issues, dataset, check, count, details):
    if count > 0:
        issues.append({
            "Dataset": dataset,
            "Check": check,
            "Issue_Count": int(count),
            "Details": details,
        })


def safe_numeric(series):
    return pd.to_numeric(series, errors="coerce")


# ============================================================
# EVALUATE ONE DATASET
# ============================================================

def evaluate_dataset(file_path, dataset_name):
    print(f"\nEvaluating {dataset_name} dataset...")

    if not file_path.exists():
        raise FileNotFoundError(
            f"Required file not found: {file_path}\n"
            "Run sla_compliance.py first."
        )

    df = pd.read_csv(file_path)
    issues = []

    # --------------------------------------------------------
    # 1. Required columns
    # --------------------------------------------------------

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing required columns: "
            f"{missing_columns}"
        )

    total_rows = len(df)

    # --------------------------------------------------------
    # 2. Resource identifiers
    # --------------------------------------------------------

    missing_ids = (
        df["Resource_ID"].isna()
        | df["Resource_ID"].astype(str).str.strip().eq("")
    ).sum()

    duplicate_ids = df["Resource_ID"].duplicated().sum()

    add_issue(
        issues, dataset_name,
        "Missing resource identifiers",
        missing_ids,
        "Each resource should have a non-empty identifier."
    )

    add_issue(
        issues, dataset_name,
        "Duplicate resource identifiers",
        duplicate_ids,
        "Check whether repeated resource IDs represent "
        "duplicate records or legitimate repeated observations."
    )

    # --------------------------------------------------------
    # 3. SLA status validation
    # --------------------------------------------------------

    sla_status = (
        df["SLA_Evidence_Status"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_sla_status = (~sla_status.isin(
        VALID_SLA_STATUSES
    )).sum()

    not_verifiable = sla_status.eq("NOT_VERIFIABLE").sum()

    add_issue(
        issues, dataset_name,
        "Invalid SLA status",
        invalid_sla_status,
        "SLA status must be COMPLIANT, VIOLATION_DETECTED, "
        "or NOT_VERIFIABLE."
    )

    # --------------------------------------------------------
    # 4. Risk score validation
    # --------------------------------------------------------

    scores = safe_numeric(df["Risk_Assessment_Score"])

    invalid_scores = (
        scores.isna()
        | (scores < 0)
        | (scores > 100)
    ).sum()

    add_issue(
        issues, dataset_name,
        "Invalid risk scores",
        invalid_scores,
        "Risk scores must be numeric and between 0 and 100."
    )

    bands = (
        df["Risk_Assessment_Band"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_bands = (~bands.isin(VALID_RISK_BANDS)).sum()

    add_issue(
        issues, dataset_name,
        "Invalid risk bands",
        invalid_bands,
        "Risk bands must be LOW, MEDIUM, HIGH, or UNKNOWN."
    )

    # --------------------------------------------------------
    # 5. Optimization decision validation
    # --------------------------------------------------------

    decisions = (
        df["Optimization_Decision"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    invalid_decisions = (
        ~decisions.isin(VALID_DECISIONS)
    ).sum()

    add_issue(
        issues, dataset_name,
        "Invalid optimization decisions",
        invalid_decisions,
        "Decision must be BLOCKED, REVIEW_REQUIRED, "
        "or PROVISIONAL_CANDIDATE."
    )

    # --------------------------------------------------------
    # 6. Safety control validation
    # --------------------------------------------------------

    def as_boolean(series):
        return (
            series.astype(str)
            .str.strip()
            .str.lower()
            .map({
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            })
        )

    certification = as_boolean(
        df["SLA_Certification_Allowed"]
    )

    automatic_changes = as_boolean(
        df["Automatic_Modification_Allowed"]
    )

    manual_approval = as_boolean(
        df["Manual_Approval_Required"]
    )

    bad_certification = (
        certification.isna() | certification
    ).sum()

    bad_automatic_changes = (
        automatic_changes.isna() | automatic_changes
    ).sum()

    missing_manual_approval = (
        manual_approval.isna() | ~manual_approval
    ).sum()

    add_issue(
        issues, dataset_name,
        "Unsafe or invalid SLA certification setting",
        bad_certification,
        "Certification must remain disabled because actual "
        "SLA evidence is unavailable."
    )

    add_issue(
        issues, dataset_name,
        "Automatic modification enabled or invalid",
        bad_automatic_changes,
        "Automatic resource modifications must remain disabled."
    )

    add_issue(
        issues, dataset_name,
        "Manual approval missing or invalid",
        missing_manual_approval,
        "Manual approval must be required for every resource."
    )

    # A resource cannot be certified when evidence is unavailable.
    certified_without_evidence = (
        certification.fillna(False)
        & sla_status.eq("NOT_VERIFIABLE")
    ).sum()

    add_issue(
        issues, dataset_name,
        "Certification without verifiable SLA evidence",
        certified_without_evidence,
        "A resource with unavailable SLA evidence must not "
        "be certified as compliant."
    )

    # --------------------------------------------------------
    # 7. Confirmed SLA violation claims
    # --------------------------------------------------------

    violation_claims = (
        df["Confirmed_SLA_Violation"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "YES", "1", "CONFIRMED"])
    ).sum()

    add_issue(
        issues, dataset_name,
        "Unsupported confirmed SLA violation claims",
        violation_claims,
        "The dataset does not contain contractual SLA targets "
        "or actual availability/response-time measurements."
    )

    # --------------------------------------------------------
    # 8. Utilization data quality
    # --------------------------------------------------------

    cpu = safe_numeric(df["CPU_Utilization"])
    memory = safe_numeric(df["Memory_Utilization"])

    invalid_cpu = (
        cpu.notna() & ((cpu < 0) | (cpu > 100))
    ).sum()

    invalid_memory = (
        memory.notna() & ((memory < 0) | (memory > 100))
    ).sum()

    add_issue(
        issues, dataset_name,
        "CPU utilization outside 0-100",
        invalid_cpu,
        "Check CPU utilization units and source values."
    )

    add_issue(
        issues, dataset_name,
        "Memory utilization outside 0-100",
        invalid_memory,
        "Check memory utilization units and source values."
    )

    # --------------------------------------------------------
    # 9. Evaluation summary
    # --------------------------------------------------------

    decision_counts = decisions.value_counts()
    risk_counts = bands.value_counts()

    valid_scores = scores.dropna()

    issue_count = sum(
        issue["Issue_Count"] for issue in issues
    )

    summary = {
        "Dataset": dataset_name,
        "Total_Resources": total_rows,
        "Missing_Resource_IDs": int(missing_ids),
        "Duplicate_Resource_IDs": int(duplicate_ids),
        "Missing_CPU_Values": int(cpu.isna().sum()),
        "Missing_Memory_Values": int(memory.isna().sum()),
        "Invalid_CPU_Values": int(invalid_cpu),
        "Invalid_Memory_Values": int(invalid_memory),
        "Invalid_SLA_Statuses": int(invalid_sla_status),
        "NOT_VERIFIABLE_Count": int(not_verifiable),
        "Invalid_Risk_Scores": int(invalid_scores),
        "Invalid_Risk_Bands": int(invalid_bands),
        "Invalid_Optimization_Decisions": int(invalid_decisions),
        "Mean_Risk_Score": (
            round(valid_scores.mean(), 2)
            if not valid_scores.empty else np.nan
        ),
        "Median_Risk_Score": (
            round(valid_scores.median(), 2)
            if not valid_scores.empty else np.nan
        ),
        "LOW_Risk_Count": int(risk_counts.get("LOW", 0)),
        "MEDIUM_Risk_Count": int(risk_counts.get("MEDIUM", 0)),
        "HIGH_Risk_Count": int(risk_counts.get("HIGH", 0)),
        "UNKNOWN_Risk_Count": int(risk_counts.get("UNKNOWN", 0)),
        "BLOCKED_Count": int(decision_counts.get("BLOCKED", 0)),
        "REVIEW_REQUIRED_Count": int(
            decision_counts.get("REVIEW_REQUIRED", 0)
        ),
        "PROVISIONAL_CANDIDATE_Count": int(
            decision_counts.get("PROVISIONAL_CANDIDATE", 0)
        ),
        "Certification_Allowed_Count": int(
            certification.fillna(False).sum()
        ),
        "Automatic_Modification_Allowed_Count": int(
            automatic_changes.fillna(False).sum()
        ),
        "Manual_Approval_Required_Count": int(
            manual_approval.fillna(False).sum()
        ),
        "Confirmed_Violation_Claims": int(violation_claims),
        "Evaluation_Issue_Count": int(issue_count),
        "Evaluation_Result": (
            "PASS" if issue_count == 0 else "REVIEW_REQUIRED"
        ),
        "Accuracy_Metrics": "NOT_AVAILABLE_NO_GROUND_TRUTH",
    }

    print(f"  Resources evaluated: {total_rows}")
    print(f"  NOT_VERIFIABLE: {not_verifiable}")
    print(f"  Mean risk score: {summary['Mean_Risk_Score']}")
    print(f"  Evaluation issues: {issue_count}")
    print(f"  Result: {summary['Evaluation_Result']}")

    return summary, issues


# ============================================================
# MAIN
# ============================================================

def main():
    all_summaries = []
    all_issues = []

    datasets = [
        (TRAIN_FILE, "TRAIN"),
        (TEST_FILE, "TEST"),
    ]

    for file_path, dataset_name in datasets:
        summary, issues = evaluate_dataset(
            file_path, dataset_name
        )
        all_summaries.append(summary)
        all_issues.extend(issues)

    summary_df = pd.DataFrame(all_summaries)
    issues_df = pd.DataFrame(
        all_issues,
        columns=[
            "Dataset",
            "Check",
            "Issue_Count",
            "Details",
        ],
    )

    summary_df.to_csv(OUTPUT_FILE, index=False)
    issues_df.to_csv(ISSUES_FILE, index=False)

    print("\n" + "=" * 60)
    print("SLA EVALUATION COMPLETED")
    print("=" * 60)
    print(f"Summary saved to: {OUTPUT_FILE}")
    print(f"Issues saved to: {ISSUES_FILE}")

    print("\nEvaluation summary:")
    print(
        summary_df[
            [
                "Dataset",
                "Total_Resources",
                "NOT_VERIFIABLE_Count",
                "Evaluation_Issue_Count",
                "Evaluation_Result",
            ]
        ].to_string(index=False)
    )

    if issues_df.empty:
        print("\nNo evaluation issues detected.")
    else:
        print("\nIssues requiring attention:")
        print(issues_df.to_string(index=False))

    print(
        "\nIMPORTANT: This evaluates data quality and safety "
        "controls, not predictive accuracy or actual SLA "
        "compliance. No ground-truth SLA labels are available."
    )


if __name__ == "__main__":
    main()
