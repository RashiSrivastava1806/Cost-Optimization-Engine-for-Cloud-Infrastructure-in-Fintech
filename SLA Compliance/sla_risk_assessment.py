
"""
SLA Risk Assessment Module
--------------------------
1. Calculate an explainable operational risk score.
2. Assign risk bands and review priorities.
3. Identify potential optimization opportunities.
4. Preserve the distinction between risk and SLA compliance.

Risk scores are heuristic indicators, not probabilities.
No cloud resources are modified automatically.
"""

from pathlib import Path

import pandas as pd

from sla_config import (
    PROJECT_ROOT,
    DATA_DIR,
    TRAIN_INPUT_FILE,
    TEST_INPUT_FILE,
    MISSING_SLA_EVIDENCE_STATUS,
)

from sla_rule_engine import apply_sla_rules


# =========================================================
# 1. OUTPUT PATHS
# =========================================================

TRAIN_OUTPUT_FILE = DATA_DIR / "sla_train_risk_assessment.csv"
TEST_OUTPUT_FILE = DATA_DIR / "sla_test_risk_assessment.csv"
SUMMARY_OUTPUT_FILE = DATA_DIR / "sla_risk_assessment_summary.csv"


# =========================================================
# 2. CONFIGURABLE THRESHOLDS AND WEIGHTS
# =========================================================

CPU_HIGH_THRESHOLD = 85.0
CPU_REVIEW_THRESHOLD = 70.0

MEMORY_HIGH_THRESHOLD = 85.0
MEMORY_REVIEW_THRESHOLD = 75.0

CPU_HIGH_POINTS = 30
CPU_REVIEW_POINTS = 15

MEMORY_HIGH_POINTS = 30
MEMORY_REVIEW_POINTS = 15

ANOMALY_POINTS = 20
HIGH_COST_POINTS = 15
MISSING_UTILIZATION_POINTS = 20

HIGH_RISK_THRESHOLD = 50
MEDIUM_RISK_THRESHOLD = 25


# =========================================================
# 3. HELPER FUNCTIONS
# =========================================================

def find_column(df, possible_names):
    """Find a column without depending on capitalization."""

    normalized_columns = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:
        match = normalized_columns.get(name.strip().lower())

        if match is not None:
            return match

    return None


def numeric_column(df, possible_names):
    """Return numeric values, or missing values if unavailable."""

    column = find_column(df, possible_names)

    if column is None:
        return pd.Series(float("nan"), index=df.index)

    return pd.to_numeric(df[column], errors="coerce")


def boolean_column(df, possible_names):
    """Convert Boolean, numeric, and common text flags safely."""

    column = find_column(df, possible_names)

    if column is None:
        return pd.Series(False, index=df.index, dtype=bool)

    values = df[column]

    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False)

    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).ne(0)

    normalized = values.astype("string").str.strip().str.lower()

    return normalized.isin(
        ["1", "true", "yes", "y"]
    ).fillna(False)


def get_risk_band(score):
    """Convert the heuristic score into a risk classification."""

    if pd.isna(score):
        return "UNKNOWN"

    if score >= HIGH_RISK_THRESHOLD:
        return "HIGH"

    if score >= MEDIUM_RISK_THRESHOLD:
        return "MEDIUM"

    return "LOW"


def get_priority(decision, risk_band, low_utilization):
    """
    Assign an investigation priority.

    Blocked resources are explicitly distinguished from resources
    classified as high risk by their numerical risk score.
    """

    if decision == "BLOCKED":
        return "P1 - Blocked / Critical Review"

    if decision == "REVIEW_REQUIRED":
        return "P2 - Manual Review"

    if risk_band == "HIGH":
        return "P1 - High-Risk Review"

    if risk_band == "MEDIUM":
        return "P2 - Manual Review"

    if low_utilization:
        return "P3 - Optimization Assessment"

    return "P4 - Routine Monitoring"


# =========================================================
# 4. RESOURCE RISK ASSESSMENT
# =========================================================

def assess_resource_risk(df):
    """Calculate risk scores, reasons, and review priorities."""

    result = apply_sla_rules(df.copy())

    # Read utilization metrics.
    cpu = numeric_column(result, ["CPU_Utilization"])
    memory = numeric_column(result, ["Memory_Utilization"])

    cpu_valid = cpu.between(0, 100)
    memory_valid = memory.between(0, 100)

    # Either metric can provide some utilization evidence.
    # Missing or invalid individual metrics are not treated as
    # high utilization.
    utilization_available = cpu_valid | memory_valid
    missing_utilization = ~utilization_available

    # Detect anomalies by combining all available anomaly flags.
    anomaly_columns = [
        column
        for column in [
            "Anomaly_Flag",
            "Contextual_Anomaly_Flag",
            "Statistical_Outlier_Flag",
        ]
        if column in result.columns
    ]

    anomaly = pd.Series(False, index=result.index, dtype=bool)

    for column in anomaly_columns:
        anomaly |= boolean_column(result, [column])

    high_cost = boolean_column(
        result, ["SLA_High_Cost_Signal"]
    )

    low_utilization = boolean_column(
        result, ["SLA_Low_Utilization_Signal"]
    )

    # -----------------------------------------------------
    # Calculate an explainable heuristic risk score.
    # Low utilization is deliberately not a risk penalty.
    # -----------------------------------------------------

    scores = pd.Series(0, index=result.index, dtype="int64")
    reasons = [[] for _ in range(len(result))]

    def add_points(mask, points, reason):
        """Add points and record the corresponding explanation."""

        nonlocal scores

        mask = mask.fillna(False).astype(bool)
        scores.loc[mask] += points

        for position, matched in enumerate(mask.tolist()):
            if matched:
                reasons[position].append(reason)

    add_points(
        cpu_valid & cpu.ge(CPU_HIGH_THRESHOLD),
        CPU_HIGH_POINTS,
        "CPU utilization is at or above the high threshold",
    )

    add_points(
        cpu_valid
        & cpu.ge(CPU_REVIEW_THRESHOLD)
        & cpu.lt(CPU_HIGH_THRESHOLD),
        CPU_REVIEW_POINTS,
        "CPU utilization requires review",
    )

    add_points(
        memory_valid & memory.ge(MEMORY_HIGH_THRESHOLD),
        MEMORY_HIGH_POINTS,
        "Memory utilization is at or above the high threshold",
    )

    add_points(
        memory_valid
        & memory.ge(MEMORY_REVIEW_THRESHOLD)
        & memory.lt(MEMORY_HIGH_THRESHOLD),
        MEMORY_REVIEW_POINTS,
        "Memory utilization requires review",
    )

    add_points(
        anomaly,
        ANOMALY_POINTS,
        "Anomaly signal requires investigation",
    )

    add_points(
        high_cost,
        HIGH_COST_POINTS,
        "Cost is high relative to its benchmark",
    )

    add_points(
        missing_utilization,
        MISSING_UTILIZATION_POINTS,
        "Utilization evidence is missing or invalid",
    )

    scores = scores.clip(upper=100)

    # If neither utilization metric is usable, the overall
    # operational risk classification is incomplete.
    risk_bands = scores.apply(get_risk_band)
    risk_bands.loc[missing_utilization] = "UNKNOWN"

    # -----------------------------------------------------
    # Attach assessment fields.
    # -----------------------------------------------------

    result["Risk_Assessment_Score"] = scores
    result["Risk_Assessment_Band"] = risk_bands

    result["Risk_Assessment_Reason"] = [
        "; ".join(row_reasons)
        if row_reasons
        else "No configured operational risk signals detected"
        for row_reasons in reasons
    ]

    result["Optimization_Opportunity_Flag"] = low_utilization

    result["Risk_Assessment_Priority"] = [
        get_priority(decision, band, low_util)
        for decision, band, low_util in zip(
            result["Optimization_Decision"],
            result["Risk_Assessment_Band"],
            low_utilization,
        )
    ]

    # The current dataset lacks the required contractual SLA
    # and performance evidence to certify compliance.
    result["SLA_Evidence_Status"] = (
        MISSING_SLA_EVIDENCE_STATUS
    )

    return result


# =========================================================
# 5. SUMMARY GENERATION
# =========================================================

def create_summary(result, dataset_name):
    """Create a summary of risk, decisions, and priorities."""

    risk_counts = result["Risk_Assessment_Band"].value_counts()
    decision_counts = result["Optimization_Decision"].value_counts()
    priority_counts = result["Risk_Assessment_Priority"].value_counts()

    return {
        "Dataset": dataset_name,
        "Total_Resources": len(result),

        "High_Risk": risk_counts.get("HIGH", 0),
        "Medium_Risk": risk_counts.get("MEDIUM", 0),
        "Low_Risk": risk_counts.get("LOW", 0),
        "Unknown_Risk": risk_counts.get("UNKNOWN", 0),

        "Optimization_Opportunities": int(
            result["Optimization_Opportunity_Flag"].sum()
        ),

        "Blocked": decision_counts.get("BLOCKED", 0),
        "Review_Required": decision_counts.get(
            "REVIEW_REQUIRED", 0
        ),
        "Provisional_Candidates": decision_counts.get(
            "PROVISIONAL_CANDIDATE", 0
        ),

        "SLA_Not_Verifiable": int(
            result["SLA_Evidence_Status"]
            .eq(MISSING_SLA_EVIDENCE_STATUS)
            .sum()
        ),

        "P1_Blocked_Critical_Review": priority_counts.get(
            "P1 - Blocked / Critical Review", 0
        ),
        "P1_High_Risk_Review": priority_counts.get(
            "P1 - High-Risk Review", 0
        ),
        "P2_Manual_Review": priority_counts.get(
            "P2 - Manual Review", 0
        ),
        "P3_Optimization_Assessment": priority_counts.get(
            "P3 - Optimization Assessment", 0
        ),
        "P4_Routine_Monitoring": priority_counts.get(
            "P4 - Routine Monitoring", 0
        ),
    }


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

    print("=" * 65)
    print("SLA RISK ASSESSMENT")
    print("=" * 65)

    summaries = []

    datasets = [
        ("TRAIN", train_path, TRAIN_OUTPUT_FILE),
        ("TEST", test_path, TEST_OUTPUT_FILE),
    ]

    for dataset_name, input_path, output_path in datasets:
        print(f"\nProcessing {dataset_name} dataset...")

        raw_df = pd.read_csv(input_path)
        assessed_df = assess_resource_risk(raw_df)

        assessed_df.to_csv(output_path, index=False)

        summary = create_summary(
            assessed_df,
            dataset_name,
        )
        summaries.append(summary)

        print(f"Input rows: {len(raw_df)}")
        print(f"Output file: {output_path}")

        print("\nRisk assessment bands:")
        print(
            assessed_df["Risk_Assessment_Band"]
            .value_counts()
            .to_string()
        )

        print("\nOptimization decisions:")
        print(
            assessed_df["Optimization_Decision"]
            .value_counts()
            .to_string()
        )

        print("\nReview priorities:")
        print(
            assessed_df["Risk_Assessment_Priority"]
            .value_counts()
            .to_string()
        )

        print(
            "\nOptimization opportunities:",
            int(
                assessed_df[
                    "Optimization_Opportunity_Flag"
                ].sum()
            ),
        )

        print(
            "SLA compliance status:",
            assessed_df["SLA_Compliance_Status"]
            .value_counts()
            .to_dict(),
        )

    summary_df = pd.DataFrame(summaries)
    summary_df.to_csv(SUMMARY_OUTPUT_FILE, index=False)

    print("\n" + "=" * 65)
    print("RISK ASSESSMENT SUMMARY")
    print("=" * 65)

    print(summary_df.to_string(index=False))
    print(f"\nSummary saved to: {SUMMARY_OUTPUT_FILE}")

    print("\nRisk assessment completed successfully.")
    print(
        "Note: Scores are heuristic indicators, not probabilities "
        "or confirmed SLA violations."
    )


if __name__ == "__main__":
    main()
