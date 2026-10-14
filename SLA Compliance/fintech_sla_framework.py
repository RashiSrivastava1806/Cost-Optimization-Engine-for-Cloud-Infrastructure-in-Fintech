
from pathlib import Path
import warnings

import numpy as np
import pandas as pd


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "canonical_run"

TRAIN_FILE = DATA_DIR / "engineered_train_cloud_billing_data.csv"
TEST_FILE = DATA_DIR / "engineered_test_cloud_billing_data.csv"

ANOMALY_TRAIN_FILE = DATA_DIR / "anomaly_train_cloud_billing_data.csv"
ANOMALY_TEST_FILE = DATA_DIR / "anomaly_test_cloud_billing_data.csv"

OUTPUT_TRAIN = DATA_DIR / "fintech_sla_train_results.csv"
OUTPUT_TEST = DATA_DIR / "fintech_sla_test_results.csv"
OUTPUT_SUMMARY = DATA_DIR / "fintech_sla_framework_summary.csv"


# ============================================================
# 2. RESEARCH THRESHOLDS
# These are proposed research thresholds, not universal SLAs.
# ============================================================

AVAILABILITY_TARGET_PERCENT = 99.9

CPU_BLOCK_THRESHOLD = 85.0
CPU_REVIEW_THRESHOLD = 70.0
CPU_LOW_THRESHOLD = 30.0

MEMORY_BLOCK_THRESHOLD = 85.0
MEMORY_REVIEW_THRESHOLD = 75.0
MEMORY_LOW_THRESHOLD = 40.0


# ============================================================
# 3. COLUMN DETECTION
# Accommodates common column naming variations.
# ============================================================

COLUMN_ALIASES = {
    "resource_id": [
        "Resource_ID", "ResourceID", "resource_id"
    ],
    "service": [
        "Service_Name", "Service", "service_name"
    ],
    "cost": [
        "Total_Cost_INR", "Total_Cost", "Rounded_Cost",
        "Unrounded_Cost", "Cost_INR"
    ],
    "cpu": [
        "CPU_Utilization", "CPU_Utilization_Percent",
        "CPU_Usage_Percent", "CPU_Usage"
    ],
    "memory": [
        "Memory_Utilization", "Memory_Utilization_Percent",
        "Memory_Usage_Percent", "Memory_Usage"
    ],
    "availability": [
        "Availability_Percent", "Uptime_Percent",
        "Availability", "Uptime"
    ],
}


def find_column(df, possible_names):
    """Find a column without depending on capitalization."""
    lookup = {str(col).strip().lower(): col for col in df.columns}

    for name in possible_names:
        if name.strip().lower() in lookup:
            return lookup[name.strip().lower()]

    return None


def to_numeric(series):
    """Convert numeric values safely."""
    return pd.to_numeric(series, errors="coerce")


def parse_boolean(series):
    """Convert common Boolean representations to True/False/NA."""
    mapping = {
        "true": True, "1": True, "yes": True, "y": True,
        "false": False, "0": False, "no": False, "n": False,
    }

    def convert(value):
        if pd.isna(value):
            return pd.NA
        if isinstance(value, (bool, np.bool_)):
            return bool(value)
        return mapping.get(str(value).strip().lower(), pd.NA)

    return series.map(convert).astype("boolean")


# ============================================================
# 4. OPTIONAL ANOMALY DATA
# ============================================================

def attach_anomaly_evidence(df, anomaly_file):
    """
    Merge row-level anomaly indicators when available.
    Does not merge aggregate summary tables.
    """
    if not anomaly_file.exists():
        warnings.warn(f"Anomaly file not found: {anomaly_file}")
        df["Detected_Anomaly"] = False
        df["Anomaly_Evidence_Available"] = False
        return df

    anomaly_df = pd.read_csv(anomaly_file)

    resource_col = find_column(
        anomaly_df, COLUMN_ALIASES["resource_id"]
    )

    if resource_col is None:
        warnings.warn("No Resource_ID in anomaly file; anomaly merge skipped.")
        df["Detected_Anomaly"] = False
        df["Anomaly_Evidence_Available"] = False
        return df

    # Only use row-level flag columns, not aggregate count columns.
    flag_cols = [
        col for col in anomaly_df.columns
        if any(token in col.lower() for token in [
            "anomaly_flag",
            "is_anomaly",
            "combined_anomaly",
            "statistical_anomaly_flag",
            "contextual_anomaly_flag",
        ])
    ]

    # Remove columns that appear to be aggregate count fields.
    flag_cols = [
        col for col in flag_cols
        if "count" not in col.lower()
        and "score" not in col.lower()
    ]

    if not flag_cols:
        warnings.warn(
            "No suitable row-level anomaly flags found. "
            "Anomaly evidence will be marked unavailable."
        )
        df["Detected_Anomaly"] = False
        df["Anomaly_Evidence_Available"] = False
        return df

    # Avoid duplicate columns and accidental row multiplication.
    selected = anomaly_df[[resource_col] + flag_cols].copy()
    selected = selected.rename(columns={resource_col: "_merge_resource_id"})

    if selected["_merge_resource_id"].duplicated().any():
        warnings.warn(
            "Duplicate Resource_ID values in anomaly file. "
            "Anomaly merge skipped to prevent row multiplication."
        )
        df["Detected_Anomaly"] = False
        df["Anomaly_Evidence_Available"] = False
        return df

    resource_col_df = find_column(df, COLUMN_ALIASES["resource_id"])
    if resource_col_df is None:
        df["Detected_Anomaly"] = False
        df["Anomaly_Evidence_Available"] = False
        return df

    # Normalize values to avoid mismatches caused by whitespace.
    df["_merge_resource_id"] = (
        df[resource_col_df].astype("string").str.strip()
    )
    selected["_merge_resource_id"] = (
        selected["_merge_resource_id"].astype("string").str.strip()
    )

    # Rename incoming columns to avoid overwriting engineered features.
    rename_map = {
        col: f"_anomaly_{col}" for col in flag_cols
    }
    selected = selected.rename(columns=rename_map)

    df = df.merge(
        selected,
        on="_merge_resource_id",
        how="left",
        validate="many_to_one",
    )
    df.drop(columns=["_merge_resource_id"], inplace=True)

    anomaly_columns = list(rename_map.values())
    anomaly_values = []

    for col in anomaly_columns:
        anomaly_values.append(parse_boolean(df[col]))

    flags = pd.concat(anomaly_values, axis=1)
    df["Anomaly_Evidence_Available"] = flags.notna().any(axis=1)
    df["Detected_Anomaly"] = (
        flags.fillna(False).any(axis=1)
    ).astype(bool)

    df.drop(columns=anomaly_columns, inplace=True)

    return df


# ============================================================
# 5. EVALUATE EACH RESOURCE
# ============================================================

def evaluate_dataset(input_file, dataset_name, anomaly_file, output_file):
    if not input_file.exists():
        raise FileNotFoundError(
            f"Missing input file: {input_file}\n"
            "Run the canonical preprocessing and feature-engineering "
            "pipeline first."
        )

    df = pd.read_csv(input_file)

    resource_col = find_column(df, COLUMN_ALIASES["resource_id"])
    service_col = find_column(df, COLUMN_ALIASES["service"])
    cost_col = find_column(df, COLUMN_ALIASES["cost"])
    cpu_col = find_column(df, COLUMN_ALIASES["cpu"])
    memory_col = find_column(df, COLUMN_ALIASES["memory"])
    availability_col = find_column(
        df, COLUMN_ALIASES["availability"]
    )

    if cpu_col is None or memory_col is None:
        raise ValueError(
            f"{input_file} must contain CPU and memory utilization. "
            f"Available columns: {list(df.columns)}"
        )

    # Standardized evaluation columns.
    df["Evaluation_Dataset"] = dataset_name
    df["Evaluation_CPU"] = to_numeric(df[cpu_col])
    df["Evaluation_Memory"] = to_numeric(df[memory_col])

    df["Evaluation_Cost_INR"] = (
        to_numeric(df[cost_col]) if cost_col else np.nan
    )

    df = attach_anomaly_evidence(df, anomaly_file)

    cpu = df["Evaluation_CPU"]
    memory = df["Evaluation_Memory"]

    # --------------------------------------------------------
    # A. Data-quality checks
    # --------------------------------------------------------

    df["Utilization_Evidence_Available"] = (
        cpu.notna()
        & memory.notna()
        & cpu.between(0, 100)
        & memory.between(0, 100)
    )

    df["Invalid_Utilization_Data"] = (
        (cpu.notna() & ~cpu.between(0, 100))
        | (memory.notna() & ~memory.between(0, 100))
    )

    # --------------------------------------------------------
    # B. Research availability assessment
    # This is not contractual SLA certification.
    # --------------------------------------------------------

    if availability_col is not None:
        availability = to_numeric(df[availability_col])

        # Accept percentages expressed as either 99.9 or 0.999.
        availability = availability.where(
            availability > 1,
            availability * 100
        )

        df["Availability_Evidence_Available"] = availability.notna()
        df["Availability_Research_Target"] = (
            AVAILABILITY_TARGET_PERCENT
        )
        df["Research_Availability_Status"] = np.select(
            [
                availability.isna(),
                availability >= AVAILABILITY_TARGET_PERCENT,
            ],
            [
                "NOT_VERIFIABLE",
                "MEETS_RESEARCH_TARGET",
            ],
            default="BELOW_RESEARCH_TARGET",
        )
    else:
        df["Availability_Evidence_Available"] = False
        df["Availability_Research_Target"] = (
            AVAILABILITY_TARGET_PERCENT
        )
        df["Research_Availability_Status"] = "NOT_VERIFIABLE"

    # The dataset does not establish each resource's contract.
    # Do not equate a research target with contractual compliance.
    df["Contractual_SLA_Status"] = "NOT_VERIFIABLE"

    # --------------------------------------------------------
    # C. Risk and optimization safety gates
    # --------------------------------------------------------

    cpu_block = cpu >= CPU_BLOCK_THRESHOLD
    memory_block = memory >= MEMORY_BLOCK_THRESHOLD

    cpu_review = (
        (cpu >= CPU_REVIEW_THRESHOLD)
        & (cpu < CPU_BLOCK_THRESHOLD)
    )
    memory_review = (
        (memory >= MEMORY_REVIEW_THRESHOLD)
        & (memory < MEMORY_BLOCK_THRESHOLD)
    )

    low_cpu = cpu <= CPU_LOW_THRESHOLD
    low_memory = memory <= MEMORY_LOW_THRESHOLD

    missing_utilization = ~df["Utilization_Evidence_Available"]
    invalid_utilization = df["Invalid_Utilization_Data"]

    df["Low_CPU_Flag"] = low_cpu.fillna(False)
    df["Low_Memory_Flag"] = low_memory.fillna(False)
    df["CPU_Block_Flag"] = cpu_block.fillna(False)
    df["Memory_Block_Flag"] = memory_block.fillna(False)

    df["Optimization_Decision"] = np.select(
        [
            invalid_utilization,
            cpu_block | memory_block,
            missing_utilization,
            cpu_review | memory_review,
            df["Detected_Anomaly"],
            low_cpu & low_memory,
        ],
        [
            "REVIEW_REQUIRED",
            "BLOCKED",
            "REVIEW_REQUIRED",
            "REVIEW_REQUIRED",
            "REVIEW_REQUIRED",
            "PROVISIONAL_CANDIDATE",
        ],
        default="ROUTINE_MONITORING",
    )

    def decision_reason(row):
        reasons = []

        if row["Invalid_Utilization_Data"]:
            reasons.append("Utilization outside the valid 0-100% range")

        if pd.isna(row["Evaluation_CPU"]) or pd.isna(
            row["Evaluation_Memory"]
        ):
            reasons.append("CPU or memory evidence missing")

        if row["CPU_Block_Flag"]:
            reasons.append("CPU at or above blocking threshold")

        if row["Memory_Block_Flag"]:
            reasons.append("Memory at or above blocking threshold")

        if (
            pd.notna(row["Evaluation_CPU"])
            and CPU_REVIEW_THRESHOLD <= row["Evaluation_CPU"] < CPU_BLOCK_THRESHOLD
        ):
            reasons.append("CPU requires review")

        if (
            pd.notna(row["Evaluation_Memory"])
            and MEMORY_REVIEW_THRESHOLD <= row["Evaluation_Memory"] < MEMORY_BLOCK_THRESHOLD
        ):
            reasons.append("Memory requires review")

        if row["Detected_Anomaly"]:
            reasons.append("Anomaly detected; investigate before changes")

        if row["Optimization_Decision"] == "PROVISIONAL_CANDIDATE":
            reasons.append("Both CPU and memory are low; validate before resizing")

        if row["Optimization_Decision"] == "ROUTINE_MONITORING":
            reasons.append("No immediate utilization-based risk detected")

        return "; ".join(reasons) or "Manual assessment required"

    df["Decision_Reason"] = df.apply(decision_reason, axis=1)

    # This is intentionally conservative.
    df["Manual_Approval_Required"] = True
    df["Automatic_Modification_Allowed"] = False

    # Contractual compliance remains unknown without contract evidence.
    df["SLA_Evidence_Status"] = np.where(
        df["Availability_Evidence_Available"],
        "PARTIAL_EVIDENCE",
        "NOT_VERIFIABLE",
    )

    df["Evaluation_Method"] = (
        "Research thresholds; not contractual SLA certification"
    )

    # Keep original columns plus evaluation fields.
    df.to_csv(output_file, index=False)

    summary = {
        "Dataset": dataset_name,
        "Total_Resources": len(df),
        "CPU_Memory_Evidence_Complete": int(
            df["Utilization_Evidence_Available"].sum()
        ),
        "Invalid_Utilization_Records": int(
            df["Invalid_Utilization_Data"].sum()
        ),
        "Availability_Evidence_Available": int(
            df["Availability_Evidence_Available"].sum()
        ),
        "Meets_Research_Availability_Target": int(
            (df["Research_Availability_Status"]
             == "MEETS_RESEARCH_TARGET").sum()
        ),
        "Below_Research_Availability_Target": int(
            (df["Research_Availability_Status"]
             == "BELOW_RESEARCH_TARGET").sum()
        ),
        "Contractual_SLA_Not_Verifiable": int(
            (df["Contractual_SLA_Status"] == "NOT_VERIFIABLE").sum()
        ),
        "Blocked": int(
            (df["Optimization_Decision"] == "BLOCKED").sum()
        ),
        "Review_Required": int(
            (df["Optimization_Decision"] == "REVIEW_REQUIRED").sum()
        ),
        "Provisional_Candidates": int(
            (df["Optimization_Decision"]
             == "PROVISIONAL_CANDIDATE").sum()
        ),
        "Routine_Monitoring": int(
            (df["Optimization_Decision"]
             == "ROUTINE_MONITORING").sum()
        ),
        "Detected_Anomalies": int(df["Detected_Anomaly"].sum()),
        "Manual_Approval_Required": int(
            df["Manual_Approval_Required"].sum()
        ),
        "Automatic_Modifications_Allowed": int(
            df["Automatic_Modification_Allowed"].sum()
        ),
    }

    return df, summary


# ============================================================
# 6. RUN TRAIN AND TEST EVALUATIONS
# ============================================================

def main():
    train_df, train_summary = evaluate_dataset(
        TRAIN_FILE,
        "TRAIN",
        ANOMALY_TRAIN_FILE,
        OUTPUT_TRAIN,
    )

    test_df, test_summary = evaluate_dataset(
        TEST_FILE,
        "TEST",
        ANOMALY_TEST_FILE,
        OUTPUT_TEST,
    )

    summary_df = pd.DataFrame([train_summary, test_summary])
    summary_df.to_csv(OUTPUT_SUMMARY, index=False)

    print("\n=== FINTECH SLA FRAMEWORK ===")
    print(summary_df.to_string(index=False))

    print(f"\nTrain results: {OUTPUT_TRAIN}")
    print(f"Test results:  {OUTPUT_TEST}")
    print(f"Summary:       {OUTPUT_SUMMARY}")

    print(
        "\nInterpretation: Research availability status is evaluated "
        "only if an availability column exists. Contractual SLA status "
        "remains NOT_VERIFIABLE because the dataset does not establish "
        "contract-specific commitments. All recommendations require "
        "manual approval."
    )


if __name__ == "__main__":
    main()
