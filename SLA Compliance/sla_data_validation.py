
"""
Purpose:
1. Validate required columns in anomaly-detection outputs.
2. Check missing values and invalid numeric measurements.
3. Check utilization ranges and timestamp consistency.
4. Identify whether contractual SLA evidence is available.
5. Generate a validation report without modifying the input data.

Important:
Missing SLA evidence means compliance cannot be verified.
It does NOT automatically mean an SLA violation occurred.
"""

import pandas as pd

from sla_config import (
    REQUIRED_COLUMNS,
    SLA_EVIDENCE_COLUMNS,
)


# ============================================================
# 1. REQUIRED NUMERIC COLUMNS
# ============================================================

NUMERIC_COLUMNS = (
    "CPU_Utilization",
    "Memory_Utilization",
    "Total_Cost_INR",
)

TIMESTAMP_COLUMNS = (
    "Usage_Start_Date",
    "Usage_End_Date",
)


# ============================================================
# 2. VALIDATE ONE DATASET
# ============================================================

def validate_dataset(df: pd.DataFrame, dataset_name: str) -> tuple[pd.DataFrame, dict]:
    """
    Validate one billing dataset.

    Returns:
        validated_df: A copy of the original DataFrame with validation
                      flags added. Original input values are preserved.
        report:       A dictionary containing dataset-level quality metrics.
    """

    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    validated_df = df.copy()
    issues = []

    # --------------------------------------------------------
    # A. Required-column validation
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in validated_df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name}: Missing required columns: {missing_columns}"
        )

    # --------------------------------------------------------
    # B. Duplicate and missing resource identifiers
    # --------------------------------------------------------

    resource_id_missing = (
        validated_df["Resource_ID"].isna()
        | validated_df["Resource_ID"].astype("string").str.strip().eq("")
    )

    validated_df["Validation_Missing_Resource_ID"] = resource_id_missing

    duplicate_resource_id = (
        validated_df["Resource_ID"].notna()
        & validated_df["Resource_ID"].duplicated(keep=False)
    )

    validated_df["Validation_Duplicate_Resource_ID"] = duplicate_resource_id

    # Note:
    # Repeated Resource_ID values may be legitimate when a resource
    # has multiple billing records. They are flagged for inspection,
    # not automatically treated as invalid records.

    # --------------------------------------------------------
    # C. Missing-value checks
    # --------------------------------------------------------

    for column in REQUIRED_COLUMNS:
        validated_df[f"Validation_Missing_{column}"] = (
            validated_df[column].isna()
        )

    missing_required_cells = int(
        validated_df[list(REQUIRED_COLUMNS)].isna().sum().sum()
    )

    # --------------------------------------------------------
    # D. Numeric validation
    # --------------------------------------------------------

    for column in NUMERIC_COLUMNS:
        numeric_values = pd.to_numeric(
            validated_df[column],
            errors="coerce",
        )

        invalid_numeric = (
            validated_df[column].notna()
            & numeric_values.isna()
        )

        non_finite = numeric_values.notna() & ~numeric_values.map(
            lambda value: pd.notna(value) and abs(value) != float("inf")
        )

        # Store validation flags without overwriting the source column.
        validated_df[f"Validation_Invalid_{column}"] = (
            invalid_numeric | non_finite
        )

        validated_df[f"Validation_Missing_{column}"] = (
            numeric_values.isna()
        )

        if column in ("CPU_Utilization", "Memory_Utilization"):
            out_of_range = (
                numeric_values.notna()
                & ~numeric_values.between(0, 100)
            )

            validated_df[f"Validation_Out_Of_Range_{column}"] = (
                out_of_range
            )

        if column == "Total_Cost_INR":
            negative_cost = (
                numeric_values.notna()
                & (numeric_values < 0)
            )

            validated_df["Validation_Negative_Total_Cost"] = negative_cost

    # --------------------------------------------------------
    # E. Timestamp validation
    # --------------------------------------------------------

    parsed_timestamps = {}

    for column in TIMESTAMP_COLUMNS:
       
        parsed = pd.to_datetime(
            validated_df[column],
            errors="coerce",
            format="%Y-%m-%d %H:%M:%S",
        )


        parsed_timestamps[column] = parsed

        validated_df[f"Validation_Invalid_{column}"] = (
            validated_df[column].notna() & parsed.isna()
        )

        validated_df[f"Validation_Missing_{column}"] = (
            validated_df[column].isna()
        )

    start = parsed_timestamps["Usage_Start_Date"]
    end = parsed_timestamps["Usage_End_Date"]

    validated_df["Validation_Invalid_Duration"] = (
        start.notna()
        & end.notna()
        & (end < start)
    )

    # --------------------------------------------------------
    # F. SLA evidence availability
    # --------------------------------------------------------

    # These fields are not expected in the current dataset.
    # Their absence prevents genuine SLA certification, but does
    # not establish a violation.

    available_sla_fields = [
        column
        for column in SLA_EVIDENCE_COLUMNS
        if column in validated_df.columns
    ]

    missing_sla_fields = [
        column
        for column in SLA_EVIDENCE_COLUMNS
        if column not in validated_df.columns
    ]

    if available_sla_fields:
        row_has_missing_sla_evidence = validated_df[
            available_sla_fields
        ].isna().any(axis=1)

        validated_df["SLA_Data_Complete"] = (
            ~row_has_missing_sla_evidence
            & (len(missing_sla_fields) == 0)
        )
    else:
        validated_df["SLA_Data_Complete"] = False

    validated_df["SLA_Compliance_Status"] = "NOT_VERIFIABLE"

    # --------------------------------------------------------
    # G. Overall validation result per row
    # --------------------------------------------------------

    row_issue_columns = [
        "Validation_Missing_Resource_ID",
        "Validation_Duplicate_Resource_ID",
        "Validation_Invalid_CPU_Utilization",
        "Validation_Invalid_Memory_Utilization",
        "Validation_Invalid_Total_Cost_INR",
        "Validation_Out_Of_Range_CPU_Utilization",
        "Validation_Out_Of_Range_Memory_Utilization",
        "Validation_Negative_Total_Cost",
        "Validation_Invalid_Usage_Start_Date",
        "Validation_Invalid_Usage_End_Date",
        "Validation_Invalid_Duration",
    ]

    # Some flags may not exist if the source schema changes.
    existing_issue_columns = [
        column
        for column in row_issue_columns
        if column in validated_df.columns
    ]

    validated_df["Validation_Has_Data_Quality_Issue"] = (
        validated_df[existing_issue_columns]
        .fillna(False)
        .astype(bool)
        .any(axis=1)
    )

    validated_df["Validation_Status"] = (
        validated_df["Validation_Has_Data_Quality_Issue"]
        .map({
            True: "REVIEW_REQUIRED",
            False: "PASSED",
        })
    )

    # --------------------------------------------------------
    # H. Dataset-level validation report
    # --------------------------------------------------------

    report = {
        "Dataset": dataset_name,
        "Rows": len(validated_df),
        "Columns_Original": len(df.columns),
        "Missing_Required_Columns": 0,
        "Missing_Required_Cells": missing_required_cells,
        "Rows_With_Data_Quality_Issues": int(
            validated_df["Validation_Has_Data_Quality_Issue"].sum()
        ),
        "Rows_With_Missing_Resource_ID": int(
            resource_id_missing.sum()
        ),
        "Rows_With_Duplicate_Resource_ID": int(
            duplicate_resource_id.sum()
        ),
        "Rows_With_Invalid_Duration": int(
            validated_df["Validation_Invalid_Duration"].sum()
        ),
        "Rows_With_Incomplete_SLA_Evidence": int(
            (~validated_df["SLA_Data_Complete"]).sum()
        ),
        "Available_SLA_Evidence_Columns": ", ".join(
            available_sla_fields
        ),
        "Missing_SLA_Evidence_Columns": ", ".join(
            missing_sla_fields
        ),
        "SLA_Compliance_Verifiable": bool(
            validated_df["SLA_Data_Complete"].all()
            and len(validated_df) > 0
        ),
    }

    return validated_df, report


# ============================================================
# 3. VALIDATE TRAIN AND TEST DATASETS
# ============================================================

def validate_train_test(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Validate training and testing datasets independently.

    Returns:
        validated_train, validated_test, validation_report
    """

    validated_train, train_report = validate_dataset(
        train_df,
        "TRAIN",
    )

    validated_test, test_report = validate_dataset(
        test_df,
        "TEST",
    )

    validation_report = pd.DataFrame(
        [train_report, test_report]
    )

    return validated_train, validated_test, validation_report


# ============================================================
# 4. STANDALONE EXECUTION TEST
# ============================================================

if __name__ == "__main__":
    from sla_config import TRAIN_INPUT_FILE, TEST_INPUT_FILE

    train_data = pd.read_csv(TRAIN_INPUT_FILE)
    test_data = pd.read_csv(TEST_INPUT_FILE)

    validated_train, validated_test, report = validate_train_test(
        train_data,
        test_data,
    )

    print("\n========== SLA DATA VALIDATION ==========")
    print(report.to_string(index=False))

    print("\n========== ROW-LEVEL VALIDATION ==========")
    print("Training rows requiring data-quality review:",
          int(validated_train["Validation_Has_Data_Quality_Issue"].sum()))

    print("Testing rows requiring data-quality review:",
          int(validated_test["Validation_Has_Data_Quality_Issue"].sum()))

    print("\nSLA compliance status counts (training):")
    print(validated_train["SLA_Compliance_Status"].value_counts())

    print("\nValidation completed. No input files were modified.")
