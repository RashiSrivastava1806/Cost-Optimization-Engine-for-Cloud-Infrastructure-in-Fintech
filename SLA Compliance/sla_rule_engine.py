
"""
SLA Rule Engine
---------------
Applies provisional optimization guardrails to cloud resources.

Uses:
- CPU and memory utilization
- Cost-related signals
- Statistical and contextual anomaly flags
- Existing low-utilization flags

Important:
This module assesses optimization risk. It does not certify SLA
compliance or automatically modify cloud resources.
"""

import pandas as pd

from sla_config import (
    CPU_BLOCK_THRESHOLD,
    CPU_REVIEW_THRESHOLD,
    CPU_LOW_UTILIZATION_THRESHOLD,
    MEMORY_BLOCK_THRESHOLD,
    MEMORY_REVIEW_THRESHOLD,
    MEMORY_LOW_UTILIZATION_THRESHOLD,
    HIGH_COST_RATIO_THRESHOLD,
    ANOMALY_REQUIRES_INVESTIGATION,
    SLA_STATUS_NOT_VERIFIABLE,
    RISK_HIGH,
    RISK_MEDIUM,
    RISK_LOW,
    RISK_UNKNOWN,
    DECISION_BLOCKED,
    DECISION_REVIEW_REQUIRED,
    DECISION_PROVISIONAL_CANDIDATE,
)


def find_column(df, possible_names):
    """Find a column without depending on capitalization."""
    normalized = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for name in possible_names:
        if name.lower() in normalized:
            return normalized[name.lower()]

    return None


def convert_to_bool(series):
    """Convert common Boolean, numeric, and text flags to Boolean."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)

    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").fillna(0).ne(0)

    normalized = series.astype("string").str.strip().str.lower()

    return normalized.isin(
        ["true", "1", "yes", "y", "anomaly", "outlier", "detected"]
    )


def apply_sla_rules(df):
    """
    Apply optimization risk rules and return an assessed DataFrame.

    The input DataFrame is not modified.
    """

    result = df.copy()

    # ---------------------------------------------------------
    # 1. Identify required utilization columns
    # ---------------------------------------------------------
    cpu_col = find_column(result, ["CPU_Utilization"])
    memory_col = find_column(result, ["Memory_Utilization"])

    if cpu_col is None or memory_col is None:
        missing = []

        if cpu_col is None:
            missing.append("CPU_Utilization")

        if memory_col is None:
            missing.append("Memory_Utilization")

        raise ValueError(
            "Missing required columns: " + ", ".join(missing)
        )

    # ---------------------------------------------------------
    # 2. Read numeric signals
    # ---------------------------------------------------------
    cpu = pd.to_numeric(result[cpu_col], errors="coerce")
    memory = pd.to_numeric(result[memory_col], errors="coerce")

    signals_available = (
        cpu.notna()
        & memory.notna()
        & cpu.between(0, 100)
        & memory.between(0, 100)
    )

    # ---------------------------------------------------------
    # 3. Detect elevated cost
    # ---------------------------------------------------------
    cost_ratio_col = find_column(
        result,
        [
            "High_Cost_Ratio",
            "Cost_vs_Service_Median",
            "Cost_vs_Region_Median",
            "Cost_Ratio",
            "Cost_to_Service_Median_Ratio",
        ],
    )

    if cost_ratio_col is not None:
        cost_ratio = pd.to_numeric(
            result[cost_ratio_col],
            errors="coerce",
        )

        high_cost_signal = (
            cost_ratio >= HIGH_COST_RATIO_THRESHOLD
        ).fillna(False)
    else:
        high_cost_signal = pd.Series(
            False,
            index=result.index,
        )

    # ---------------------------------------------------------
    # 4. Read existing low-utilization signal
    # ---------------------------------------------------------
    low_utilization = (
        (cpu <= CPU_LOW_UTILIZATION_THRESHOLD)
        & (memory <= MEMORY_LOW_UTILIZATION_THRESHOLD)
    ).fillna(False)

    low_utilization_col = find_column(
        result,
        ["Low_Utilization_Flag", "Is_Low_Utilization"],
    )

    if low_utilization_col is not None:
        existing_low_utilization = convert_to_bool(
            result[low_utilization_col]
        )

        # Preserve the conservative CPU-and-memory condition OR
        # an existing pipeline-generated low-utilization flag.
        low_utilization = (
            low_utilization | existing_low_utilization
        )

    # ---------------------------------------------------------
    # 5. Detect anomalies using actual pipeline columns
    # ---------------------------------------------------------
    anomaly_columns = [
        column
        for column in [
            "Anomaly_Flag",
            "Contextual_Anomaly_Flag",
            "Statistical_Outlier_Flag",
        ]
        if column in result.columns
    ]

    anomaly_detected = pd.Series(
        False,
        index=result.index,
    )

    for column in anomaly_columns:
        anomaly_detected |= convert_to_bool(result[column])

    # ---------------------------------------------------------
    # 6. Apply utilization thresholds
    # ---------------------------------------------------------
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

    # ---------------------------------------------------------
    # 7. Initialize decisions conservatively
    # ---------------------------------------------------------
    decisions = pd.Series(
        DECISION_PROVISIONAL_CANDIDATE,
        index=result.index,
        dtype="object",
    )

    risk_levels = pd.Series(
        RISK_LOW,
        index=result.index,
        dtype="object",
    )

    block_reasons = pd.Series(
        "",
        index=result.index,
        dtype="object",
    )

    risk_reasons = pd.Series(
        "",
        index=result.index,
        dtype="object",
    )

    # Missing or invalid utilization cannot be considered safe.
    decisions.loc[~signals_available] = DECISION_REVIEW_REQUIRED
    risk_levels.loc[~signals_available] = RISK_UNKNOWN

    risk_reasons.loc[~signals_available] = (
        "Missing or invalid CPU/memory utilization measurements"
    )

    # ---------------------------------------------------------
    # 8. Block optimization at high utilization
    # ---------------------------------------------------------
    cpu_block_rows = signals_available & cpu_block
    memory_block_rows = signals_available & memory_block

    blocked_rows = cpu_block_rows | memory_block_rows

    decisions.loc[blocked_rows] = DECISION_BLOCKED
    risk_levels.loc[blocked_rows] = RISK_HIGH

    block_reasons.loc[cpu_block_rows] = (
        "CPU utilization meets or exceeds the blocking threshold"
    )

    block_reasons.loc[memory_block_rows] = (
        "Memory utilization meets or exceeds the blocking threshold"
    )

    both_block = cpu_block_rows & memory_block_rows

    block_reasons.loc[both_block] = (
        "CPU and memory utilization meet or exceed blocking thresholds"
    )

    risk_reasons.loc[blocked_rows] = (
        "High resource utilization; downsizing is blocked"
    )

    # ---------------------------------------------------------
    # 9. Elevated utilization requires manual review
    # ---------------------------------------------------------
    review_utilization = (
        signals_available
        & (cpu_review | memory_review)
        & (~blocked_rows)
    )

    decisions.loc[review_utilization] = DECISION_REVIEW_REQUIRED
    risk_levels.loc[review_utilization] = RISK_MEDIUM

    risk_reasons.loc[review_utilization] = (
        "Elevated CPU or memory utilization requires review"
    )

    # ---------------------------------------------------------
    # 10. High cost requires investigation
    # ---------------------------------------------------------
    cost_review = (
        high_cost_signal
        & signals_available
        & (~blocked_rows)
    )

    decisions.loc[cost_review] = DECISION_REVIEW_REQUIRED
    risk_levels.loc[cost_review] = RISK_MEDIUM

    existing_reason = risk_reasons.loc[cost_review]

    risk_reasons.loc[cost_review] = existing_reason.where(
        existing_reason.eq(""),
        existing_reason + "; ",
    ) + "Elevated cost signal requires investigation"

    # ---------------------------------------------------------
    # 11. Anomalies require investigation
    # ---------------------------------------------------------
    if ANOMALY_REQUIRES_INVESTIGATION:
        anomaly_review = (
            anomaly_detected
            & signals_available
            & (~blocked_rows)
        )

        decisions.loc[anomaly_review] = DECISION_REVIEW_REQUIRED
        risk_levels.loc[anomaly_review] = RISK_MEDIUM

        existing_reason = risk_reasons.loc[anomaly_review]

        risk_reasons.loc[anomaly_review] = existing_reason.where(
            existing_reason.eq(""),
            existing_reason + "; ",
        ) + "Anomaly detected; investigate before optimization"

    # ---------------------------------------------------------
    # 12. Explain provisional candidates
    # ---------------------------------------------------------
    provisional = (
        decisions == DECISION_PROVISIONAL_CANDIDATE
    )

    low_utilization_candidates = provisional & low_utilization

    risk_reasons.loc[low_utilization_candidates] = (
        "Low-utilization signal detected; provisional optimization "
        "candidate requiring manual review and SLA verification"
    )

    other_candidates = provisional & (~low_utilization)

    risk_reasons.loc[other_candidates] = (
        "No blocking signal detected; manual review and SLA "
        "verification are still required"
    )

    # ---------------------------------------------------------
    # 13. Preserve honest SLA compliance status
    # ---------------------------------------------------------
    # Do not infer compliance from utilization or anomaly signals.
    result["SLA_Compliance_Status"] = SLA_STATUS_NOT_VERIFIABLE

    # ---------------------------------------------------------
    # 14. Add output columns
    # ---------------------------------------------------------
    result["SLA_Risk_Level"] = risk_levels
    result["Optimization_Decision"] = decisions
    result["Optimization_Block_Reason"] = block_reasons
    result["SLA_Risk_Reason"] = risk_reasons
    result["SLA_Low_Utilization_Signal"] = low_utilization
    result["SLA_High_Cost_Signal"] = high_cost_signal
    result["SLA_Anomaly_Investigation_Required"] = anomaly_detected

    return result


if __name__ == "__main__":
    from sla_config import TRAIN_INPUT_FILE, TEST_INPUT_FILE

    datasets = [
        ("TRAIN", TRAIN_INPUT_FILE),
        ("TEST", TEST_INPUT_FILE),
    ]

    for dataset_name, input_file in datasets:
        if not input_file.exists():
            print(
                f"\n{dataset_name}: input file not found: {input_file}"
            )
            continue

        dataset = pd.read_csv(input_file)
        assessed = apply_sla_rules(dataset)

        print(
            f"\n========== {dataset_name} SLA RULE ENGINE =========="
        )
        print(f"Input rows: {len(dataset)}")

        print("\nOptimization decisions:")
        print(
            assessed["Optimization_Decision"]
            .value_counts()
            .to_string()
        )

        print("\nRisk levels:")
        print(
            assessed["SLA_Risk_Level"]
            .value_counts()
            .to_string()
        )

        print("\nSLA compliance status:")
        print(
            assessed["SLA_Compliance_Status"]
            .value_counts()
            .to_string()
        )

        print("\nSignals recognized:")
        print(f"Cost ratio column: {cost_ratio_col if False else find_column(dataset, ['High_Cost_Ratio', 'Cost_vs_Service_Median', 'Cost_vs_Region_Median', 'Cost_Ratio', 'Cost_to_Service_Median_Ratio'])}")
        print(f"Anomaly columns: {[c for c in ['Anomaly_Flag', 'Contextual_Anomaly_Flag', 'Statistical_Outlier_Flag'] if c in dataset.columns]}")
        print(f"Low-utilization column: {find_column(dataset, ['Low_Utilization_Flag', 'Is_Low_Utilization'])}")

        print("\nInput file was not modified.")
