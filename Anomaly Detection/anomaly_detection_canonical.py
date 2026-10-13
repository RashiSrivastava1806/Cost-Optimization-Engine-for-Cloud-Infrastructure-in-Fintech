
from pathlib import Path
import re

import numpy as np
import pandas as pd


# ==========================================================
# 1. PROJECT PATHS
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CANONICAL_DATA_DIR = DATA_DIR / "canonical_run"

TRAIN_INPUT = CANONICAL_DATA_DIR / "engineered_train_cloud_billing_data.csv"
TEST_INPUT = CANONICAL_DATA_DIR / "engineered_test_cloud_billing_data.csv"

TRAIN_OUTPUT = CANONICAL_DATA_DIR / "anomaly_train_cloud_billing_data.csv"
TEST_OUTPUT = CANONICAL_DATA_DIR / "anomaly_test_cloud_billing_data.csv"
SUMMARY_OUTPUT = CANONICAL_DATA_DIR / "anomaly_detection_summary.csv"


# ==========================================================
# 2. CONFIGURATION
# ==========================================================

IQR_MULTIPLIER = 1.5
ZSCORE_THRESHOLD = 3.0
LOW_UTILIZATION_THRESHOLD = 30.0
HIGH_COST_RATIO = 1.25

# Corrected feature names match feature_engineering.py.
CANDIDATE_FEATURES = [
    "Total_Cost_INR",
    "Usage_Quantity",
    "CPU_Utilization",
    "Memory_Utilization",
    "Network_Inbound_Data_Bytes",
    "Network_Outbound_Data_Bytes",
    "Total_Network_Data_Bytes",
    "Cost_per_Hour",
    "Cost_per_Usage_Unit",
]

REQUIRED_COLUMNS = [
    "Resource_ID",
    "Service_Name",
    "Usage_Start_Date",
    "Total_Cost_INR",
    "CPU_Utilization",
    "Memory_Utilization",
]


# ==========================================================
# 3. LOAD AND VALIDATE DATA
# ==========================================================

def load_data(path):
    if not path.exists():
        raise FileNotFoundError(
            f"Required file not found: {path}\n"
            "Run feature_engineering.py first."
        )

    df = pd.read_csv(path)

    if df.empty:
        raise ValueError(f"Dataset is empty: {path}")

    missing = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns in {path.name}: {missing}"
        )

    df["Usage_Start_Date"] = pd.to_datetime(
        df["Usage_Start_Date"],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )

    for column in CANDIDATE_FEATURES:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    return df


# ==========================================================
# 4. FIT STATISTICAL THRESHOLDS ON TRAINING DATA
# ==========================================================

def fit_thresholds(train_df, features):
    thresholds = {}

    for feature in features:
        values = (
            pd.to_numeric(train_df[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
            .dropna()
        )

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        mean = values.mean()
        std = values.std(ddof=0)

        thresholds[feature] = {
            "lower_iqr": q1 - IQR_MULTIPLIER * iqr,
            "upper_iqr": q3 + IQR_MULTIPLIER * iqr,
            "mean": mean,
            "std": std,
        }

    return thresholds


# ==========================================================
# 5. APPLY STATISTICAL ANOMALY DETECTION
# ==========================================================

def detect_statistical_anomalies(df, thresholds):
    result = df.copy()

    result["IQR_Outlier_Count"] = 0
    result["ZScore_Outlier_Count"] = 0

    for feature, limits in thresholds.items():
        values = (
            pd.to_numeric(result[feature], errors="coerce")
            .replace([np.inf, -np.inf], np.nan)
        )

        iqr_flag = (
            (values < limits["lower_iqr"])
            | (values > limits["upper_iqr"])
        ).fillna(False)

        if limits["std"] > 0:
            z_scores = (
                values - limits["mean"]
            ) / limits["std"]

            z_flag = (
                z_scores.abs() > ZSCORE_THRESHOLD
            ).fillna(False)
        else:
            z_flag = pd.Series(
                False,
                index=result.index,
            )

        safe_name = re.sub(
            r"[^A-Za-z0-9_]",
            "_",
            feature,
        )

        result[f"{safe_name}_IQR_Outlier"] = (
            iqr_flag.astype(int)
        )

        result[f"{safe_name}_ZScore_Outlier"] = (
            z_flag.astype(int)
        )

        result["IQR_Outlier_Count"] += iqr_flag.astype(int)
        result["ZScore_Outlier_Count"] += z_flag.astype(int)

    result["Statistical_Outlier_Flag"] = (
        (result["IQR_Outlier_Count"] > 0)
        | (result["ZScore_Outlier_Count"] > 0)
    ).astype(int)

    return result


# ==========================================================
# 6. CONTEXTUAL ANOMALY DETECTION
# ==========================================================

def add_contextual_detection(train_df, target_df):
    result = target_df.copy()

    # Derive service benchmarks from training data only.
    training_cost = train_df[
        ["Service_Name", "Total_Cost_INR"]
    ].copy()

    training_cost["Total_Cost_INR"] = pd.to_numeric(
        training_cost["Total_Cost_INR"],
        errors="coerce",
    )

    overall_median = training_cost["Total_Cost_INR"].median()

    if pd.isna(overall_median):
        raise ValueError(
            "Cannot calculate the training cost benchmark."
        )

    service_medians = training_cost.groupby(
        "Service_Name"
    )["Total_Cost_INR"].median()

    result["Is_Unseen_Service"] = (
        ~result["Service_Name"].isin(service_medians.index)
    ).astype(int)

    result["Service_Median_Cost"] = (
        result["Service_Name"]
        .map(service_medians)
        .fillna(overall_median)
    )

    cpu = pd.to_numeric(
        result["CPU_Utilization"],
        errors="coerce",
    )

    memory = pd.to_numeric(
        result["Memory_Utilization"],
        errors="coerce",
    )

    cost = pd.to_numeric(
        result["Total_Cost_INR"],
        errors="coerce",
    )

    result["Detection_Utilization_Score"] = (
        cpu + memory
    ) / 2

    utilization_available = (
        cpu.notna()
        & memory.notna()
    )

    result["Low_Utilization_Flag"] = (
        utilization_available
        & (
            result["Detection_Utilization_Score"]
            < LOW_UTILIZATION_THRESHOLD
        )
    ).astype(int)

    result["High_Cost_Ratio"] = (
        cost
        / result["Service_Median_Cost"].replace(0, np.nan)
    )

    result["High_Cost_Relative_To_Service_Flag"] = (
        result["High_Cost_Ratio"] >= HIGH_COST_RATIO
    ).fillna(False).astype(int)

    # Unknown services use a fallback benchmark but are not
    # automatically classified as contextual anomalies.
    result["Contextual_Anomaly_Flag"] = (
        (result["Low_Utilization_Flag"] == 1)
        & (
            result["High_Cost_Relative_To_Service_Flag"] == 1
        )
        & (result["Is_Unseen_Service"] == 0)
    ).astype(int)

    return result


# ==========================================================
# 7. COMBINE RESULTS
# ==========================================================

def finalize_results(df):
    result = df.copy()

    result["Anomaly_Flag"] = (
        (result["Statistical_Outlier_Flag"] == 1)
        | (result["Contextual_Anomaly_Flag"] == 1)
    ).astype(int)

    # This is a review queue, not confirmed savings.
    result["Potential_Optimization_Candidate"] = (
        result["Contextual_Anomaly_Flag"]
    ).astype(int)

    def explain_anomaly(row):
        reasons = []

        if row["Statistical_Outlier_Flag"] == 1:
            reasons.append("Statistical outlier")

        if row["Contextual_Anomaly_Flag"] == 1:
            reasons.append(
                "Low utilization with high service-relative cost"
            )

        return "; ".join(reasons) if reasons else "No anomaly detected"

    result["Anomaly_Reason"] = result.apply(
        explain_anomaly,
        axis=1,
    )

    return result


# ==========================================================
# 8. SUMMARY REPORT
# ==========================================================

def create_summary(df, dataset_name):
    rows = []

    for feature in CANDIDATE_FEATURES:
        iqr_column = f"{feature}_IQR_Outlier"
        z_column = f"{feature}_ZScore_Outlier"

        if iqr_column not in df.columns:
            continue

        rows.append({
            "Dataset": dataset_name,
            "Feature": feature,
            "IQR_Outlier_Count": int(df[iqr_column].sum()),
            "ZScore_Outlier_Count": int(df[z_column].sum()),
        })

    rows.append({
        "Dataset": dataset_name,
        "Feature": "OVERALL",
        "IQR_Outlier_Count": int(
            (df["IQR_Outlier_Count"] > 0).sum()
        ),
        "ZScore_Outlier_Count": int(
            (df["ZScore_Outlier_Count"] > 0).sum()
        ),
        "Statistical_Anomaly_Count": int(
            df["Statistical_Outlier_Flag"].sum()
        ),
        "Contextual_Anomaly_Count": int(
            df["Contextual_Anomaly_Flag"].sum()
        ),
        "Combined_Anomaly_Count": int(
            df["Anomaly_Flag"].sum()
        ),
        "Optimization_Candidate_Count": int(
            df["Potential_Optimization_Candidate"].sum()
        ),
        "Unseen_Service_Count": int(
            df["Is_Unseen_Service"].sum()
        ),
        "Total_Rows": len(df),
    })

    return rows


# ==========================================================
# 9. MAIN EXECUTION
# ==========================================================

def main():
    print("Loading engineered datasets...")

    train_df = load_data(TRAIN_INPUT)
    test_df = load_data(TEST_INPUT)

    print("Training rows:", len(train_df))
    print("Testing rows:", len(test_df))

    # Validate temporal separation.
    if (
        train_df["Usage_Start_Date"].max()
        >= test_df["Usage_Start_Date"].min()
    ):
        raise ValueError(
            "Chronological separation failed. "
            "Check the train/test split."
        )

    print("Chronological separation verified.")

    features = [
        feature
        for feature in CANDIDATE_FEATURES
        if feature in train_df.columns
        and pd.api.types.is_numeric_dtype(train_df[feature])
    ]

    if not features:
        raise ValueError(
            "No numeric anomaly features found."
        )

    print("Features used:", ", ".join(features))

    print("\nFitting thresholds on training data...")
    thresholds = fit_thresholds(train_df, features)

    if not thresholds:
        raise ValueError(
            "No usable statistical thresholds could be fitted."
        )

    print("Detecting statistical anomalies...")

    train_result = detect_statistical_anomalies(
        train_df,
        thresholds,
    )

    test_result = detect_statistical_anomalies(
        test_df,
        thresholds,
    )

    print("Detecting contextual anomalies...")

    train_result = add_contextual_detection(
        train_df,
        train_result,
    )

    test_result = add_contextual_detection(
        train_df,
        test_result,
    )

    train_result = finalize_results(train_result)
    test_result = finalize_results(test_result)

    # Save detailed outputs.
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    train_result.to_csv(TRAIN_OUTPUT, index=False)
    test_result.to_csv(TEST_OUTPUT, index=False)

    summary_rows = (
        create_summary(train_result, "Train")
        + create_summary(test_result, "Test")
    )

    pd.DataFrame(summary_rows).to_csv(
        SUMMARY_OUTPUT,
        index=False,
    )

    print("\nAnomaly detection completed successfully.")
    print("Training results:", TRAIN_OUTPUT)
    print("Testing results:", TEST_OUTPUT)
    print("Summary report:", SUMMARY_OUTPUT)

    for name, result in [
        ("Train", train_result),
        ("Test", test_result),
    ]:
        print(f"\n{name} summary")
        print("Total rows:", len(result))
        print(
            "Statistical anomalies:",
            int(result["Statistical_Outlier_Flag"].sum()),
        )
        print(
            "Contextual anomalies:",
            int(result["Contextual_Anomaly_Flag"].sum()),
        )
        print(
            "Combined anomalies:",
            int(result["Anomaly_Flag"].sum()),
        )
        print(
            "Optimization candidates:",
            int(result["Potential_Optimization_Candidate"].sum()),
        )

    # Confirm no records were lost.
    assert len(train_result) == len(train_df)
    assert len(test_result) == len(test_df)

    print("\nRow preservation checks passed.")


if __name__ == "__main__":
    main()
