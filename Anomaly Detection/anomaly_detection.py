
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# 1. PATHS AND CONFIGURATION
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

TRAIN_FILE = DATA_DIR / "engineered_train_cloud_billing_data.csv"
TEST_FILE = DATA_DIR / "engineered_test_cloud_billing_data.csv"

TRAIN_OUTPUT = DATA_DIR / "anomaly_train_cloud_billing_data.csv"
TEST_OUTPUT = DATA_DIR / "anomaly_test_cloud_billing_data.csv"
SUMMARY_OUTPUT = DATA_DIR / "anomaly_detection_summary.csv"

# Features for individual statistical outlier detection.
# Only columns present in both datasets will be analyzed.
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

# IQR multiplier: standard exploratory outlier boundary.
IQR_MULTIPLIER = 1.5

# Secondary statistical signal.
Z_SCORE_THRESHOLD = 3.0

# Initial business-rule thresholds; validate these against EDA.
LOW_UTILIZATION_THRESHOLD = 30.0
HIGH_RELATIVE_COST_THRESHOLD = 1.25


# ---------------------------------------------------------
# 2. LOAD DATA
# ---------------------------------------------------------

def load_data(file_path):
    """Load a dataset and validate that it is not empty."""
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}\n"
            "Check that feature engineering has been completed."
        )

    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError(f"The dataset is empty: {file_path}")

    return df


# ---------------------------------------------------------
# 3. SELECT NUMERIC FEATURES
# ---------------------------------------------------------

def get_features(train_df, test_df):
    """Select suitable numeric columns available in both datasets."""
    features = []

    for col in CANDIDATE_FEATURES:
        if col not in train_df.columns or col not in test_df.columns:
            print(f"Skipping {col}: column missing from train or test.")
            continue

        # Convert invalid numeric strings to NaN.
        train_df[col] = pd.to_numeric(train_df[col], errors="coerce")
        test_df[col] = pd.to_numeric(test_df[col], errors="coerce")

        # Infinite values cannot be used for threshold calculation.
        train_df[col] = train_df[col].replace([np.inf, -np.inf], np.nan)
        test_df[col] = test_df[col].replace([np.inf, -np.inf], np.nan)

        # A feature needs at least some valid training observations.
        if train_df[col].notna().sum() < 4:
            print(f"Skipping {col}: insufficient valid training values.")
            continue

        features.append(col)

    if not features:
        raise ValueError(
            "No usable numeric features found. "
            "Check the engineered dataset column names."
        )

    return features


# ---------------------------------------------------------
# 4. LEARN STATISTICAL THRESHOLDS FROM TRAINING DATA
# ---------------------------------------------------------

def fit_thresholds(train_df, features):
    """
    Calculate IQR boundaries and mean/standard deviation
    from training data only.
    """
    thresholds = {}

    for col in features:
        values = train_df[col].dropna()

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)
        iqr = q3 - q1

        thresholds[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "iqr_lower": q1 - IQR_MULTIPLIER * iqr,
            "iqr_upper": q3 + IQR_MULTIPLIER * iqr,
            "mean": values.mean(),
            "std": values.std(ddof=0),
        }

    return thresholds


# ---------------------------------------------------------
# 5. APPLY OUTLIER DETECTION
# ---------------------------------------------------------

def apply_statistical_detection(df, features, thresholds):
    """
    Apply training-derived IQR and Z-score thresholds.
    No thresholds are learned from the dataset being scored.
    """
    df = df.copy()
    feature_outlier_columns = []
    feature_zscore_columns = []

    for col in features:
        t = thresholds[col]
        values = df[col]

        # IQR detection
        iqr_flag_col = f"{col}_IQR_Outlier"
        df[iqr_flag_col] = (
            (values < t["iqr_lower"]) |
            (values > t["iqr_upper"])
        ).fillna(False)

        feature_outlier_columns.append(iqr_flag_col)

        # Z-score detection
        zscore_flag_col = f"{col}_ZScore_Outlier"

        if t["std"] > 0 and np.isfinite(t["std"]):
            z_score = (values - t["mean"]) / t["std"]
            df[f"{col}_ZScore"] = z_score
            df[zscore_flag_col] = (
                z_score.abs() > Z_SCORE_THRESHOLD
            ).fillna(False)
        else:
            # A constant training feature has no usable Z-score.
            df[f"{col}_ZScore"] = np.nan
            df[zscore_flag_col] = False

        feature_zscore_columns.append(zscore_flag_col)

    # Count how many individual features were flagged.
    df["IQR_Outlier_Count"] = df[feature_outlier_columns].sum(axis=1)
    df["ZScore_Outlier_Count"] = df[feature_zscore_columns].sum(axis=1)

    df["Statistical_Outlier_Flag"] = (
        (df["IQR_Outlier_Count"] > 0) |
        (df["ZScore_Outlier_Count"] > 0)
    )

    return df


# ---------------------------------------------------------
# 6. CONTEXTUAL ANOMALY DETECTION
# ---------------------------------------------------------

def apply_contextual_detection(df):
    """
    Identify a potential cost-optimization signal using
    low utilization plus high cost relative to service median.
    """
    df = df.copy()

    required = {
        "CPU_Utilization",
        "Memory_Utilization",
        "Total_Cost_INR",
        "Service_Median_Cost",
    }

    missing = required - set(df.columns)

    if missing:
        raise ValueError(
            "Missing columns needed for contextual detection: "
            + ", ".join(sorted(missing))
        )

    cpu = pd.to_numeric(df["CPU_Utilization"], errors="coerce")
    memory = pd.to_numeric(df["Memory_Utilization"], errors="coerce")
    cost = pd.to_numeric(df["Total_Cost_INR"], errors="coerce")
    service_median = pd.to_numeric(
        df["Service_Median_Cost"], errors="coerce"
    )

    # Simple, explainable baseline utilization score.
    df["Detection_Utilization_Score"] = (cpu + memory) / 2

    df["Detection_Low_Utilization"] = (
        df["Detection_Utilization_Score"]
        < LOW_UTILIZATION_THRESHOLD
    ).fillna(False)

    # Relative cost is meaningful only for a positive benchmark.
    valid_benchmark = service_median > 0
    df["Detection_Cost_to_Service_Median"] = np.where(
        valid_benchmark,
        cost / service_median,
        np.nan,
    )

    df["Detection_High_Relative_Cost"] = (
        df["Detection_Cost_to_Service_Median"]
        >= HIGH_RELATIVE_COST_THRESHOLD
    ).fillna(False)

    # Both conditions must hold to flag this contextual signal.
    df["Contextual_Anomaly_Flag"] = (
        df["Detection_Low_Utilization"] &
        df["Detection_High_Relative_Cost"]
    )

    return df


# ---------------------------------------------------------
# 7. COMBINE SIGNALS AND EXPLAIN EACH FLAG
# ---------------------------------------------------------

def combine_results(df):
    df = df.copy()

    df["Anomaly_Flag"] = (
        df["Statistical_Outlier_Flag"] |
        df["Contextual_Anomaly_Flag"]
    )

    reasons = []

    for _, row in df.iterrows():
        row_reasons = []

        if row["IQR_Outlier_Count"] > 0:
            row_reasons.append(
                f"IQR outlier in {int(row['IQR_Outlier_Count'])} feature(s)"
            )

        if row["ZScore_Outlier_Count"] > 0:
            row_reasons.append(
                f"Z-score outlier in {int(row['ZScore_Outlier_Count'])} feature(s)"
            )

        if row["Contextual_Anomaly_Flag"]:
            ratio = row["Detection_Cost_to_Service_Median"]
            row_reasons.append(
                "Low utilization with cost at "
                f"{ratio:.2f}x the service median"
            )

        reasons.append("; ".join(row_reasons) if row_reasons else "No rule triggered")

    df["Anomaly_Reason"] = reasons

    # This is a candidate flag, not proof of waste or a command to
    # downsize the resource. SLA and compliance checks come later.
    df["Potential_Optimization_Candidate"] = (
        df["Contextual_Anomaly_Flag"]
    )

    return df


# ---------------------------------------------------------
# 8. CREATE SUMMARY REPORT
# ---------------------------------------------------------

def create_summary(train_df, test_df, features, thresholds):
    rows = []

    datasets = [
        ("Train", train_df),
        ("Test", test_df),
    ]

    for dataset_name, df in datasets:
        for col in features:
            t = thresholds[col]

            rows.append({
                "Dataset": dataset_name,
                "Feature": col,
                "Valid_Values": int(df[col].notna().sum()),
                "IQR_Lower_Bound": t["iqr_lower"],
                "IQR_Upper_Bound": t["iqr_upper"],
                "Mean_Training_Value": t["mean"],
                "Std_Training_Value": t["std"],
                "IQR_Outlier_Count": int(
                    df[f"{col}_IQR_Outlier"].sum()
                ),
                "ZScore_Outlier_Count": int(
                    df[f"{col}_ZScore_Outlier"].sum()
                ),
            })

        rows.append({
            "Dataset": dataset_name,
            "Feature": "OVERALL",
            "Valid_Values": len(df),
            "IQR_Lower_Bound": np.nan,
            "IQR_Upper_Bound": np.nan,
            "Mean_Training_Value": np.nan,
            "Std_Training_Value": np.nan,
            "IQR_Outlier_Count": int(df["Statistical_Outlier_Flag"].sum()),
            "ZScore_Outlier_Count": int(df["Contextual_Anomaly_Flag"].sum()),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# 9. MAIN EXECUTION
# ---------------------------------------------------------

def main():
    print("=" * 65)
    print("CLOUD BILLING OUTLIER AND ANOMALY DETECTION")
    print("=" * 65)

    train_df = load_data(TRAIN_FILE)
    test_df = load_data(TEST_FILE)

    print(f"\nTraining dataset shape: {train_df.shape}")
    print(f"Testing dataset shape:  {test_df.shape}")

    features = get_features(train_df, test_df)
    print("\nFeatures selected:")
    for col in features:
        print(f"  - {col}")

    # Learn thresholds exclusively from training data.
    thresholds = fit_thresholds(train_df, features)

    train_df = apply_statistical_detection(
        train_df, features, thresholds
    )
    test_df = apply_statistical_detection(
        test_df, features, thresholds
    )

    train_df = apply_contextual_detection(train_df)
    test_df = apply_contextual_detection(test_df)

    train_df = combine_results(train_df)
    test_df = combine_results(test_df)

    train_df.to_csv(TRAIN_OUTPUT, index=False)
    test_df.to_csv(TEST_OUTPUT, index=False)

    summary = create_summary(
        train_df, test_df, features, thresholds
    )
    summary.to_csv(SUMMARY_OUTPUT, index=False)

    print("\n" + "=" * 65)
    print("RESULTS")
    print("=" * 65)

    for name, df in [("TRAIN", train_df), ("TEST", test_df)]:
        print(f"\n{name} DATASET")
        print(f"Total records: {len(df)}")
        print(
            "Records with statistical outliers:",
            int(df["Statistical_Outlier_Flag"].sum()),
        )
        print(
            "Records with contextual anomalies:",
            int(df["Contextual_Anomaly_Flag"].sum()),
        )
        print(
            "Records flagged by either method:",
            int(df["Anomaly_Flag"].sum()),
        )
        print(
            "Potential optimization candidates:",
            int(df["Potential_Optimization_Candidate"].sum()),
        )

    print("\nSaved files:")
    print(TRAIN_OUTPUT)
    print(TEST_OUTPUT)
    print(SUMMARY_OUTPUT)

    print("\nDetection completed.")
    print(
        "Review flagged records before treating them as genuine "
        "anomalies or optimization opportunities."
    )


if __name__ == "__main__":
    main()