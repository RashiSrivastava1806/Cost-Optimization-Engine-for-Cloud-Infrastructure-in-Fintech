
from pathlib import Path

import numpy as np
import pandas as pd


# ==========================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

CANONICAL_DATA_DIR = DATA_DIR / "canonical_run"

TRAIN_INPUT = CANONICAL_DATA_DIR / "train_cloud_billing_data.csv"
TEST_INPUT = CANONICAL_DATA_DIR / "test_cloud_billing_data.csv"

TRAIN_OUTPUT = CANONICAL_DATA_DIR / "engineered_train_cloud_billing_data.csv"
TEST_OUTPUT = CANONICAL_DATA_DIR / "engineered_test_cloud_billing_data.csv"

LOW_UTILIZATION_THRESHOLD = 30
HIGH_COST_QUANTILE = 0.90
SERVICE_COST_MULTIPLIER = 1.25

ORIGINAL_COLUMNS = [
    "Resource_ID",
    "Service_Name",
    "Usage_Quantity",
    "Usage_Unit",
    "Region_Zone",
    "CPU_Utilization",
    "Memory_Utilization",
    "Network_Inbound_Data_Bytes",
    "Network_Outbound_Data_Bytes",
    "Usage_Start_Date",
    "Usage_End_Date",
    "Cost_per_Quantity",
    "Unrounded_Cost",
    "Rounded_Cost",
    "Total_Cost_INR",
]

NUMERIC_COLUMNS = [
    "Usage_Quantity",
    "CPU_Utilization",
    "Memory_Utilization",
    "Network_Inbound_Data_Bytes",
    "Network_Outbound_Data_Bytes",
    "Cost_per_Quantity",
    "Unrounded_Cost",
    "Rounded_Cost",
    "Total_Cost_INR",
]


# ==========================================================
# 2. LOAD AND VALIDATE DATASETS
# ==========================================================

def load_dataset(file_path):
    if not file_path.exists():
        raise FileNotFoundError(
            f"Required dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    missing_columns = [
        column
        for column in ORIGINAL_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in {file_path.name}: "
            f"{missing_columns}"
        )

    if df.empty:
        raise ValueError(
            f"The dataset {file_path.name} is empty."
        )

    # Parse the already-corrected ISO-formatted timestamps.
    for column in [
        "Usage_Start_Date",
        "Usage_End_Date",
    ]:
        df[column] = pd.to_datetime(
            df[column],
            format="%Y-%m-%d %H:%M:%S",
            errors="raise",
        )

    # Ensure quantitative fields are numeric.
    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    return df


train_df = load_dataset(TRAIN_INPUT)
test_df = load_dataset(TEST_INPUT)

print("Datasets loaded successfully.")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


# ==========================================================
# 3. VALIDATE CHRONOLOGICAL SPLIT
# ==========================================================

train_max_date = train_df["Usage_Start_Date"].max()
test_min_date = test_df["Usage_Start_Date"].min()

if train_max_date >= test_min_date:
    raise ValueError(
        "Chronological split validation failed. "
        "Training timestamps must precede testing timestamps."
    )

print("\nChronological split verified.")
print("Last training timestamp:", train_max_date)
print("First testing timestamp:", test_min_date)


# ==========================================================
# 4. BASIC FEATURE ENGINEERING
# ==========================================================

def create_basic_features(df):
    df = df.copy()

    # ------------------------------------------------------
    # 4.1 Usage duration
    # ------------------------------------------------------

    duration = (
        df["Usage_End_Date"] - df["Usage_Start_Date"]
    ).dt.total_seconds() / 3600

    # Flag invalid durations instead of silently hiding them.
    df["Has_Invalid_Duration"] = (
        duration < 0
    ).astype(int)

    df["Usage_Duration_Hours"] = duration.where(
        duration >= 0,
        np.nan,
    )

    # ------------------------------------------------------
    # 4.2 Total network data
    # ------------------------------------------------------

    inbound = df["Network_Inbound_Data_Bytes"]
    outbound = df["Network_Outbound_Data_Bytes"]

    df["Total_Network_Data_Bytes"] = (
        inbound.fillna(0) + outbound.fillna(0)
    )

    df["Has_Missing_Network_Data"] = (
        inbound.isna() | outbound.isna()
    ).astype(int)

    # ------------------------------------------------------
    # 4.3 Utilization features
    # ------------------------------------------------------

    cpu = df["CPU_Utilization"]
    memory = df["Memory_Utilization"]

    df["Has_Missing_Utilization"] = (
        cpu.isna() | memory.isna()
    ).astype(int)

    df["CPU_Memory_Avg"] = (
        cpu + memory
    ) / 2

    df["CPU_Memory_Gap"] = (
        cpu - memory
    ).abs()

    # Equal weighting is an initial heuristic, not
    # a measured estimate of actual resource capacity.
    df["Utilization_Score"] = (
        0.5 * cpu + 0.5 * memory
    )

    df["Idle_Score"] = (
        100 - df["Utilization_Score"]
    ).clip(lower=0, upper=100)

    # ------------------------------------------------------
    # 4.4 Cost per usage unit
    # ------------------------------------------------------

    df["Cost_per_Usage_Unit"] = np.where(
        df["Usage_Quantity"] > 0,
        df["Total_Cost_INR"] / df["Usage_Quantity"],
        np.nan,
    )

    # ------------------------------------------------------
    # 4.5 Cost per hour
    # ------------------------------------------------------

    df["Cost_per_Hour"] = np.where(
        df["Usage_Duration_Hours"] > 0,
        df["Total_Cost_INR"]
        / df["Usage_Duration_Hours"],
        np.nan,
    )

    # ------------------------------------------------------
    # 4.6 Cost per network byte
    # ------------------------------------------------------

    df["Cost_per_Network_Byte"] = np.where(
        df["Total_Network_Data_Bytes"] > 0,
        df["Total_Cost_INR"]
        / df["Total_Network_Data_Bytes"],
        np.nan,
    )

    # ------------------------------------------------------
    # 4.7 Temporal features
    # ------------------------------------------------------

    start = df["Usage_Start_Date"]

    df["Usage_Start_Hour"] = start.dt.hour
    df["Usage_Day_of_Week"] = start.dt.dayofweek
    df["Usage_Month"] = start.dt.month
    df["Usage_Year"] = start.dt.year
    df["Usage_Day"] = start.dt.day

    # Cyclical hour features preserve the relationship
    # between 23:00 and 00:00.
    df["Hour_Sin"] = np.sin(
        2 * np.pi * start.dt.hour / 24
    )

    df["Hour_Cos"] = np.cos(
        2 * np.pi * start.dt.hour / 24
    )

    return df


train_df = create_basic_features(train_df)
test_df = create_basic_features(test_df)


# ==========================================================
# 5. BUILD BENCHMARKS USING TRAINING DATA ONLY
# ==========================================================

print("\nCreating training-only benchmarks...")

# A global fallback allows unseen services or regions
# to be handled without using information from the test set.
global_median_cost = train_df["Total_Cost_INR"].median()

if pd.isna(global_median_cost):
    raise ValueError(
        "Cannot calculate a global cost benchmark."
    )

service_median_cost = (
    train_df.groupby("Service_Name")["Total_Cost_INR"]
    .median()
)

region_median_cost = (
    train_df.groupby("Region_Zone")["Total_Cost_INR"]
    .median()
)


def add_benchmark_features(df):
    df = df.copy()

    # Identify categories not observed during training.
    df["Is_Unseen_Service"] = (
        ~df["Service_Name"].isin(service_median_cost.index)
    ).astype(int)

    df["Is_Unseen_Region"] = (
        ~df["Region_Zone"].isin(region_median_cost.index)
    ).astype(int)

    # Use only training-derived benchmarks.
    df["Service_Median_Cost"] = (
        df["Service_Name"]
        .map(service_median_cost)
        .fillna(global_median_cost)
    )

    df["Region_Median_Cost"] = (
        df["Region_Zone"]
        .map(region_median_cost)
        .fillna(global_median_cost)
    )

    # Cost relative to the typical cost for the service.
    df["Cost_vs_Service_Median"] = np.where(
        df["Service_Median_Cost"] > 0,
        df["Total_Cost_INR"]
        / df["Service_Median_Cost"],
        np.nan,
    )

    # Cost relative to the typical cost for the region.
    df["Cost_vs_Region_Median"] = np.where(
        df["Region_Median_Cost"] > 0,
        df["Total_Cost_INR"]
        / df["Region_Median_Cost"],
        np.nan,
    )

    return df


train_df = add_benchmark_features(train_df)
test_df = add_benchmark_features(test_df)


# ==========================================================
# 6. LOW UTILIZATION INDICATOR
# ==========================================================

# Missing CPU or memory data must not automatically classify
# a resource as underutilized.
for df in [train_df, test_df]:
    df["Is_Low_Utilization"] = (
        df["Utilization_Score"].notna()
        & (
            df["Utilization_Score"]
            < LOW_UTILIZATION_THRESHOLD
        )
    ).astype(int)


# ==========================================================
# 7. HIGH COST INDICATOR
# ==========================================================

# The threshold is calculated from training data only.
cost_threshold = train_df["Total_Cost_INR"].quantile(
    HIGH_COST_QUANTILE
)

for df in [train_df, test_df]:
    df["Is_High_Cost"] = (
        df["Total_Cost_INR"] >= cost_threshold
    ).astype(int)

print(
    "\nTraining 90th-percentile cost threshold:",
    round(cost_threshold, 2),
    "INR",
)


# ==========================================================
# 8. INITIAL OPTIMIZATION SIGNAL
# ==========================================================

for df in [train_df, test_df]:
    # A candidate requires both low utilization and a cost
    # at least 25% above the service benchmark.
    # Unseen services are excluded because their benchmark
    # is only a global fallback.
    df["Potential_Optimization"] = (
        (df["Is_Low_Utilization"] == 1)
        & (
            df["Cost_vs_Service_Median"]
            >= SERVICE_COST_MULTIPLIER
        )
        & (df["Is_Unseen_Service"] == 0)
    ).astype(int)


# ==========================================================
# 9. DATA QUALITY CHECKS
# ==========================================================

for name, df in [
    ("Training", train_df),
    ("Testing", test_df),
]:
    print(f"\n{name} data quality:")

    print(
        "Missing start dates:",
        df["Usage_Start_Date"].isna().sum(),
    )

    print(
        "Missing total costs:",
        df["Total_Cost_INR"].isna().sum(),
    )

    print(
        "Invalid durations:",
        df["Has_Invalid_Duration"].sum(),
    )

    print(
        "Missing utilization:",
        df["Has_Missing_Utilization"].sum(),
    )

    print(
        "Potential optimization candidates:",
        df["Potential_Optimization"].sum(),
    )


# ==========================================================
# 10. SAVE ENGINEERED DATASETS
# ==========================================================

train_df.to_csv(TRAIN_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)

print("\nFeature engineering completed.")
print("--------------------------------")

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

new_columns = [
    column
    for column in train_df.columns
    if column not in ORIGINAL_COLUMNS
]

print("\nEngineered features created:")

for column in new_columns:
    print("-", column)

print("\nOutput files:")
print("-", TRAIN_OUTPUT)
print("-", TEST_OUTPUT)

print("\nAll original rows preserved:")
print(
    "Training:",
    len(train_df),
    "| Testing:",
    len(test_df),
    "| Total:",
    len(train_df) + len(test_df),
)
