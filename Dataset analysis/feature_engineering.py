import pandas as pd
import numpy as np


# ==========================================================
# 1. FILE PATHS
# ==========================================================

train_input_path = "data/train_cloud_billing_data.csv"
test_input_path = "data/test_cloud_billing_data.csv"

train_output_path = "data/engineered_train_cloud_billing_data.csv"
test_output_path = "data/engineered_test_cloud_billing_data.csv"


# ==========================================================
# 2. LOAD TRAINING AND TESTING DATA
# ==========================================================

train_df = pd.read_csv(train_input_path)
test_df = pd.read_csv(test_input_path)

print("Datasets loaded successfully.")
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))


# ==========================================================
# 3. CONVERT DATE COLUMNS
# ==========================================================

date_columns = [
    "Usage_Start_Date",
    "Usage_End_Date"
]

for column in date_columns:

    if column in train_df.columns:
        train_df[column] = pd.to_datetime(
            train_df[column],
            errors="coerce"
        )

    if column in test_df.columns:
        test_df[column] = pd.to_datetime(
            test_df[column],
            errors="coerce"
        )


# ==========================================================
# 4. BASIC FEATURE ENGINEERING FUNCTION
# ==========================================================

def create_basic_features(df):

    df = df.copy()

    # ------------------------------------------------------
    # 4.1 Usage Duration
    # ------------------------------------------------------

    if (
        "Usage_Start_Date" in df.columns
        and "Usage_End_Date" in df.columns
    ):

        duration = (
            df["Usage_End_Date"]
            - df["Usage_Start_Date"]
        )

        df["Usage_Duration_Hours"] = (
            duration.dt.total_seconds() / 3600
        )

        # Prevent negative durations
        df["Usage_Duration_Hours"] = (
            df["Usage_Duration_Hours"].clip(lower=0)
        )


    # ------------------------------------------------------
    # 4.2 Total Network Data
    # ------------------------------------------------------

    if (
        "Network_Inbound_Data_Bytes" in df.columns
        and "Network_Outbound_Data_Bytes" in df.columns
    ):

        df["Total_Network_Data_Bytes"] = (
            df["Network_Inbound_Data_Bytes"].fillna(0)
            +
            df["Network_Outbound_Data_Bytes"].fillna(0)
        )


    # ------------------------------------------------------
    # 4.3 CPU + Memory Average
    # ------------------------------------------------------

    if (
        "CPU_Utilization" in df.columns
        and "Memory_Utilization" in df.columns
    ):

        df["CPU_Memory_Avg"] = (
            df["CPU_Utilization"]
            +
            df["Memory_Utilization"]
        ) / 2


    # ------------------------------------------------------
    # 4.4 CPU-Memory Gap
    # ------------------------------------------------------

    if (
        "CPU_Utilization" in df.columns
        and "Memory_Utilization" in df.columns
    ):

        df["CPU_Memory_Gap"] = (
            abs(
                df["CPU_Utilization"]
                -
                df["Memory_Utilization"]
            )
        )


    # ------------------------------------------------------
    # 4.5 Overall Utilization Score
    # ------------------------------------------------------

    if (
        "CPU_Utilization" in df.columns
        and "Memory_Utilization" in df.columns
    ):

        df["Utilization_Score"] = (
            0.5 * df["CPU_Utilization"]
            +
            0.5 * df["Memory_Utilization"]
        )


    # ------------------------------------------------------
    # 4.6 Idle Score
    # ------------------------------------------------------

    if "Utilization_Score" in df.columns:

        df["Idle_Score"] = (
            100 - df["Utilization_Score"]
        ).clip(lower=0)


    # ------------------------------------------------------
    # 4.7 Cost per Usage Unit
    # ------------------------------------------------------

    if (
        "Total_Cost_INR" in df.columns
        and "Usage_Quantity" in df.columns
    ):

        df["Cost_per_Usage_Unit"] = np.where(
            df["Usage_Quantity"] > 0,
            df["Total_Cost_INR"]
            / df["Usage_Quantity"],
            np.nan
        )


    # ------------------------------------------------------
    # 4.8 Cost per Hour
    # ------------------------------------------------------

    if (
        "Total_Cost_INR" in df.columns
        and "Usage_Duration_Hours" in df.columns
    ):

        df["Cost_per_Hour"] = np.where(
            df["Usage_Duration_Hours"] > 0,
            df["Total_Cost_INR"]
            / df["Usage_Duration_Hours"],
            np.nan
        )


    # ------------------------------------------------------
    # 4.9 Cost per Network Byte
    # ------------------------------------------------------

    if (
        "Total_Cost_INR" in df.columns
        and "Total_Network_Data_Bytes" in df.columns
    ):

        df["Cost_per_Network_Byte"] = np.where(
            df["Total_Network_Data_Bytes"] > 0,
            df["Total_Cost_INR"]
            / df["Total_Network_Data_Bytes"],
            np.nan
        )


    # ------------------------------------------------------
    # 4.10 Temporal Features
    # ------------------------------------------------------

    if "Usage_Start_Date" in df.columns:

        df["Usage_Start_Hour"] = (
            df["Usage_Start_Date"].dt.hour
        )

        df["Usage_Day_of_Week"] = (
            df["Usage_Start_Date"].dt.dayofweek
        )

        df["Usage_Month"] = (
            df["Usage_Start_Date"].dt.month
        )

        df["Usage_Year"] = (
            df["Usage_Start_Date"].dt.year
        )

        df["Usage_Day"] = (
            df["Usage_Start_Date"].dt.day
        )


    return df


# ==========================================================
# 5. CREATE BASIC FEATURES
# ==========================================================

train_df = create_basic_features(train_df)
test_df = create_basic_features(test_df)


# ==========================================================
# 6. CREATE TRAINING-SET BENCHMARKS
# ==========================================================

print("\nCreating training-set benchmarks...")


# Service median cost
service_median_cost = (
    train_df
    .groupby("Service_Name")["Total_Cost_INR"]
    .median()
)


# Region median cost
region_median_cost = (
    train_df
    .groupby("Region_Zone")["Total_Cost_INR"]
    .median()
)


# ==========================================================
# 7. APPLY SERVICE BENCHMARK
# ==========================================================

train_df["Service_Median_Cost"] = (
    train_df["Service_Name"]
    .map(service_median_cost)
)

test_df["Service_Median_Cost"] = (
    test_df["Service_Name"]
    .map(service_median_cost)
)


# ==========================================================
# 8. APPLY REGION BENCHMARK
# ==========================================================

train_df["Region_Median_Cost"] = (
    train_df["Region_Zone"]
    .map(region_median_cost)
)

test_df["Region_Median_Cost"] = (
    test_df["Region_Zone"]
    .map(region_median_cost)
)


# ==========================================================
# 9. COST VS SERVICE MEDIAN
# ==========================================================

train_df["Cost_vs_Service_Median"] = np.where(
    train_df["Service_Median_Cost"] > 0,
    train_df["Total_Cost_INR"]
    / train_df["Service_Median_Cost"],
    np.nan
)

test_df["Cost_vs_Service_Median"] = np.where(
    test_df["Service_Median_Cost"] > 0,
    test_df["Total_Cost_INR"]
    / test_df["Service_Median_Cost"],
    np.nan
)


# ==========================================================
# 10. COST VS REGION MEDIAN
# ==========================================================

train_df["Cost_vs_Region_Median"] = np.where(
    train_df["Region_Median_Cost"] > 0,
    train_df["Total_Cost_INR"]
    / train_df["Region_Median_Cost"],
    np.nan
)

test_df["Cost_vs_Region_Median"] = np.where(
    test_df["Region_Median_Cost"] > 0,
    test_df["Total_Cost_INR"]
    / test_df["Region_Median_Cost"],
    np.nan
)


# ==========================================================
# 11. LOW UTILIZATION FLAG
# ==========================================================

train_df["Is_Low_Utilization"] = (
    train_df["Utilization_Score"] < 30
).astype(int)

test_df["Is_Low_Utilization"] = (
    test_df["Utilization_Score"] < 30
).astype(int)


# ==========================================================
# 12. HIGH COST FLAG
# ==========================================================

cost_threshold = (
    train_df["Total_Cost_INR"]
    .quantile(0.90)
)

print("\nHigh-cost threshold:", cost_threshold)


train_df["Is_High_Cost"] = (
    train_df["Total_Cost_INR"]
    >= cost_threshold
).astype(int)

test_df["Is_High_Cost"] = (
    test_df["Total_Cost_INR"]
    >= cost_threshold
).astype(int)


# ==========================================================
# 13. INITIAL OPTIMIZATION SIGNAL
# ==========================================================

train_df["Potential_Optimization"] = (
    (
        (train_df["Is_Low_Utilization"] == 1)
        &
        (train_df["Cost_vs_Service_Median"] >= 1.25)
    )
).astype(int)


test_df["Potential_Optimization"] = (
    (
        (test_df["Is_Low_Utilization"] == 1)
        &
        (test_df["Cost_vs_Service_Median"] >= 1.25)
    )
).astype(int)


# ==========================================================
# 14. SAVE ENGINEERED DATASETS
# ==========================================================

train_df.to_csv(
    train_output_path,
    index=False
)

test_df.to_csv(
    test_output_path,
    index=False
)


# ==========================================================
# 15. DISPLAY RESULTS
# ==========================================================

print("\nFeature Engineering Completed")
print("--------------------------------")

print("Training dataset shape:", train_df.shape)
print("Testing dataset shape:", test_df.shape)

print("\nNew features created:")

original_columns = [
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
    "Total_Cost_INR"
]

new_columns = [
    column
    for column in train_df.columns
    if column not in original_columns
]

for column in new_columns:
    print("-", column)


# ==========================================================
# 16. OUTPUT FILES
# ==========================================================

print("\nOutput files:")
print("-", train_output_path)
print("-", test_output_path)