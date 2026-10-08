import pandas as pd
import numpy as np
# 1. File paths
files = {
    "Train": "data/engineered_train_cloud_billing_data.csv",
    "Test": "data/engineered_test_cloud_billing_data.csv"
}

# 2. Columns and expected ranges
utilization_columns = [
    "CPU_Utilization",
    "Memory_Utilization"
]

cost_columns = [
    "Unrounded_Cost",
    "Rounded_Cost",
    "Total_Cost_INR"
]

network_columns = [
    "Network_Inbound_Data_Bytes",
    "Network_Outbound_Data_Bytes",
    "Total_Network_Data_Bytes"
]

date_columns = [
    "Usage_Start_Date",
    "Usage_End_Date"
]

# 3. Validate each dataset
for dataset_name, file_path in files.items():

    print("\n" + "=" * 65)
    print(f"{dataset_name.upper()} DATASET VALIDATION")
    print("=" * 65)

    df = pd.read_csv(file_path)

    # A. Dataset structure
    print("\n[A] DATASET STRUCTURE")
    print("Rows:", len(df))
    print("Columns:", len(df.columns))
    print("Shape:", df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes.to_string())

    # B. Missing values
    print("\n[B] MISSING VALUES")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        print("No missing values found.")
    else:
        print(missing.sort_values(ascending=False).to_string())

    print(
        "Total missing cells:",
        int(df.isnull().sum().sum())
    )

    # C. Duplicate records
    print("\n[C] DUPLICATE RECORDS")
    print("Exact duplicate rows:", int(df.duplicated().sum()))

    if "Resource_ID" in df.columns:
        print(
            "Repeated Resource_ID rows:",
            int(df["Resource_ID"].duplicated().sum())
        )