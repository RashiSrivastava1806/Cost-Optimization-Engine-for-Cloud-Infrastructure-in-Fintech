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
     # D. Numeric summaries
    print("\n[D] NUMERIC SUMMARY")
    print(df.describe().T.to_string())

    # E. Invalid utilization values
    print("\n[E] UTILIZATION VALIDATION")

    for col in utilization_columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")

            print(
                f"{col}: below 0 = {(values < 0).sum()}, "
                f"above 100 = {(values > 100).sum()}"
            )
     # F. Negative costs and quantities
    print("\n[F] COST AND USAGE VALIDATION")

    for col in cost_columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            print(f"{col}: negative values = {(values < 0).sum()}")

    if "Usage_Quantity" in df.columns:
        quantity = pd.to_numeric(
            df["Usage_Quantity"], errors="coerce"
        )
        print(
            "Usage_Quantity: negative values =",
            int((quantity < 0).sum())
        )
        print(
            "Usage_Quantity: zero values =",
            int((quantity == 0).sum())
        )
    # G. Negative network measurements
    print("\n[G] NETWORK VALIDATION")

    for col in network_columns:
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce")
            print(
                f"{col}: negative values = {(values < 0).sum()}"
            )

    # H. Date validation
    print("\n[H] DATE VALIDATION")

    for col in date_columns:
        if col in df.columns:
            dates = pd.to_datetime(df[col], errors="coerce")
            print(f"{col}: invalid/missing dates = {dates.isna().sum()}")

    if all(col in df.columns for col in date_columns):
        start = pd.to_datetime(df["Usage_Start_Date"], errors="coerce")
        end = pd.to_datetime(df["Usage_End_Date"], errors="coerce")

        print("End before start:", int((end < start).sum()))

     # I. Infinite numeric values
    print("\n[I] INFINITE VALUES")

    numeric_df = df.select_dtypes(include=[np.number])
    infinite_counts = np.isinf(numeric_df).sum()
    infinite_counts = infinite_counts[infinite_counts > 0]

    if infinite_counts.empty:
        print("No infinite numeric values found.")
    else:
        print(infinite_counts.to_string())
                  