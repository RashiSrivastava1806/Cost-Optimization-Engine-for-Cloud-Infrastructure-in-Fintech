import pandas as pd

# --------------------------------------------------
# 1. Load cleaned dataset
# --------------------------------------------------
input_path = "data/cleaned_cloud_billing_data.csv"

df = pd.read_csv(input_path)

print("Cleaned dataset loaded successfully.")
print("Total rows:", len(df))


# --------------------------------------------------
# 2. Display column names
# --------------------------------------------------
print("\nColumns in cleaned dataset:")
for column in df.columns:
    print("-", column)


# --------------------------------------------------
# 3. Identify the date column
# --------------------------------------------------
date_column = None

possible_date_columns = [
    "Usage Start Date",
    "Start Date",
    "Usage Date",
    "Date",
    "Timestamp",
    "Usage_Start_Date",
    "usage_start_date"
]

for column in possible_date_columns:
    if column in df.columns:
        date_column = column
        break


# --------------------------------------------------
# 4. Stop if no date column is found
# --------------------------------------------------
if date_column is None:
    print("\nERROR: No suitable date column was found.")
    print("Please check the column names printed above.")
    raise SystemExit


print("\nDate column selected:", date_column)


# --------------------------------------------------
# 5. Convert date column to datetime
# --------------------------------------------------
df[date_column] = pd.to_datetime(
    df[date_column],
    errors="coerce"
)


# --------------------------------------------------
# 6. Check for invalid dates
# --------------------------------------------------
invalid_dates = df[date_column].isna().sum()

print("Invalid dates:", invalid_dates)

if invalid_dates > 0:
    print("Warning: Removing rows with invalid dates.")

    df = df.dropna(
        subset=[date_column]
    ).copy()


# --------------------------------------------------
# 7. Sort dataset chronologically
# --------------------------------------------------
df = df.sort_values(
    by=date_column
).reset_index(drop=True)


# --------------------------------------------------
# 8. Split dataset: 80% training, 20% testing
# --------------------------------------------------
split_index = int(len(df) * 0.80)

train_df = df.iloc[:split_index].copy()
test_df = df.iloc[split_index:].copy()


# --------------------------------------------------
# 9. Save training and testing datasets
# --------------------------------------------------
train_path = "data/train_cloud_billing_data.csv"
test_path = "data/test_cloud_billing_data.csv"

train_df.to_csv(
    train_path,
    index=False
)

test_df.to_csv(
    test_path,
    index=False
)


# --------------------------------------------------
# 10. Display split information
# --------------------------------------------------
print("\nDataset Split Completed")
print("-----------------------")

print("Total rows:", len(df))
print("Training rows:", len(train_df))
print("Testing rows:", len(test_df))

print(
    "Training percentage:",
    round(len(train_df) / len(df) * 100, 2),
    "%"
)

print(
    "Testing percentage:",
    round(len(test_df) / len(df) * 100, 2),
    "%"
)


# --------------------------------------------------
# 11. Display date ranges
# --------------------------------------------------
print("\nTraining date range:")
print(
    train_df[date_column].min(),
    "to",
    train_df[date_column].max()
)

print("\nTesting date range:")
print(
    test_df[date_column].min(),
    "to",
    test_df[date_column].max()
)


# --------------------------------------------------
# 12. Verify chronological separation
# --------------------------------------------------
train_max_date = train_df[date_column].max()
test_min_date = test_df[date_column].min()

print("\nChronological Split Check:")

if train_max_date <= test_min_date:
    print("PASS: Training data occurs before testing data.")
else:
    print("WARNING: Training and testing dates overlap.")


# --------------------------------------------------
# 13. Confirm output files
# --------------------------------------------------
print("\nFiles created:")
print("-", train_path)
print("-", test_path)