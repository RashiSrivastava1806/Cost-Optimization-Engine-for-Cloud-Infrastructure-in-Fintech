
import pandas as pd
from pathlib import Path

# --------------------------------------------------
# 1. FILE PATHS
# --------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "cleaned_cloud_billing_data.csv"
TRAIN_FILE = DATA_DIR / "train_cloud_billing_data.csv"
TEST_FILE = DATA_DIR / "test_cloud_billing_data.csv"

DATE_COLUMN = "Usage_Start_Date"
TRAIN_RATIO = 0.80


# --------------------------------------------------
# 2. LOAD CLEANED DATASET
# --------------------------------------------------
print("=" * 70)
print("FINOPTICLOUD - CHRONOLOGICAL DATASET SPLITTING")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print("\nDataset loaded successfully.")
print("Total rows:", len(df))
print("Total columns:", len(df.columns))


# --------------------------------------------------
# 3. VALIDATE DATE COLUMN
# --------------------------------------------------
if DATE_COLUMN not in df.columns:
    raise ValueError(
        f"Required date column '{DATE_COLUMN}' not found. "
        f"Available columns: {df.columns.tolist()}"
    )


# --------------------------------------------------
# 4. PARSE CLEANED ISO-FORMAT DATES
# --------------------------------------------------
# Preprocessing has already converted the source timestamps
# from DD/MM/YYYY HH:MM into ISO-style timestamps.
df[DATE_COLUMN] = pd.to_datetime(
    df[DATE_COLUMN],
    format="ISO8601",
    errors="raise"
)

if df[DATE_COLUMN].isna().any():
    raise ValueError(
        "Missing dates found. Resolve them before splitting."
    )

print("\nDate validation successful.")
print("Earliest timestamp:", df[DATE_COLUMN].min())
print("Latest timestamp:", df[DATE_COLUMN].max())
print(
    "Unique calendar dates:",
    df[DATE_COLUMN].dt.normalize().nunique()
)


# --------------------------------------------------
# 5. SORT DATA CHRONOLOGICALLY
# --------------------------------------------------
df = df.sort_values(
    by=DATE_COLUMN
).reset_index(drop=True)


# --------------------------------------------------
# 6. SPLIT BY UNIQUE CALENDAR DATES
# --------------------------------------------------
# All records from one calendar day remain in the same split.
# This avoids putting records from the same day in both sets.

df["_Calendar_Date"] = df[DATE_COLUMN].dt.normalize()

unique_dates = sorted(
    df["_Calendar_Date"].unique()
)

if len(unique_dates) < 2:
    raise ValueError(
        "At least two unique calendar dates are required "
        "for a chronological train/test split."
    )

split_index = int(len(unique_dates) * TRAIN_RATIO)

# Ensure both datasets contain at least one calendar date.
split_index = max(
    1,
    min(split_index, len(unique_dates) - 1)
)

cutoff_date = unique_dates[split_index]

train_df = df[
    df["_Calendar_Date"] < cutoff_date
].copy()

test_df = df[
    df["_Calendar_Date"] >= cutoff_date
].copy()

# Remove the temporary helper column.
train_df.drop(
    columns=["_Calendar_Date"],
    inplace=True
)

test_df.drop(
    columns=["_Calendar_Date"],
    inplace=True
)


# --------------------------------------------------
# 7. VALIDATE THE SPLIT
# --------------------------------------------------
if train_df.empty or test_df.empty:
    raise ValueError(
        "Train/test split produced an empty dataset."
    )

train_max_date = train_df[DATE_COLUMN].max()
test_min_date = test_df[DATE_COLUMN].min()

if train_max_date >= test_min_date:
    raise ValueError(
        "Chronological split failed: training and testing "
        "timestamps overlap."
    )

train_dates = train_df[DATE_COLUMN].dt.normalize().nunique()
test_dates = test_df[DATE_COLUMN].dt.normalize().nunique()


# --------------------------------------------------
# 8. SAVE TRAINING AND TESTING DATASETS
# --------------------------------------------------
train_df.to_csv(
    TRAIN_FILE,
    index=False
)

test_df.to_csv(
    TEST_FILE,
    index=False
)


# --------------------------------------------------
# 9. DISPLAY RESULTS
# --------------------------------------------------
print("\n" + "=" * 70)
print("DATASET SPLIT COMPLETED")
print("=" * 70)

print("\nTraining dataset")
print("----------------")
print("Rows:", len(train_df))
print("Unique calendar dates:", train_dates)
print("Date range:", train_df[DATE_COLUMN].min(),
      "to", train_df[DATE_COLUMN].max())
print(
    "Percentage of rows:",
    round(len(train_df) / len(df) * 100, 2),
    "%"
)

print("\nTesting dataset")
print("---------------")
print("Rows:", len(test_df))
print("Unique calendar dates:", test_dates)
print("Date range:", test_df[DATE_COLUMN].min(),
      "to", test_df[DATE_COLUMN].max())
print(
    "Percentage of rows:",
    round(len(test_df) / len(df) * 100, 2),
    "%"
)

print("\nChronological Split Check")
print("-------------------------")
print("Training ends:", train_max_date)
print("Testing begins:", test_min_date)
print("PASS: Training occurs strictly before testing.")

print("\nFiles created:")
print(TRAIN_FILE)
print(TEST_FILE)

print("\nTotal rows preserved:",
      len(train_df) + len(test_df), "/", len(df))

print("\nDataset splitting completed successfully.")
