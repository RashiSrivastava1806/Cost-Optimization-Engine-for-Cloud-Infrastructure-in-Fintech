
from pathlib import Path

import pandas as pd


# ==========================================================
# 1. PATHS AND CONFIGURATION
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

INPUT_FILE = DATA_DIR / "cleaned_cloud_billing_data_canonical.csv"
OUTPUT_DIR = DATA_DIR / "canonical_run"

TRAIN_FILE = OUTPUT_DIR / "train_cloud_billing_data.csv"
TEST_FILE = OUTPUT_DIR / "test_cloud_billing_data.csv"

DATE_COLUMN = "Usage_Start_Date"
END_DATE_COLUMN = "Usage_End_Date"
TRAIN_RATIO = 0.80

EXPECTED_TOTAL_ROWS = 1000


# ==========================================================
# 2. LOAD CANONICAL DATA
# ==========================================================

print("=" * 65)
print("FINOPTICLOUD - CANONICAL DATASET SPLITTING")
print("=" * 65)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"Canonical dataset not found: {INPUT_FILE}\n"
        "Run utils/standardize_dates.py first."
    )

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("The canonical dataset is empty.")

if len(df) != EXPECTED_TOTAL_ROWS:
    raise ValueError(
        f"Expected {EXPECTED_TOTAL_ROWS} rows, "
        f"but found {len(df)}. Check the input dataset."
    )

required_columns = [DATE_COLUMN, END_DATE_COLUMN]

missing_columns = [
    column for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ==========================================================
# 3. STRICT DATE VALIDATION
# ==========================================================

for column in required_columns:
    df[column] = pd.to_datetime(
        df[column],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )

    if df[column].isna().any():
        raise ValueError(
            f"Missing timestamps found in {column}."
        )

if (df[END_DATE_COLUMN] < df[DATE_COLUMN]).any():
    raise ValueError(
        "Invalid records found: usage end precedes usage start."
    )

print("\nCanonical data validated.")
print("Total rows:", len(df))
print("Earliest start:", df[DATE_COLUMN].min())
print("Latest start:", df[DATE_COLUMN].max())
print(
    "Unique start dates:",
    df[DATE_COLUMN].dt.normalize().nunique()
)


# ==========================================================
# 4. CHRONOLOGICAL SPLIT BY CALENDAR DATE
# ==========================================================

df = df.sort_values(DATE_COLUMN).reset_index(drop=True)

df["_Calendar_Date"] = df[DATE_COLUMN].dt.normalize()
unique_dates = sorted(df["_Calendar_Date"].unique())

if len(unique_dates) < 2:
    raise ValueError("At least two unique dates are required.")

split_index = int(len(unique_dates) * TRAIN_RATIO)
split_index = max(1, min(split_index, len(unique_dates) - 1))

cutoff_date = unique_dates[split_index]

train_df = df.loc[
    df["_Calendar_Date"] < cutoff_date
].drop(columns="_Calendar_Date").copy()

test_df = df.loc[
    df["_Calendar_Date"] >= cutoff_date
].drop(columns="_Calendar_Date").copy()


# ==========================================================
# 5. VALIDATE SPLIT
# ==========================================================

if train_df.empty or test_df.empty:
    raise ValueError("The split produced an empty dataset.")

train_max = train_df[DATE_COLUMN].max()
test_min = test_df[DATE_COLUMN].min()

if train_max >= test_min:
    raise ValueError(
        "Chronological leakage detected between train and test."
    )

if len(train_df) + len(test_df) != len(df):
    raise ValueError("Row counts do not reconcile after splitting.")

if set(train_df[DATE_COLUMN].dt.normalize()).intersection(
    set(test_df[DATE_COLUMN].dt.normalize())
):
    raise ValueError(
        "A calendar date appears in both train and test."
    )


# ==========================================================
# 6. SAVE TO SEPARATE OUTPUT DIRECTORY
# ==========================================================

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

train_df.to_csv(TRAIN_FILE, index=False)
test_df.to_csv(TEST_FILE, index=False)


# ==========================================================
# 7. REPORT
# ==========================================================

print("\n" + "=" * 65)
print("CANONICAL SPLIT COMPLETED")
print("=" * 65)

print("\nTraining dataset")
print("Rows:", len(train_df))
print(
    "Unique dates:",
    train_df[DATE_COLUMN].dt.normalize().nunique()
)
print("Range:", train_df[DATE_COLUMN].min(), "to", train_max)
print("Row percentage:", round(len(train_df) / len(df) * 100, 2), "%")

print("\nTesting dataset")
print("Rows:", len(test_df))
print(
    "Unique dates:",
    test_df[DATE_COLUMN].dt.normalize().nunique()
)
print("Range:", test_min, "to", test_df[DATE_COLUMN].max())
print("Row percentage:", round(len(test_df) / len(df) * 100, 2), "%")

print("\nSaved files:")
print(TRAIN_FILE)
print(TEST_FILE)

print("\nOriginal datasets and existing results were not overwritten.")

