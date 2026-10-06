import pandas as pd
import csv
import os

# 1. FILE PATHS
input_file = "data/cloud_billing_data.csv"
output_file = "data/cleaned_cloud_billing_data.csv"

# 2. DETECT CSV SEPARATOR
with open(input_file, "r", encoding="utf-8-sig") as file:
    sample = file.read(5000)

try:
    dialect = csv.Sniffer().sniff(
        sample,
        delimiters=",;\t|"
    )
    separator = dialect.delimiter
except csv.Error:
    separator = ","


print("=" * 70)
print("FINOPTICLOUD - DATA PREPROCESSING")
print("=" * 70)

print("\nDetected separator:", repr(separator))

# 3. LOAD DATASET
df = pd.read_csv(
    input_file,
    sep=separator
)

print("\nDataset loaded successfully.")

print("Original rows:", len(df))
print("Original columns:", len(df.columns))

# 4. CLEAN COLUMN NAMES
print("\n" + "=" * 70)
print("STEP 1: CLEANING COLUMN NAMES")
print("=" * 70)

# Remove leading/trailing spaces
df.columns = df.columns.str.strip()

# Replace spaces with underscores
df.columns = df.columns.str.replace(" ", "_")

# Remove special characters where possible
df.columns = df.columns.str.replace(
    r"[^\w]+",
    "_",
    regex=True
)

# Remove repeated underscores
df.columns = df.columns.str.replace(
    r"_+",
    "_",
    regex=True
)

# Remove underscores from beginning/end
df.columns = df.columns.str.strip("_")

print("Cleaned column names:")

for column in df.columns:
    print("-", column)

# 5. REMOVE COMPLETELY EMPTY ROWS
print("\n" + "=" * 70)
print("STEP 2: REMOVING COMPLETELY EMPTY ROWS")
print("=" * 70)

before = len(df)

df = df.dropna(
    how="all"
)

after = len(df)

print("Empty rows removed:", before - after)

# 6. REMOVE COMPLETELY EMPTY COLUMNS
print("\n" + "=" * 70)
print("STEP 3: REMOVING COMPLETELY EMPTY COLUMNS")
print("=" * 70)

empty_columns = df.columns[
    df.isna().all()
].tolist()

if empty_columns:
    print("Empty columns removed:")

    for column in empty_columns:
        print("-", column)

    df = df.drop(
        columns=empty_columns
    )

else:
    print("No completely empty columns found.")

# 7. REMOVE DUPLICATE ROWS
print("\n" + "=" * 70)
print("STEP 4: DUPLICATE CHECK")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print("Duplicate rows found:", duplicate_count)

if duplicate_count > 0:

    df = df.drop_duplicates()

    print(
        "Duplicate rows removed:",
        duplicate_count
    )

else:
    print("No duplicate rows found.")

# 8. REMOVE EXTRA WHITESPACE FROM TEXT
print("\n" + "=" * 70)
print("STEP 5: CLEANING TEXT VALUES")
print("=" * 70)

text_columns = df.select_dtypes(
    include="object"
).columns

for column in text_columns:

    df[column] = (
        df[column]
        .astype("string")
        .str.strip()
    )

print(
    "Whitespace cleaned from",
    len(text_columns),
    "text columns."
)

# 9. DETECT DATE COLUMNS
print("\n" + "=" * 70)
print("STEP 6: DATE COLUMN DETECTION")
print("=" * 70)

date_columns = []

for column in df.columns:

    column_name = column.lower()

    if any(
        keyword in column_name
        for keyword in [
            "date",
            "time",
            "timestamp"
        ]
    ):

        date_columns.append(column)


print("Potential date columns:")

if date_columns:

    for column in date_columns:
        print("-", column)

else:

    print("No obvious date column found.")

# 10. CONVERT DATE COLUMNS
print("\n" + "=" * 70)
print("STEP 7: CONVERTING DATE COLUMNS")
print("=" * 70)

for column in date_columns:

    original_non_null = df[column].notna().sum()

    converted = pd.to_datetime(
        df[column],
        errors="coerce"
    )

    converted_non_null = converted.notna().sum()

    success_rate = (
        converted_non_null /
        original_non_null
        if original_non_null > 0
        else 0
    )

    # Only convert if most values were successfully recognized
    if success_rate >= 0.80:

        df[column] = converted

        print(
            f"{column}: converted to datetime "
            f"({success_rate * 100:.2f}% recognized)"
        )

    else:

        print(
            f"{column}: NOT converted "
            f"({success_rate * 100:.2f}% recognized)"
        )

# 11. CHECK NUMERICAL COLUMNS
print("\n" + "=" * 70)
print("STEP 8: NUMERICAL COLUMN CHECK")
print("=" * 70)

numeric_columns = df.select_dtypes(
    include="number"
).columns

print("Numerical columns:")

for column in numeric_columns:
    print("-", column)

# 12. MISSING VALUE ANALYSIS
print("\n" + "=" * 70)
print("STEP 9: MISSING VALUE ANALYSIS")
print("=" * 70)

missing_count = df.isnull().sum()

missing_percentage = (
    missing_count /
    len(df)
) * 100

missing_report = pd.DataFrame({
    "Missing_Count": missing_count,
    "Missing_Percentage": missing_percentage
})

missing_report = missing_report[
    missing_report["Missing_Count"] > 0
]

if len(missing_report) > 0:

    print(missing_report)

else:

    print("No missing values found.")

# 13. NUMERICAL SUMMARY
print("\n" + "=" * 70)
print("STEP 10: NUMERICAL SUMMARY")
print("=" * 70)

if len(numeric_columns) > 0:

    print(
        df[numeric_columns].describe().T
    )

else:

    print("No numerical columns found.")

# 14. CHECK NEGATIVE VALUES
print("\n" + "=" * 70)
print("STEP 11: NEGATIVE VALUE CHECK")
print("=" * 70)

for column in numeric_columns:

    negative_count = (
        df[column] < 0
    ).sum()

    if negative_count > 0:

        print(
            f"{column}: "
            f"{negative_count} negative values"
        )

    else:

        print(
            f"{column}: no negative values"
        )

# 15. CHECK ZERO VALUES
print("\n" + "=" * 70)
print("STEP 12: ZERO VALUE CHECK")
print("=" * 70)

for column in numeric_columns:

    zero_count = (
        df[column] == 0
    ).sum()

    print(
        f"{column}: {zero_count} zero values"
    )

# 16. CHECK POTENTIAL COST COLUMNS
print("\n" + "=" * 70)
print("STEP 13: POTENTIAL COST COLUMNS")
print("=" * 70)

cost_columns = []

for column in df.columns:

    name = column.lower()

    if any(
        keyword in name
        for keyword in [
            "cost",
            "price",
            "amount",
            "charge",
            "spend",
            "billing"
        ]
    ):

        cost_columns.append(column)


if cost_columns:

    for column in cost_columns:
        print("-", column)

else:

    print("No obvious cost column found.")

# 17. FINAL DATASET INFORMATION

print("\n" + "=" * 70)
print("FINAL DATASET")
print("=" * 70)

print("Rows:", len(df))
print("Columns:", len(df.columns))

print("\nColumns:")

for column in df.columns:
    print("-", column)

# 18. SAVE CLEANED DATASET
df.to_csv(
    output_file,
    index=False
)

print("\n" + "=" * 70)
print("PREPROCESSING COMPLETE")
print("=" * 70)

print(
    f"\nCleaned dataset saved to:\n{output_file}"
)