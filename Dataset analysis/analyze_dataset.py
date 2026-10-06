import pandas as pd
import csv
# 1. LOAD DATASET
file_path = "data/cloud_billing_data.csv"

# Detect the separator used in the CSV file
with open(file_path, "r", encoding="utf-8-sig") as file:
    sample = file.read(5000)

try:
    dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
    separator = dialect.delimiter
except csv.Error:
    separator = ","

print("=" * 60)
print("DATASET LOADING")
print("=" * 60)

print("Detected separator:", repr(separator))

# Load the dataset using the detected separator
df = pd.read_csv(file_path, sep=separator)

print("Dataset loaded successfully!")

# 2. NUMBER OF ROWS AND COLUMNS 
rows, columns=df.shape
print("\n2.DATASET SIZE")
print("-" * 40)
print("Number of rows:", rows)
print("Number of columns:", columns)

# 3. COLUMN NAMES
print("\n3. COLUMN NAMES")
print("-" * 40)

for column in df.columns:
    print(column)

# 4. DATA TYPES
print("\n4. DATA TYPES")
print("-" * 40)

print(df.dtypes)

# 5. MISSING VALUES
print("\n5. MISSING VALUES")
print("-" * 40)

print(df.isnull().sum())

# 6. DUPLICATE ROWS
print("\n6. DUPLICATE ROWS")
print("-" * 40)

print("Number of duplicate rows:", df.duplicated().sum())

# 7. UNIQUE VALUES
print("\n7. UNIQUE VALUES PER COLUMN")
print("-" * 40)

for column in df.columns:
    print(column, ":", df[column].nunique())

# 8. STATISTICAL SUMMARY
print("\n8. STATISTICAL SUMMARY")
print("-" * 40)

print(df.describe(include="all"))

# 9. MEMORY USAGE
print("\n9. MEMORY USAGE")
print("-" * 40)

memory = df.memory_usage(deep=True).sum()

print("Memory usage:", memory, "bytes")

# 10. CATEGORICAL COLUMN VALUES
print("\n11. CATEGORICAL COLUMN VALUES")
print("-" * 40)

categorical_columns = df.select_dtypes(include="object").columns

for column in categorical_columns:
    print("\n" + column)
    print(df[column].unique())

# 11. NUMERICAL COLUMNS
print("\n11. NUMERICAL COLUMNS")
print("-" * 40)

numerical_columns = df.select_dtypes(include="number").columns

print(list(numerical_columns))

# 12. CATEGORICAL COLUMNS
print("\n12. CATEGORICAL COLUMNS")
print("-" * 40)

print(list(categorical_columns))

# 13. MISSING VALUE PERCENTAGE
print("\n13. MISSING VALUE PERCENTAGE")
print("-" * 40)

missing_percentage = (df.isnull().sum() / len(df)) * 100

print(missing_percentage)

# 14. POTENTIAL COST COLUMNS
print("\n14. POTENTIAL COST COLUMNS")
print("-" * 40)

for column in df.columns:
    if (
        "cost" in column.lower()
        or "price" in column.lower()
        or "amount" in column.lower()
    ):
        print("Possible cost column:", column)

# 15. DATASET PREVIEW
print("\n15. DATASET PREVIEW")
print("-" * 40)

print(df.head(10))


# ANALYSIS COMPLETE
print("\n" + "=" * 60)
print("INITIAL DATASET ANALYSIS COMPLETE")
print("=" * 60)


