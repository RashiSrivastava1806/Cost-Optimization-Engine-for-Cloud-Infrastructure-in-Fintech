import pandas as pd

file_path = "data/cleaned_cloud_billing_data.csv"

df = pd.read_csv(file_path)

print("Dataset shape:")
print(df.shape)

print("\nColumns:")
for i, column in enumerate(df.columns, start=1):
    print(f"{i}. {column}")

print("\nData types:")
print(df.dtypes)

print("\nFirst 5 rows:")
print(df.head())