
from pathlib import Path

path = Path("Dataset analysis/feature_engineering_canonical.py")
code = path.read_text()

old_paths = '''TRAIN_INPUT = DATA_DIR / "train_cloud_billing_data.csv"
TEST_INPUT = DATA_DIR / "test_cloud_billing_data.csv"

TRAIN_OUTPUT = DATA_DIR / "engineered_train_cloud_billing_data.csv"
TEST_OUTPUT = DATA_DIR / "engineered_test_cloud_billing_data.csv"'''

new_paths = '''CANONICAL_DATA_DIR = DATA_DIR / "canonical_run"

TRAIN_INPUT = CANONICAL_DATA_DIR / "train_cloud_billing_data.csv"
TEST_INPUT = CANONICAL_DATA_DIR / "test_cloud_billing_data.csv"

TRAIN_OUTPUT = CANONICAL_DATA_DIR / "engineered_train_cloud_billing_data.csv"
TEST_OUTPUT = CANONICAL_DATA_DIR / "engineered_test_cloud_billing_data.csv"'''

if old_paths not in code:
    raise SystemExit("Expected path block not found. No changes made.")

code = code.replace(old_paths, new_paths, 1)

old_dates = '''df[column] = pd.to_datetime(
            df[column],
            errors="raise",
        )'''

new_dates = '''df[column] = pd.to_datetime(
            df[column],
            format="%Y-%m-%d %H:%M:%S",
            errors="raise",
        )'''

if old_dates not in code:
    raise SystemExit("Expected date-parsing block not found. No changes saved.")

code = code.replace(old_dates, new_dates, 1)
path.write_text(code)

print("Canonical feature-engineering script updated.")
print("Original feature_engineering.py remains unchanged.")
