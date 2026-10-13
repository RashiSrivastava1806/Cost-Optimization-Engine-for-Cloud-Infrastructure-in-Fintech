
from pathlib import Path
import shutil
import pandas as pd

# --------------------------------------------------
# PATHS
# --------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SOURCE_FILE = DATA_DIR / "cleaned_cloud_billing_data.csv"
OUTPUT_FILE = DATA_DIR / "cleaned_cloud_billing_data_canonical.csv"
AUDIT_FILE = DATA_DIR / "date_standardization_audit.csv"

DATE_COLUMNS = ["Usage_Start_Date", "Usage_End_Date"]

# Explicit supported formats. Slash dates are interpreted
# as month/day/year based on the previously validated timeline.
DATE_FORMATS = [
    "%Y-%m-%d %H:%M:%S",
    "%Y-%m-%d %H:%M",
    "%Y-%m-%d",
    "%m/%d/%Y %H:%M:%S",
    "%m/%d/%Y %H:%M",
    "%m/%d/%Y",
    "%m-%d-%Y %H:%M:%S",
    "%m-%d-%Y %H:%M",
    "%m-%d-%Y",
]


def parse_dates_strictly(series):
    """Parse only explicitly supported date formats."""
    original = series.astype("string").str.strip()
    parsed = pd.Series(
        pd.NaT,
        index=series.index,
        dtype="datetime64[ns]"
    )

    for fmt in DATE_FORMATS:
        remaining = parsed.isna() & original.notna() & original.ne("")
        if not remaining.any():
            break

        candidate = pd.to_datetime(
            original.loc[remaining],
            format=fmt,
            errors="coerce"
        )
        parsed.loc[remaining] = candidate

    # Empty values remain missing; unrecognized nonempty
    # values are reported separately.
    invalid = (
        original.notna()
        & original.ne("")
        & parsed.isna()
    )

    return original, parsed, invalid


def main():
    if not SOURCE_FILE.exists():
        raise FileNotFoundError(
            f"Source file not found: {SOURCE_FILE}"
        )

    if OUTPUT_FILE.exists():
        raise FileExistsError(
            f"Output already exists: {OUTPUT_FILE}\n"
            "Review or rename it before rerunning this script. "
            "The script will not overwrite an earlier result."
        )

    df = pd.read_csv(SOURCE_FILE)

    missing_columns = [
        col for col in DATE_COLUMNS if col not in df.columns
    ]
    if missing_columns:
        raise ValueError(
            f"Missing date columns: {missing_columns}"
        )

    original_row_count = len(df)
    audit_rows = []

    for column in DATE_COLUMNS:
        original, parsed, invalid = parse_dates_strictly(df[column])

        for idx in df.index[invalid]:
            audit_rows.append({
                "Row_Number": int(idx) + 2,
                "Column": column,
                "Original_Value": str(original.loc[idx]),
                "Issue": "Unrecognized date format"
            })

        for idx in df.index[
            original.isna() | original.eq("")
        ]:
            audit_rows.append({
                "Row_Number": int(idx) + 2,
                "Column": column,
                "Original_Value": "",
                "Issue": "Missing date"
            })

        df[column] = parsed

    audit = pd.DataFrame(
        audit_rows,
        columns=[
            "Row_Number",
            "Column",
            "Original_Value",
            "Issue"
        ]
    )

    audit.to_csv(AUDIT_FILE, index=False)

    # Fail safely instead of silently losing or guessing dates.
    if not audit.empty:
        print("\nDate validation failed.")
        print("Audit report:", AUDIT_FILE)
        print(audit.head(20).to_string(index=False))
        raise ValueError(
            "Some dates are missing or unrecognized. "
            "No canonical dataset was written. "
            "Inspect the audit report before proceeding."
        )

    # Write a separate canonical file. The source is untouched.
    # ISO timestamps remove day/month ambiguity.
    df.to_csv(
        OUTPUT_FILE,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S"
    )

    # Validate output by reading it back.
    check = pd.read_csv(OUTPUT_FILE)

    for column in DATE_COLUMNS:
        check[column] = pd.to_datetime(
            check[column],
            format="%Y-%m-%d %H:%M:%S",
            errors="raise"
        )

    if len(check) != original_row_count:
        raise AssertionError("Row count changed during conversion.")

    print("\nDate standardization successful.")
    print("Original file preserved:", SOURCE_FILE)
    print("Canonical file:", OUTPUT_FILE)
    print("Audit report:", AUDIT_FILE)
    print("Rows:", len(check))

    for column in DATE_COLUMNS:
        print(f"\n{column}")
        print("Earliest:", check[column].min())
        print("Latest:", check[column].max())
        print("Unique calendar dates:", check[column].dt.date.nunique())
        print("Missing:", check[column].isna().sum())

    print("\nFirst 5 canonical timestamps:")
    print(check[DATE_COLUMNS].head().to_string(index=False))


if __name__ == "__main__":
    main()
