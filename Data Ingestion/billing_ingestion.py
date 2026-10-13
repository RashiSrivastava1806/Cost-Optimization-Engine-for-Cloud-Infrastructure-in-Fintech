
from __future__ import annotations

import io
import re
from typing import Any

import numpy as np
import pandas as pd


# ============================================================
# 1. CANONICAL SCHEMA
# ============================================================

CANONICAL_COLUMNS = [
    "Billing_Date",
    "Resource_ID",
    "Service_Name",
    "Region",
    "Cost",
    "Currency",
    "CPU_Utilization",
    "Memory_Utilization",
    "Usage_Quantity",
    "Usage_Unit",
    "Billing_Record_ID",
]

REQUIRED_COLUMNS = ["Cost"]

COLUMN_ALIASES = {
    "Billing_Date": [
        "date", "billing date", "usage date", "invoice date",
        "usage start date", "usage_start_date", "start date",
        "usage start time", "timestamp", "billing period start",
        "line item usage start date",
    ],
    "Resource_ID": [
        "resource", "resource id", "resource_id", "resource name",
        "instance id", "instance_id", "instance name",
        "instance", "asset id", "virtual machine id",
        "resource identifier",
    ],
    "Service_Name": [
        "service", "service name", "service_name", "product",
        "product name", "cloud service", "service type",
        "line item product code", "service family",
    ],
    "Region": [
        "region", "region zone", "region_zone", "location",
        "availability zone", "availability_zone", "datacenter",
        "data center",
    ],
    "Cost": [
        "cost", "total cost", "total_cost", "cost inr",
        "total cost inr", "total_cost_inr", "billed amount",
        "billing amount", "amount", "amount billed",
        "net cost", "net cost amount", "unrounded cost",
        "rounded cost", "effective cost", "line item cost",
        "line item unblended cost", "cost amount",
        "pretax cost", "pre tax cost", "invoice amount",
        "extended cost", "charge amount",
    ],
    "Currency": [
        "currency", "currency code", "billing currency",
        "pricing currency", "invoice currency",
    ],
    "CPU_Utilization": [
        "cpu", "cpu utilization", "cpu_utilization",
        "cpu usage", "cpu usage percent", "cpu utilization percent",
        "cpu percent", "cpu percentage", "processor utilization",
    ],
    "Memory_Utilization": [
        "memory", "memory utilization", "memory_utilization",
        "memory usage", "memory usage percent",
        "memory utilization percent", "memory percent",
        "memory percentage", "ram utilization",
    ],
    "Usage_Quantity": [
        "usage quantity", "usage_quantity", "usage amount",
        "quantity", "usage", "consumed quantity",
        "consumed amount", "usage value",
    ],
    "Usage_Unit": [
        "usage unit", "usage_unit", "unit", "usage type",
        "unit of measure", "uom", "measurement unit",
    ],
    "Billing_Record_ID": [
        "billing record id", "record id", "line item id",
        "line item identifier", "transaction id",
        "charge id", "billing id",
    ],
}

# Some cloud exports use these fields to describe a charge.
# We do NOT add multiple charge columns together automatically.
COST_COLUMN_HINTS = [
    "unblended cost",
    "amortized cost",
    "effective cost",
    "net cost",
    "billed cost",
    "total cost",
    "cost",
    "amount",
]


# ============================================================
# 2. NORMALIZE COLUMN NAMES
# ============================================================

def normalize_column_name(name: Any) -> str:
    """Convert column labels into a comparable form."""
    text = str(name).strip().lower()
    text = text.replace("%", " percent ")
    text = text.replace("_", " ")
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def suggest_column_mapping(
    columns: list[str],
) -> dict[str, str | None]:
    """
    Suggest source columns for canonical fields.

    Matching is conservative: aliases are matched exactly after
    normalization. Ambiguous matches are left unmapped.
    """
    normalized_to_original: dict[str, list[str]] = {}

    for column in columns:
        normalized = normalize_column_name(column)
        normalized_to_original.setdefault(normalized, []).append(column)

    mapping: dict[str, str | None] = {}

    for canonical, aliases in COLUMN_ALIASES.items():
        accepted = {
            normalize_column_name(canonical),
            *(normalize_column_name(alias) for alias in aliases),
        }

        matches = []
        for normalized, originals in normalized_to_original.items():
            if normalized in accepted:
                matches.extend(originals)

        mapping[canonical] = matches[0] if len(matches) == 1 else None

    return mapping


def suggest_cost_columns(columns: list[str]) -> list[str]:
    """
    Return likely cost fields for manual selection.

    This is a suggestion list only. It does not choose a cost field
    when multiple plausible fields exist.
    """
    result = []

    for column in columns:
        name = normalize_column_name(column)

        if (
            name == "cost"
            or name.endswith(" cost")
            or name.endswith(" amount")
            or name in {"amount", "billed amount", "invoice amount"}
        ):
            result.append(column)

    return result


# ============================================================
# 3. READ CSV / EXCEL FILES
# ============================================================

def read_billing_file(
    uploaded_file: Any,
    filename: str | None = None,
) -> pd.DataFrame:
    """
    Read a CSV or Excel file from a path, bytes, or Streamlit upload.

    For CSV files, attempts common delimiters and encodings.
    Excel workbooks use the first worksheet by default.
    """
    name = filename or getattr(uploaded_file, "name", "")
    name = str(name).lower()

    if isinstance(uploaded_file, (bytes, bytearray)):
        raw = bytes(uploaded_file)
    elif hasattr(uploaded_file, "getvalue"):
        raw = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        raw = uploaded_file.read()
    else:
        # Supports local file paths too.
        path = str(uploaded_file)

        if path.lower().endswith((".xlsx", ".xls")):
            return pd.read_excel(path)

        if path.lower().endswith((".csv", ".txt", ".tsv")):
            return _read_csv_bytes(open(path, "rb").read())

        raise ValueError(
            "Unsupported input. Upload CSV or Excel, or provide a file path."
        )

    if not raw:
        raise ValueError("The uploaded file is empty.")

    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(raw))

    if name.endswith((".csv", ".txt", ".tsv")):
        return _read_csv_bytes(raw)

    # Try Excel first, then CSV if the extension is missing or unusual.
    try:
        return pd.read_excel(io.BytesIO(raw))
    except Exception:
        return _read_csv_bytes(raw)


def _read_csv_bytes(raw: bytes) -> pd.DataFrame:
    """Read CSV, TSV, or semicolon-delimited data."""
    errors = []

    for encoding in ("utf-8-sig", "utf-8", "cp1252", "latin1"):
        try:
            frame = pd.read_csv(
                io.BytesIO(raw),
                sep=None,
                engine="python",
                encoding=encoding,
                on_bad_lines="error",
            )

            if frame.shape[1] == 1:
                # A single parsed column may mean delimiter detection failed.
                for delimiter in ("\t", ";", "|", ","):
                    candidate = pd.read_csv(
                        io.BytesIO(raw),
                        sep=delimiter,
                        encoding=encoding,
                        on_bad_lines="error",
                    )
                    if candidate.shape[1] > 1:
                        frame = candidate
                        break

            if frame.empty and len(frame.columns) == 0:
                raise ValueError("No columns found.")

            frame.columns = [str(c).strip() for c in frame.columns]
            return frame

        except Exception as exc:
            errors.append(f"{encoding}: {exc}")

    raise ValueError(
        "Could not read the file as CSV. Check its delimiter and encoding. "
        + " | ".join(errors[-2:])
    )


# ============================================================
# 4. STANDARDIZE THE BILLING DATASET
# ============================================================

def standardize_billing_data(
    source: pd.DataFrame,
    column_mapping: dict[str, str | None],
    default_currency: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    Convert client columns to our canonical schema.

    Parameters
    ----------
    source:
        Original client billing DataFrame.
    column_mapping:
        Mapping from canonical field name to source column name.
        Example: {"Cost": "BilledAmount", "Billing_Date": "UsageDate"}.
    default_currency:
        Optional currency explicitly supplied by the client. It is
        not inferred from the size or appearance of cost values.

    Returns
    -------
    canonical:
        Standardized billing DataFrame.
    report:
        Data-quality and validation summary.
    """
    if source is None or source.empty:
        raise ValueError("The billing dataset contains no data rows.")

    if len(source.columns) == 0:
        raise ValueError("The billing dataset has no columns.")

    source = source.copy()
    source.columns = [str(c).strip() for c in source.columns]

    if source.columns.duplicated().any():
        duplicates = source.columns[source.columns.duplicated()].tolist()
        raise ValueError(
            f"Duplicate column names detected: {duplicates}. "
            "Rename duplicate columns before importing."
        )

    missing_required = [
        field
        for field in REQUIRED_COLUMNS
        if not column_mapping.get(field)
        or column_mapping.get(field) not in source.columns
    ]

    if missing_required:
        raise ValueError(
            "Map the following required field(s) before continuing: "
            + ", ".join(missing_required)
        )

    canonical = pd.DataFrame(index=source.index)

    # Map fields, leaving absent optional fields genuinely missing.
    for field in CANONICAL_COLUMNS:
        source_column = column_mapping.get(field)

        if source_column and source_column in source.columns:
            canonical[field] = source[source_column]
        else:
            canonical[field] = pd.NA

    # Cost parsing: support commas, currency symbols and parentheses.
    canonical["Cost"] = _parse_numeric(canonical["Cost"])

    # Never interpret missing or invalid costs as zero.
    invalid_cost_mask = canonical["Cost"].isna()
    negative_cost_mask = canonical["Cost"].lt(0).fillna(False)

    # Currency is explicit if mapped, or provided as a user-confirmed default.
    currency_values = canonical["Currency"].astype("string").str.strip().str.upper()
    currency_values = currency_values.mask(currency_values.eq(""))

    if default_currency:
        currency_values = currency_values.fillna(default_currency.strip().upper())

    canonical["Currency"] = currency_values

    # Dates: unparseable dates stay missing and are reported.
    original_dates = canonical["Billing_Date"].copy()
    canonical["Billing_Date"] = pd.to_datetime(
        canonical["Billing_Date"],
        errors="coerce",
        utc=True,
    ).dt.tz_convert(None)

    # Utilization fields: numeric values only; no imputation.
    for field in ("CPU_Utilization", "Memory_Utilization"):
        canonical[field] = _parse_numeric(canonical[field])

    # Identify percent-like values entered as fractions.
    # We do NOT automatically multiply fractional values by 100,
    # because the source unit may be ambiguous.
    utilization_warnings = {}

    for field in ("CPU_Utilization", "Memory_Utilization"):
        values = canonical[field].dropna()
        out_of_range = values[(values < 0) | (values > 100)]

        utilization_warnings[field] = {
            "out_of_range_count": int(len(out_of_range)),
            "possible_fractional_values": int(
                ((values >= 0) & (values <= 1)).sum()
            ),
        }

    canonical["Usage_Quantity"] = _parse_numeric(
        canonical["Usage_Quantity"]
    )

    # Remove leading/trailing whitespace from descriptive fields.
    for field in (
        "Resource_ID",
        "Service_Name",
        "Region",
        "Usage_Unit",
        "Billing_Record_ID",
    ):
        canonical[field] = (
            canonical[field]
            .astype("string")
            .str.strip()
            .replace("", pd.NA)
        )

    # Detect exact duplicate source rows without silently dropping them.
    duplicate_mask = source.duplicated(keep="first")

    # Duplicate billing IDs are a separate warning; multiple rows per
    # resource are normal and must not be deduplicated automatically.
    billing_id = canonical["Billing_Record_ID"].dropna()
    duplicate_billing_ids = int(billing_id.duplicated(keep=False).sum())

    missing_by_field = {
        field: int(canonical[field].isna().sum())
        for field in CANONICAL_COLUMNS
    }

    report: dict[str, Any] = {
        "row_count": int(len(canonical)),
        "source_column_count": int(len(source.columns)),
        "exact_duplicate_rows": int(duplicate_mask.sum()),
        "invalid_or_missing_cost_rows": int(invalid_cost_mask.sum()),
        "negative_cost_rows": int(negative_cost_mask.sum()),
        "missing_values_by_field": missing_by_field,
        "missing_date_rows": int(canonical["Billing_Date"].isna().sum()),
        "unparseable_date_rows": int(
            original_dates.notna().sum()
            - canonical["Billing_Date"].notna().sum()
        ),
        "unknown_currency_rows": int(canonical["Currency"].isna().sum()),
        "distinct_currencies": sorted(
            canonical["Currency"].dropna().astype(str).unique().tolist()
        ),
        "duplicate_billing_id_rows": duplicate_billing_ids,
        "utilization_warnings": utilization_warnings,
        "warnings": [],
    }

    if report["invalid_or_missing_cost_rows"]:
        report["warnings"].append(
            "Some costs are missing or could not be parsed. "
            "Review them before calculating totals."
        )

    if report["unknown_currency_rows"]:
        report["warnings"].append(
            "Some rows have unknown currency. Do not combine currencies "
            "into one total until they are identified or converted."
        )

    if report["missing_date_rows"]:
        report["warnings"].append(
            "Some dates are missing or invalid. Date-based analysis may "
            "exclude these records."
        )

    if report["exact_duplicate_rows"]:
        report["warnings"].append(
            "Exact duplicate source rows were detected but not removed."
        )

    if report["negative_cost_rows"]:
        report["warnings"].append(
            "Negative costs were detected. These may be credits or refunds; "
            "review them rather than automatically deleting them."
        )

    for field, warning in utilization_warnings.items():
        if warning["out_of_range_count"]:
            report["warnings"].append(
                f"{field} has values outside 0–100. Check the source units."
            )

        if warning["possible_fractional_values"]:
            report["warnings"].append(
                f"{field} contains values between 0 and 1. Confirm whether "
                "these represent percentages or fractions before analysis."
            )

    return canonical.reset_index(drop=True), report


# ============================================================
# 5. NUMERIC PARSING
# ============================================================

def _parse_numeric(series: pd.Series) -> pd.Series:
    """
    Parse numbers with commas, currency symbols and accounting negatives.

    Examples:
      ₹1,250.50 -> 1250.50
      $250.00   -> 250.00
      (75.00)   -> -75.00

    This does not perform currency conversion.
    """
    text = series.astype("string").str.strip()

    is_parenthesized = text.str.match(r"^\(.*\)$", na=False)

    text = text.str.replace(",", "", regex=False)
    text = text.str.replace(r"^\((.*)\)$", r"\1", regex=True)
    text = text.str.replace(
        r"[^0-9eE.+\-]",
        "",
        regex=True,
    )

    values = pd.to_numeric(text, errors="coerce")
    values = values.mask(~np.isfinite(values))
    values.loc[is_parenthesized & values.notna()] = (
        -values.loc[is_parenthesized & values.notna()].abs()
    )

    return values


# ============================================================
# 6. DATA QUALITY SUMMARY
# ============================================================

def create_data_quality_table(report: dict[str, Any]) -> pd.DataFrame:
    """Create a dashboard-friendly table from the validation report."""
    missing = report.get("missing_values_by_field", {})

    rows = [
        {"Check": "Total rows", "Result": report.get("row_count", 0)},
        {
            "Check": "Invalid or missing cost rows",
            "Result": report.get("invalid_or_missing_cost_rows", 0),
        },
        {
            "Check": "Missing or invalid date rows",
            "Result": report.get("missing_date_rows", 0),
        },
        {
            "Check": "Rows with unknown currency",
            "Result": report.get("unknown_currency_rows", 0),
        },
        {
            "Check": "Exact duplicate rows",
            "Result": report.get("exact_duplicate_rows", 0),
        },
        {
            "Check": "Duplicate billing ID rows",
            "Result": report.get("duplicate_billing_id_rows", 0),
        },
    ]

    for field, count in missing.items():
        rows.append(
            {
                "Check": f"Missing: {field}",
                "Result": count,
            }
        )

    return pd.DataFrame(rows)
