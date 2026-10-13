
from pathlib import Path

import numpy as np
import pandas as pd


# ==================================================
# 1. PROJECT PATHS AND CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = DATA_DIR / "canonical_run" / "forecasting_artifacts"

TRAIN_FILE = DATA_DIR / "canonical_run" / "engineered_train_cloud_billing_data.csv"
TEST_FILE = DATA_DIR / "canonical_run" / "engineered_test_cloud_billing_data.csv"

MODEL_FILE = ARTIFACTS_DIR / "forecasting_model.npz"

FORECAST_DAYS = 30

DATE_COLUMN = "Usage_Start_Date"
COST_COLUMN = "Total_Cost_INR"


# ==================================================
# 2. LOAD AND VALIDATE DATA
# ==================================================

def load_billing_data(file_path):
    """
    Load and validate a cloud billing CSV file.

    Returns a DataFrame with valid dates and costs.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = [DATE_COLUMN, COST_COLUMN]

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    df[DATE_COLUMN] = pd.to_datetime(
        df[DATE_COLUMN],
        format="%Y-%m-%d %H:%M:%S",
        errors="raise",
    )

    df[COST_COLUMN] = pd.to_numeric(
        df[COST_COLUMN],
        errors="coerce"
    )

    # Remove invalid dates, missing costs, and non-finite values.
    valid_mask = (
        df[DATE_COLUMN].notna()
        & df[COST_COLUMN].notna()
        & np.isfinite(df[COST_COLUMN])
        & (df[COST_COLUMN] >= 0)
    )

    df = df.loc[valid_mask].copy()

    if df.empty:
        raise ValueError(
            f"No valid billing records found in {file_path}"
        )

    return df


# ==================================================
# 3. AGGREGATE DAILY CLOUD EXPENDITURE
# ==================================================

def aggregate_daily_costs(df):
    """
    Aggregate resource-level expenditure by calendar day.

    Important:
    Missing dates are not automatically filled with zero,
    because they may indicate missing billing records.
    """

    daily = (
        df.assign(
            Usage_Date=df[DATE_COLUMN].dt.normalize()
        )
        .groupby("Usage_Date")[COST_COLUMN]
        .sum()
        .sort_index()
    )

    daily.index = pd.DatetimeIndex(
        daily.index,
        name="Usage_Date"
    )

    if daily.empty:
        raise ValueError(
            "Cannot aggregate an empty billing dataset."
        )

    return daily.astype(float)


# ==================================================
# 4. FEATURE ENGINEERING
# ==================================================

def create_time_features(dates, reference_date):
    """
    Create features for daily cost forecasting.

    Features:
    1. Elapsed days (trend)
    2. Day-of-week indicators

    Monday is the reference weekday.
    """

    dates = pd.DatetimeIndex(
        pd.to_datetime(dates)
    )

    if dates.isna().any():
        raise ValueError(
            "Dates contain invalid values."
        )

    reference_date = pd.Timestamp(
        reference_date
    ).normalize()

    elapsed_days = (
        (dates - reference_date).total_seconds()
        / 86400
    )

    features = pd.DataFrame(
        {
            "trend": np.asarray(
                elapsed_days,
                dtype=float
            )
        },
        index=dates
    )

    for day in range(1, 7):
        features[f"weekday_{day}"] = (
            dates.dayofweek == day
        ).astype(int)

    return features


# ==================================================
# 5. PREPARE MODEL INPUTS
# ==================================================

def prepare_model_matrix(features):
    """
    Convert features into a numerical matrix and add
    an intercept column for linear regression.
    """

    matrix = features.to_numpy(dtype=float)

    if not np.isfinite(matrix).all():
        raise ValueError(
            "Model features contain non-finite values."
        )

    return np.column_stack(
        [
            np.ones(len(matrix)),
            matrix
        ]
    )


# ==================================================
# 6. GENERATE PREDICTIONS
# ==================================================

def predict_daily_costs(features, coefficients):
    """
    Generate predictions using fitted linear-model
    coefficients saved during training.
    """

    coefficients = np.asarray(
        coefficients,
        dtype=float
    )

    matrix = prepare_model_matrix(features)

    if matrix.shape[1] != len(coefficients):
        raise ValueError(
            "Feature count does not match model coefficients."
        )

    predictions = matrix @ coefficients

    # Cloud expenditure cannot be negative.
    return np.maximum(predictions, 0)


# ==================================================
# 7. EVALUATION METRICS
# ==================================================

def calculate_metrics(actual, predicted):
    """
    Calculate MAE, RMSE, and WAPE.

    WAPE is reported as a percentage. If total actual
    expenditure is zero, WAPE is undefined and returned
    as NaN.
    """

    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if actual.size == 0 or actual.shape != predicted.shape:
        raise ValueError(
            "Actual and predicted values must be non-empty "
            "arrays with matching shapes."
        )

    if (
        not np.isfinite(actual).all()
        or not np.isfinite(predicted).all()
    ):
        raise ValueError(
            "Actual or predicted costs contain non-finite values."
        )

    if (actual < 0).any():
        raise ValueError(
            "Actual cloud costs cannot be negative."
        )

    errors = actual - predicted

    mae = np.mean(np.abs(errors))
    rmse = np.sqrt(np.mean(errors ** 2))

    total_actual = np.sum(np.abs(actual))

    wape = (
        np.sum(np.abs(errors)) / total_actual * 100
        if total_actual > 0
        else np.nan
    )

    return {
        "MAE_INR": float(mae),
        "RMSE_INR": float(rmse),
        "WAPE_Percent": float(wape)
    }


# ==================================================
# 8. SAVE MODEL PARAMETERS
# ==================================================

def save_model(coefficients, reference_date):
    """
    Save fitted coefficients and the training reference date.

    The evaluation and prediction scripts can load these
    parameters without retraining the model.
    """

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.savez(
        MODEL_FILE,
        coefficients=np.asarray(
            coefficients,
            dtype=float
        ),
        reference_date=str(
            pd.Timestamp(reference_date).date()
        ),
        feature_names=np.array(
            [
                "intercept",
                "trend",
                "weekday_1",
                "weekday_2",
                "weekday_3",
                "weekday_4",
                "weekday_5",
                "weekday_6"
            ]
        )
    )

    print(f"Model saved: {MODEL_FILE}")


# ==================================================
# 9. LOAD SAVED MODEL
# ==================================================

def load_model():
    """
    Load saved coefficients and training reference date.
    """

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}. "
            "Run train_forecasting.py first."
        )

    with np.load(
        MODEL_FILE,
        allow_pickle=False
    ) as model:

        coefficients = model["coefficients"].copy()

        reference_date = pd.Timestamp(
            str(model["reference_date"].item())
        )

        feature_names = model["feature_names"].tolist()

    if len(coefficients) != 8:
        raise ValueError(
            "Unexpected coefficient count in saved model."
        )

    if not np.isfinite(coefficients).all():
        raise ValueError(
            "Saved model contains non-finite coefficients."
        )

    return coefficients, reference_date, feature_names
