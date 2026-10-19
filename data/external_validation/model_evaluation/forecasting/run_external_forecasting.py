from pathlib import Path
import json
import zipfile
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=FutureWarning)

ROOT = Path.cwd()
EXTERNAL = ROOT / "data" / "external_validation"
OUT = EXTERNAL / "model_evaluation" / "forecasting"
OUT.mkdir(parents=True, exist_ok=True)

ZIP_PATH = EXTERNAL / "raw" / "dataset-examples.zip"
CSV_NAME = "EA-Cost-FOCUS_1.0.csv"

def load_dataset():
    # Support either an already-extracted CSV or the original archive.
    candidates = [
        EXTERNAL / CSV_NAME,
        EXTERNAL / "raw" / CSV_NAME,
        EXTERNAL / "EA-Cost-FOCUS_1.0" / CSV_NAME,
    ]
    for path in candidates:
        if path.exists():
            print(f"Reading dataset: {path}")
            return pd.read_csv(path, low_memory=False)

    if ZIP_PATH.exists():
        with zipfile.ZipFile(ZIP_PATH) as archive:
            matches = [
                name for name in archive.namelist()
                if name.endswith(CSV_NAME)
            ]
            if not matches:
                raise FileNotFoundError(
                    f"{CSV_NAME} was not found inside {ZIP_PATH}"
                )
            member = matches[0]
            print(f"Reading {member} from {ZIP_PATH}")
            with archive.open(member) as stream:
                return pd.read_csv(stream, low_memory=False)

    raise FileNotFoundError(
        "Cannot locate the FOCUS CSV or dataset-examples.zip. "
        "Expected the archive at data/external_validation/raw/dataset-examples.zip."
    )

def metric_values(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    errors = actual - predicted

    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors ** 2)))
    denominator = float(np.sum(np.abs(actual)))
    wape = (
        float(np.sum(np.abs(errors)) / denominator * 100)
        if denominator > 0 else None
    )
    return {"MAE": mae, "RMSE": rmse, "WAPE_percent": wape}

def main():
    df = load_dataset()

    required = {"ChargePeriodStart", "BilledCost"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Required columns missing: {sorted(missing)}")

    # Parse dates in UTC and use BilledCost consistently as the target.
    df["ChargePeriodStart"] = pd.to_datetime(
        df["ChargePeriodStart"], errors="coerce", utc=True
    )
    df["BilledCost"] = pd.to_numeric(df["BilledCost"], errors="coerce")
    df = df.dropna(subset=["ChargePeriodStart", "BilledCost"]).copy()

    if df.empty:
        raise ValueError("No valid dated billing records were found.")

    # Daily net billed cost: negative entries are retained rather than
    # silently removed, because their meaning is not independently verified.
    df["Date"] = df["ChargePeriodStart"].dt.floor("D")
    daily = (
        df.groupby("Date", as_index=False)["BilledCost"]
        .sum()
        .rename(columns={"BilledCost": "Actual_BilledCost"})
        .sort_values("Date")
        .reset_index(drop=True)
    )

    # Include missing calendar dates as zero-cost days. This makes the daily
    # series explicit; the report records the resulting series length.
    full_dates = pd.date_range(
        daily["Date"].min(), daily["Date"].max(), freq="D", tz="UTC"
    )
    daily = (
        daily.set_index("Date")
        .reindex(full_dates, fill_value=0)
        .rename_axis("Date")
        .reset_index()
    )
    daily["Actual_BilledCost"] = pd.to_numeric(
        daily["Actual_BilledCost"], errors="coerce"
    ).fillna(0.0)

    n = len(daily)
    if n < 8:
        raise ValueError(
            f"Only {n} daily observations are available. "
            "At least 8 are required for this chronological evaluation."
        )

    # Keep the final ~25% for testing; never shuffle a time series.
    test_n = max(3, int(np.ceil(n * 0.25)))
    test_n = min(test_n, n - 5)
    split = n - test_n

    train = daily.iloc[:split].copy()
    test = daily.iloc[split:].copy()
    y_train = train["Actual_BilledCost"].to_numpy(dtype=float)
    y_test = test["Actual_BilledCost"].to_numpy(dtype=float)

    # Candidate 1: linear time trend estimated using training data only.
    x_train = np.arange(split, dtype=float)
    x_test = np.arange(split, n, dtype=float)
    slope, intercept = np.polyfit(x_train, y_train, deg=1)
    trend_pred = intercept + slope * x_test

    # Candidate 2: historical-average baseline using training data only.
    average_pred = np.full(len(test), np.mean(y_train), dtype=float)

    # Candidate 3: naive baseline, carrying the latest training observation
    # forward across the test horizon.
    naive_pred = np.full(len(test), y_train[-1], dtype=float)

    predictions = {
        "Linear Trend": trend_pred,
        "Historical Average": average_pred,
        "Naive Last Value": naive_pred,
    }

    metrics = []
    result = test[["Date", "Actual_BilledCost"]].copy()

    for name, pred in predictions.items():
        result[name.replace(" ", "_") + "_Prediction"] = pred
        values = metric_values(y_test, pred)
        metrics.append({
            "Model": name,
            **values,
            "Train_Observations": int(len(train)),
            "Test_Observations": int(len(test)),
        })

    metrics_df = pd.DataFrame(metrics).sort_values("MAE")
    metrics_df.to_csv(OUT / "model_comparison.csv", index=False)
    result.to_csv(OUT / "forecast_results.csv", index=False)

    best_name = str(metrics_df.iloc[0]["Model"])
    best_prediction = predictions[best_name]
    best_metrics = metric_values(y_test, best_prediction)

    pd.DataFrame([{
        "Model": best_name,
        **best_metrics,
        "Train_Observations": int(len(train)),
        "Test_Observations": int(len(test)),
        "First_Test_Date": test["Date"].iloc[0].isoformat(),
        "Last_Test_Date": test["Date"].iloc[-1].isoformat(),
    }]).to_csv(OUT / "forecast_metrics.csv", index=False)

    # Figure 1: holdout actuals vs predictions.
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        train["Date"], train["Actual_BilledCost"],
        label="Training actuals", alpha=0.65
    )
    ax.plot(
        test["Date"], y_test,
        marker="o", linewidth=2, label="Test actuals"
    )
    ax.plot(
        test["Date"], best_prediction,
        marker="x", linestyle="--", linewidth=2,
        label=f"Best holdout model: {best_name}"
    )
    ax.axvline(test["Date"].iloc[0], linestyle=":", label="Test period starts")
    ax.set_title("External FOCUS Dataset: Daily Billed Cost Forecast Evaluation")
    ax.set_xlabel("Charge period start date (UTC)")
    ax.set_ylabel("Net daily BilledCost (USD)")
    ax.legend()
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "actual_vs_predicted.png", dpi=180)
    plt.close(fig)

    # Figure 2: compare MAE across candidate models.
    fig, ax = plt.subplots(figsize=(9, 5))
    ordered = metrics_df.sort_values("MAE", ascending=True)
    ax.bar(ordered["Model"], ordered["MAE"])
    ax.set_title("Holdout Forecasting Error Comparison")
    ax.set_ylabel("MAE (USD; lower is better)")
    ax.set_xlabel("Model")
    ax.tick_params(axis="x", rotation=15)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "model_mae_comparison.png", dpi=180)
    plt.close(fig)

    summary = {
        "dataset": "Microsoft FinOps Toolkit demo FOCUS dataset",
        "source_file": CSV_NAME,
        "target": "Daily net BilledCost in USD",
        "aggregation": "Sum of BilledCost by UTC ChargePeriodStart calendar day",
        "negative_costs": "Retained in net daily sums; not assumed to be refunds",
        "total_input_records_after_valid_date_cost_filter": int(len(df)),
        "daily_observations": int(n),
        "train_observations": int(len(train)),
        "test_observations": int(len(test)),
        "train_start": train["Date"].iloc[0].isoformat(),
        "train_end": train["Date"].iloc[-1].isoformat(),
        "test_start": test["Date"].iloc[0].isoformat(),
        "test_end": test["Date"].iloc[-1].isoformat(),
        "best_model_by_MAE": best_name,
        "best_model_metrics": best_metrics,
        "models": metrics,
        "limitations": [
            "This is a demo dataset, not production/customer billing evidence.",
            "The short date range and small holdout make performance estimates uncertain.",
            "A model winning this holdout does not establish general forecasting reliability.",
            "The linear trend and baselines are standalone comparison models; "
            "this run does not claim to reproduce the project's canonical model.",
            "Forecasting billed cost does not establish SLA compliance or actual savings."
        ]
    }

    with open(OUT / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # A readable inventory so all outputs are easy to find later.
    inventory = pd.DataFrame([
        ["forecast_metrics.csv", "Best model's holdout metrics"],
        ["model_comparison.csv", "MAE, RMSE and WAPE for all baselines"],
        ["forecast_results.csv", "Daily actuals and test predictions"],
        ["actual_vs_predicted.png", "Time-series holdout comparison graph"],
        ["model_mae_comparison.png", "Model error comparison graph"],
        ["run_summary.json", "Dataset details, split, metrics and caveats"],
    ], columns=["File", "Description"])
    inventory.to_csv(OUT / "output_index.csv", index=False)

    print("\n========== FORECASTING EVALUATION COMPLETE ==========")
    print(f"Output folder: {OUT}")
    print(f"Daily observations: {n}")
    print(f"Training observations: {len(train)}")
    print(f"Test observations: {len(test)}")
    print(f"Best model by MAE: {best_name}")
    print("\nModel comparison:")
    print(metrics_df.to_string(index=False))
    print("\nGenerated files:")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name}")

if __name__ == "__main__":
    main()
