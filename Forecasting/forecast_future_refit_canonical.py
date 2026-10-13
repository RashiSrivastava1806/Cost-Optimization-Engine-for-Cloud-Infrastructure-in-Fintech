
import numpy as np
import pandas as pd

from forecasting_utils_canonical import (
    TRAIN_FILE,
    TEST_FILE,
    MODEL_FILE,
    ARTIFACTS_DIR,
    FORECAST_DAYS,
    load_billing_data,
    aggregate_daily_costs,
    create_time_features,
    predict_daily_costs,
)


def main():
    print("=" * 60)
    print("CLOUD COST OPTIMIZATION - FUTURE FORECAST")
    print("=" * 60)

    # Step 1: Load all observed canonical data for the final refit.
    train_df = load_billing_data(TRAIN_FILE)
    test_df = load_billing_data(TEST_FILE)

    combined_df = pd.concat([train_df, test_df], ignore_index=True)
    daily_costs = aggregate_daily_costs(combined_df).sort_index()

    if daily_costs.empty or daily_costs.isna().any():
        raise ValueError("Daily billing data is empty or contains missing costs.")

    if len(daily_costs) < 7:
        raise ValueError("At least 7 observed dates are required.")

    reference_date = daily_costs.index.min()
    last_observed_date = daily_costs.index.max()

    # Fit the same linear trend + weekday model on all observed dates.
    historical_features = create_time_features(
        daily_costs.index,
        reference_date,
    )
    feature_columns = historical_features.columns.tolist()

    design_matrix = np.column_stack([
        np.ones(len(historical_features)),
        historical_features.to_numpy(dtype=float),
    ])
    coefficients, _, rank, _ = np.linalg.lstsq(
        design_matrix,
        daily_costs.to_numpy(dtype=float),
        rcond=None,
    )

    if rank < design_matrix.shape[1]:
        raise ValueError("Full-data forecasting design matrix is rank deficient.")

    model_name = "Linear Trend with Weekday Effects (refit on all observed data)"
    print(f"\nRefitted model: {model_name}")
    print(f"Observed dates used: {len(daily_costs)}")
    print(f"Training period: {reference_date.date()} to {last_observed_date.date()}")

    # Step 4: Create the next 30 calendar dates
    future_dates = pd.date_range(
        start=last_observed_date + pd.Timedelta(days=1),
        periods=FORECAST_DAYS,
        freq="D",
    )

    # Step 5: Create the same features used during training
    future_features = create_time_features(
        future_dates,
        reference_date,
    )

    if future_features.columns.tolist() != feature_columns:
        raise ValueError(
            "Future forecast features do not match the saved model."
        )

    # Step 6: Predict future daily expenditure
    future_matrix = np.column_stack([
        np.ones(len(future_features)),
        future_features.to_numpy(dtype=float),
    ])
    raw_predicted_costs = future_matrix @ coefficients
    negative_prediction_count = int((raw_predicted_costs < 0).sum())
    predicted_costs = np.maximum(raw_predicted_costs, 0.0)

    # Step 7: Create and save daily forecast
    forecast_df = pd.DataFrame({
        "Forecast_Date": future_dates,
        "Predicted_Cost_INR": predicted_costs,
        "Raw_Predicted_Cost_INR": raw_predicted_costs,
    })

    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    forecast_file = ARTIFACTS_DIR / "future_30_day_forecast.csv"
    forecast_df.to_csv(forecast_file, index=False)

    # Step 8: Calculate forecast summary
    total_forecast_cost = float(predicted_costs.sum())
    average_daily_cost = float(predicted_costs.mean())
    minimum_daily_cost = float(predicted_costs.min())
    maximum_daily_cost = float(predicted_costs.max())

    summary_df = pd.DataFrame([{
        "Forecast_Start_Date": future_dates.min().date(),
        "Forecast_End_Date": future_dates.max().date(),
        "Forecast_Days": FORECAST_DAYS,
        "Predicted_Total_Cost_INR": total_forecast_cost,
        "Average_Daily_Cost_INR": average_daily_cost,
        "Minimum_Daily_Cost_INR": minimum_daily_cost,
        "Maximum_Daily_Cost_INR": maximum_daily_cost,
        "Model": model_name,
        "Last_Observed_Date": last_observed_date.date(),
        "Negative_Raw_Predictions_Clipped": negative_prediction_count,
    }])

    summary_file = ARTIFACTS_DIR / "forecast_summary.csv"
    summary_df.to_csv(summary_file, index=False)

    # Step 9: Display results
    print("\n30-DAY FORECAST SUMMARY")
    print("-" * 40)
    print(f"Last observed billing date: {last_observed_date.date()}")
    print(f"Forecast starts: {future_dates.min().date()}")
    print(f"Forecast ends:   {future_dates.max().date()}")
    print(f"Forecast days: {FORECAST_DAYS}")
    print(f"Predicted total expenditure: ₹{total_forecast_cost:,.2f}")
    print(f"Average daily expenditure:   ₹{average_daily_cost:,.2f}")
    print(f"Minimum daily prediction:    ₹{minimum_daily_cost:,.2f}")
    print(f"Maximum daily prediction:    ₹{maximum_daily_cost:,.2f}")
    print(f"Negative raw predictions clipped to zero: {negative_prediction_count}")

    print(f"\nDaily forecast saved: {forecast_file}")
    print(f"Forecast summary saved: {summary_file}")

    print(
        "\nLIMITATION: This is a statistical extrapolation from "
        "historical billing data. It does not account for future "
        "resource changes, pricing changes, promotions, or workload "
        "growth. Forecast quality should be judged using the held-out "
        "test metrics before relying on the predictions."
    )


if __name__ == "__main__":
    main()
