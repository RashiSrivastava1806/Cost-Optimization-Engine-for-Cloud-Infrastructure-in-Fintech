
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

    # Step 1: Verify the trained model exists
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_FILE}\n"
            "Run train_forecasting.py first."
        )

    # Step 2: Load model coefficients and metadata
    with np.load(MODEL_FILE, allow_pickle=False) as model:
        coefficients = model["coefficients"]
        reference_date = pd.Timestamp(
            str(model["reference_date"].item())
        )
        feature_columns = model["feature_columns"].tolist()
        model_name = str(model["model_name"].item())

    print(f"\nLoaded model: {model_name}")

    # Step 3: Find the last observed date across available data
    train_df = load_billing_data(TRAIN_FILE)
    test_df = load_billing_data(TEST_FILE)

    train_daily = aggregate_daily_costs(train_df)
    test_daily = aggregate_daily_costs(test_df)

    last_observed_date = max(
        train_daily.index.max(),
        test_daily.index.max(),
    )

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
    predicted_costs = predict_daily_costs(
        future_features,
        coefficients,
    )

    # Step 7: Create and save daily forecast
    forecast_df = pd.DataFrame({
        "Forecast_Date": future_dates,
        "Predicted_Cost_INR": predicted_costs,
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
