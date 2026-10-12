
import pandas as pd

from forecasting_utils import (
    TEST_FILE,
    MODEL_FILE,
    ARTIFACTS_DIR,
    load_billing_data,
    aggregate_daily_costs,
    create_time_features,
    predict_daily_costs,
    calculate_metrics,
)


def main():
    print("=" * 60)
    print("CLOUD COST OPTIMIZATION - MODEL EVALUATION")
    print("=" * 60)

    # Step 1: Verify that the trained model exists
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_FILE}\n"
            "Run train_forecasting.py first."
        )

    # Step 2: Load the trained model
    model = pd.io.common.get_handle if False else None

    import numpy as np

    with np.load(MODEL_FILE, allow_pickle=False) as saved_model:
        coefficients = saved_model["coefficients"]
        reference_date = pd.Timestamp(
            str(saved_model["reference_date"].item())
        )
        saved_features = saved_model["feature_columns"].tolist()
        model_name = str(saved_model["model_name"].item())

    print(f"\nLoaded model: {model_name}")

    # Step 3: Load the independent test dataset
    df = load_billing_data(TEST_FILE)
    print(f"Test billing records loaded: {len(df)}")

    # Step 4: Aggregate costs by observed calendar date
    daily_costs = aggregate_daily_costs(df)

    print(f"Observed test dates: {len(daily_costs)}")
    print(f"Test start date: {daily_costs.index.min().date()}")
    print(f"Test end date:   {daily_costs.index.max().date()}")

    # Step 5: Confirm the test period occurs after training
    if daily_costs.index.min() <= reference_date:
        raise ValueError(
            "Test dates overlap the model reference date. "
            "Check the chronological train/test split."
        )

    # Step 6: Recreate the same time features
    features = create_time_features(
        daily_costs.index,
        reference_date,
    )

    if features.columns.tolist() != saved_features:
        raise ValueError(
            "Evaluation features do not match the saved model."
        )

    # Step 7: Predict test-period daily costs
    predictions = predict_daily_costs(
        features,
        coefficients,
    )

    actual = daily_costs.to_numpy(dtype=float)

    # Step 8: Calculate evaluation metrics
    metrics = calculate_metrics(actual, predictions)

    # Step 9: Save daily test predictions
    results = pd.DataFrame({
        "Date": daily_costs.index,
        "Actual_Cost_INR": actual,
        "Predicted_Cost_INR": predictions,
        "Absolute_Error_INR": abs(actual - predictions),
    })

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    predictions_file = (
        ARTIFACTS_DIR / "test_predictions.csv"
    )
    results.to_csv(predictions_file, index=False)

    # Step 10: Save evaluation metrics
    metrics_file = ARTIFACTS_DIR / "evaluation_metrics.csv"

    metrics_df = pd.DataFrame([
        {"Metric": name, "Value": value}
        for name, value in metrics.items()
    ])
    metrics_df.to_csv(metrics_file, index=False)

    # Step 11: Display results
    print("\nEVALUATION RESULTS")
    print("-" * 40)
    print(f"MAE:  ₹{metrics['MAE_INR']:,.2f}")
    print(f"RMSE: ₹{metrics['RMSE_INR']:,.2f}")

    if pd.notna(metrics["WAPE_Percent"]):
        print(f"WAPE: {metrics['WAPE_Percent']:.2f}%")
    else:
        print("WAPE: Undefined (total actual cost is zero)")

    print(f"\nDaily predictions saved: {predictions_file}")
    print(f"Metrics saved: {metrics_file}")

    print(
        "\nInterpretation: MAE is the average absolute "
        "prediction error, RMSE penalizes large errors more "
        "strongly, and WAPE expresses total absolute error "
        "as a percentage of total actual expenditure."
    )


if __name__ == "__main__":
    main()
