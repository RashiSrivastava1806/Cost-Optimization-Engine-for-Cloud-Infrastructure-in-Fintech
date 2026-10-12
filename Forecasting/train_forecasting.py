
import numpy as np
import pandas as pd

from forecasting_utils import (
    TRAIN_FILE,
    ARTIFACTS_DIR,
    MODEL_FILE,
    load_billing_data,
    aggregate_daily_costs,
    create_time_features,
    prepare_model_matrix,
    predict_daily_costs,
)


def main():
    print("=" * 60)
    print("CLOUD COST OPTIMIZATION - FORECAST MODEL TRAINING")
    print("=" * 60)

    # Step 1: Load the training dataset
    df = load_billing_data(TRAIN_FILE)
    print(f"\nTraining records loaded: {len(df)}")

    # Step 2: Aggregate resource costs by calendar date
    daily_costs = aggregate_daily_costs(df)

    if len(daily_costs) < 7:
        raise ValueError(
            "At least 7 observed training dates are required."
        )

    print(f"Observed training dates: {len(daily_costs)}")
    print(f"Start date: {daily_costs.index.min().date()}")
    print(f"End date:   {daily_costs.index.max().date()}")

    # Step 3: Establish the reference date using training data only
    reference_date = daily_costs.index.min()

    # Step 4: Generate time-based features
    features = create_time_features(
        daily_costs.index,
        reference_date,
    )

    # Step 5: Prepare the feature matrix
    X = prepare_model_matrix(features)
    y = daily_costs.to_numpy(dtype=float)

    # Step 6: Train the linear regression model
    coefficients, _, rank, _ = np.linalg.lstsq(
        X,
        y,
        rcond=None,
    )

    if rank < X.shape[1]:
        raise ValueError(
            "Training matrix is rank-deficient. "
            "Check the training dates and features."
        )

    # Step 7: Generate training predictions for diagnostics
    predictions = predict_daily_costs(
        features,
        coefficients,
    )

    # Step 8: Create the artifacts directory
    ARTIFACTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Step 9: Save model coefficients and metadata
    np.savez_compressed(
        MODEL_FILE,
        coefficients=coefficients,
        reference_date=np.array(
            reference_date.strftime("%Y-%m-%d")
        ),
        feature_columns=np.array(
            features.columns.tolist()
        ),
        model_name=np.array(
            "Linear Trend with Weekday Effects"
        ),
    )

    # Step 10: Save training predictions
    training_results = pd.DataFrame({
        "Date": daily_costs.index,
        "Actual_Cost_INR": y,
        "Predicted_Cost_INR": predictions,
        "Absolute_Error_INR": np.abs(y - predictions),
    })

    results_file = (
        ARTIFACTS_DIR / "training_predictions.csv"
    )

    training_results.to_csv(
        results_file,
        index=False,
    )

    # Step 11: Display results
    print("\nMODEL TRAINING COMPLETED")
    print("-" * 40)
    print("Model: Linear Trend with Weekday Effects")
    print(f"Features: {', '.join(features.columns)}")
    print(
        f"Average actual daily cost: "
        f"₹{daily_costs.mean():,.2f}"
    )
    print(
        f"Average predicted daily cost: "
        f"₹{predictions.mean():,.2f}"
    )
    print(f"\nSaved model: {MODEL_FILE}")
    print(f"Saved diagnostics: {results_file}")

    print(
        "\nNote: Training predictions are in-sample results. "
        "We will evaluate generalization on the separate "
        "chronological test dataset."
    )


if __name__ == "__main__":
    main()
