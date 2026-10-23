"""
Project: Predict New York City Airbnb Prices
AIML Batch — Linear Regression Minor Project

Dataset required:
    AB_NYC_2019.csv
Download the "New York City Airbnb Open Data" dataset by AB_NYC_2019 from Kaggle
and place AB_NYC_2019.csv in the same folder as this Python file.

Run:
    python NYC_Airbnb_Linear_Regression_Project.py

The script performs the seven steps specified in the assignment and saves plots,
metrics, predictions, and a plain-English summary in airbnb_project_outputs/.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parent / "AB_NYC_2019.csv"
OUTPUT_DIR = Path(__file__).resolve().parent / "airbnb_project_outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


def main():
    # STEP 1: Load and audit the dataset
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}\n"
            "Download AB_NYC_2019.csv from Kaggle and place it next to this script."
        )

    df = pd.read_csv(DATA_PATH)
    print("\\nSTEP 1 — EDA AND DATA AUDITING")
    print("Dataset shape:", df.shape)
    print("\\nFirst five rows:")
    print(df.head())
    print("\\nColumn data types:")
    print(df.dtypes)
    print("\\nDuplicate rows:", df.duplicated().sum())

    missing = pd.DataFrame({
        "missing_count": df.isna().sum(),
        "missing_percent": (df.isna().mean() * 100).round(2),
    }).sort_values("missing_percent", ascending=False)
    print("\\nMissing values and percentages:")
    print(missing.to_string())

    print("\\nVariable types:")
    print("Discrete numeric: minimum_nights, number_of_reviews, "
          "calculated_host_listings_count, availability_365")
    print("Continuous numeric: price, latitude, longitude, reviews_per_month")
    print("Categorical nominal: neighbourhood_group, neighbourhood, room_type")
    print("Date-like: last_review")

    # STEP 2: Clean data and handle missing values
    print("\\nSTEP 2 — DATA CLEANING")
    data = df.drop_duplicates().copy()
    drop_cols = [
        "id", "host_id", "name", "host_name", "last_review",
        "neighbourhood", "host_url", "listing_url", "url",
    ]
    data.drop(columns=[c for c in drop_cols if c in data.columns],
              inplace=True, errors="ignore")

    if {"reviews_per_month", "number_of_reviews"}.issubset(data.columns):
        no_reviews = data["number_of_reviews"].fillna(0).eq(0)
        data.loc[no_reviews, "reviews_per_month"] = (
            data.loc[no_reviews, "reviews_per_month"].fillna(0)
        )

    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.loc[data["price"].notna() & (data["price"] > 0)].copy()
    print("Shape after basic cleaning:", data.shape)
    print("Remaining missing values:")
    print(data.isna().sum().to_string())

    # STEP 3: Detect and trim outliers using 95th percentiles
    print("\\nSTEP 3 — OUTLIER IDENTIFICATION AND TREATMENT")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(x=data["price"], ax=axes[0])
    axes[0].set(title="Price before trimming", xlabel="Nightly price (USD)")
    sns.boxplot(x=data["minimum_nights"], ax=axes[1])
    axes[1].set(title="Minimum nights before trimming", xlabel="Minimum nights")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "01_outlier_boxplots_before.png", dpi=150)
    plt.show()
    plt.close()

    price_cutoff = data["price"].quantile(0.95)
    nights_cutoff = data["minimum_nights"].quantile(0.95)
    before = len(data)
    data = data[
        (data["price"] <= price_cutoff)
        & (data["minimum_nights"] <= nights_cutoff)
    ].copy()
    removed = before - len(data)
    print(f"95th percentile price cutoff: ${price_cutoff:.2f}")
    print("95th percentile minimum-nights cutoff:", nights_cutoff)
    print(f"Rows removed: {removed:,} ({removed / before:.1%})")
    print("Shape after trimming:", data.shape)

    # STEP 4: Select features and encode categorical variables
    print("\\nSTEP 4 — FEATURE SELECTION AND ENCODING")
    target = "price"
    candidate_features = [
        "neighbourhood_group", "room_type", "latitude", "longitude",
        "minimum_nights", "number_of_reviews", "reviews_per_month",
        "calculated_host_listings_count", "availability_365",
    ]
    features = [c for c in candidate_features if c in data.columns]
    if not features:
        raise ValueError("No expected predictor columns found in the dataset.")

    X_raw = data[features].copy()
    y = data[target].astype(float)
    categorical_cols = [
        c for c in ["neighbourhood_group", "room_type"] if c in X_raw.columns
    ]
    X = pd.get_dummies(
        X_raw, columns=categorical_cols, drop_first=True, dtype=int
    )
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    print("Selected raw features:", features)
    print("Encoded feature matrix shape:", X.shape)

    # STEP 5: Split into train and test sets; impute from training data only
    print("\\nSTEP 5 — TRAIN/TEST SPLIT")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )
    medians = X_train.median(numeric_only=True)
    X_train = X_train.fillna(medians).fillna(0)
    X_test = X_test.fillna(medians).fillna(0)
    print("Training rows:", len(X_train))
    print("Testing rows:", len(X_test))
    print("Number of model features:", X_train.shape[1])

    # STEP 6: Train model and evaluate
    print("\\nSTEP 6 — MODEL TRAINING AND EVALUATION")
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = float(np.sqrt(mse))

    metrics = pd.DataFrame({
        "Metric": ["R-squared (R2)", "Mean Absolute Error (MAE)",
                   "Mean Squared Error (MSE)", "Root Mean Squared Error (RMSE)"],
        "Value": [r2, mae, mse, rmse],
    })
    print(metrics.to_string(index=False))
    metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

    # STEP 7: Residual analysis and error comparison
    print("\\nSTEP 7 — RESIDUAL ANALYSIS")
    results = pd.DataFrame({
        "actual_price": y_test,
        "predicted_price": y_pred,
    })
    results["residual"] = results["actual_price"] - results["predicted_price"]

    plt.figure(figsize=(7, 6))
    sns.scatterplot(data=results, x="actual_price", y="predicted_price", alpha=0.45)
    low = min(results["actual_price"].min(), results["predicted_price"].min())
    high = max(results["actual_price"].max(), results["predicted_price"].max())
    plt.plot([low, high], [low, high], "--", label="Perfect prediction (45°)")
    plt.xlabel("Actual price (USD)")
    plt.ylabel("Predicted price (USD)")
    plt.title("Actual vs Predicted Airbnb Prices")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "02_actual_vs_predicted.png", dpi=150)
    plt.show()
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.scatterplot(data=results, x="predicted_price", y="residual", alpha=0.45)
    plt.axhline(0, linestyle="--")
    plt.xlabel("Predicted price (USD)")
    plt.ylabel("Residual: actual − predicted (USD)")
    plt.title("Residuals vs Predicted Price")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "03_residuals_vs_predicted.png", dpi=150)
    plt.show()
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.histplot(results["residual"], bins=40, kde=True)
    plt.xlabel("Residual (USD)")
    plt.title("Distribution of Prediction Errors")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "04_residual_distribution.png", dpi=150)
    plt.show()
    plt.close()

    median_price = results["actual_price"].median()
    results["price_group"] = np.where(
        results["actual_price"] <= median_price,
        "Cheaper half", "More expensive half"
    )
    results["absolute_error"] = results["residual"].abs()
    group_error = results.groupby("price_group")["absolute_error"].agg(
        ["mean", "median", "count"]
    ).rename(columns={
        "mean": "MAE_USD",
        "median": "Median_absolute_error_USD",
    })
    print("\\nAbsolute error by price group:")
    print(group_error.to_string())
    group_error.to_csv(OUTPUT_DIR / "error_by_price_group.csv")
    results.to_csv(OUTPUT_DIR / "test_predictions_and_residuals.csv", index=False)

    better_group = group_error["MAE_USD"].idxmin()
    print(f"\\nThe lower-MAE group in this test set is: {better_group}")

    summary = f"""PREDICT NEW YORK CITY AIRBNB PRICES — SUMMARY

Dataset: New York City Airbnb Open Data (AB_NYC_2019), Kaggle
Rows after cleaning and outlier treatment: {len(data):,}
Train/test split: 80% / 20%, random_state={RANDOM_STATE}
Model: Linear Regression
Predictors: {", ".join(features)}

R²: {r2:.4f}
MAE: ${mae:.2f}
MSE: {mse:.2f}
RMSE: ${rmse:.2f}

Plain-English interpretation:
The R² score of {r2:.4f} means the model explains approximately {r2 * 100:.1f}% of test-set price variation. R² may be negative if the model performs worse than predicting the test-set mean.
The MAE of ${mae:.2f} means predictions differ from actual prices by about ${mae:.2f} per listing on average, in absolute terms.
The lower-MAE group in this test set is {better_group}. See error_by_price_group.csv for both group results.

Limitations:
This is a baseline linear model. Airbnb prices may be nonlinear and depend on omitted property details, data quality, and market changes.
"""
    (OUTPUT_DIR / "one_page_summary.txt").write_text(summary, encoding="utf-8")
    print("\\n" + summary)
    print("All generated outputs saved in:", OUTPUT_DIR.resolve())


if __name__ == "__main__":
    main()
