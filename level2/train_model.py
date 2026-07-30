# =============================================================
# train_model.py
# Purpose : Complete end-to-end training pipeline.
#           Runs Steps 1-10: Data Loading -> Preprocessing ->
#           EDA -> Model Training -> Hyperparameter Tuning ->
#           Evaluation -> Save Model.
# Usage   : python train_model.py
# =============================================================

import sys
import io
# Force UTF-8 output so rupee / bullet characters print correctly
# on Windows terminals that default to cp1252.
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import os
import pickle
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore")

# -- Ensure utils is importable regardless of CWD ------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import RandomizedSearchCV

from utils.preprocessing import run_preprocessing_pipeline, load_data, feature_engineering
from utils.visualization import (
    plot_histograms,
    plot_distributions,
    plot_boxplots,
    plot_count_plots,
    plot_pie_charts,
    plot_pairplot,
    plot_scatter,
    plot_correlation_heatmap,
    plot_feature_importance,
)
from utils.evaluation import (
    compute_metrics,
    compare_models,
    plot_actual_vs_predicted,
    plot_residuals,
    plot_prediction_error,
)

# -- Paths ----------------------------------------------------
DATA_PATH         = "car.csv"
ENCODER_PATH      = "encoder.pkl"
MODEL_PATH        = "model.pkl"
MODEL_DIR_PATH    = os.path.join("models", "random_forest_model.pkl")
PREDICTIONS_PATH  = os.path.join("outputs", "predictions.csv")
os.makedirs("outputs", exist_ok=True)
os.makedirs("models", exist_ok=True)

RANDOM_STATE = 42


# =============================================================
# STEP 1 - PROBLEM DEFINITION (printed overview)
# =============================================================
def print_problem_definition() -> None:
    """
    Describe the business problem, objective, and variable types.
    """
    banner = "=" * 65
    print(f"\n{banner}")
    print("  CAR SELLING PRICE PREDICTION - ML PROJECT")
    print(banner)
    print("""
  BUSINESS PROBLEM
  ----------------------------------------------------------
  Used-car dealers and private sellers struggle to price
  their vehicles accurately. Over-pricing loses buyers;
  under-pricing means lost revenue. A data-driven pricing
  tool removes guesswork and brings market-aligned estimates.

  OBJECTIVE
  ----------------------------------------------------------
  Build a regression model that predicts the Selling_Price
  of a used car (in Rs. Lakhs) given its attributes.

  PROBLEM TYPE
  ----------------------------------------------------------
  Supervised Learning -> Regression
  Target variable is continuous (price in Rs. Lakhs).

  INPUT VARIABLES (Features)
  ----------------------------------------------------------
  [+] Present_Price  - Showroom price (Rs. Lakhs)
  [+] Car_Age        - Years since manufacture
  [+] Kms_Driven     - Total kilometres driven
  [+] Fuel_Type      - Petrol / Diesel / CNG
  [+] Seller_Type    - Dealer / Individual
  [+] Transmission   - Manual / Automatic
  [+] Owner          - No. of previous owners (0,1,2,3)

  OUTPUT VARIABLE (Target)
  ----------------------------------------------------------
  [>] Selling_Price  - Estimated resale price (Rs. Lakhs)
""")
    print(banner + "\n")


# =============================================================
# STEP 4 - EDA (uses raw df before encoding)
# =============================================================
def run_eda(df_raw: pd.DataFrame, df_encoded: pd.DataFrame) -> None:
    """
    Execute all EDA visualisation functions.

    Parameters
    ----------
    df_raw     : Original dataframe after feature engineering (has Car_Age,
                 but categoricals still as strings).
    df_encoded : Fully encoded numeric dataframe.
    """
    print("\n" + "=" * 65)
    print("  STEP 4 - EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 65)

    # -- Numerical Analysis -----------------------------------
    print("\n[A] Numerical Analysis")
    print("  -> Generating Histograms...")
    plot_histograms(df_encoded)

    print("  -> Generating Distribution Plots...")
    plot_distributions(df_encoded)

    print("  -> Generating Box Plots...")
    plot_boxplots(df_encoded)

    # -- Categorical Analysis ---------------------------------
    print("\n[B] Categorical Analysis")
    print("  -> Generating Count Plots ...")
    plot_count_plots(df_raw)

    print("  -> Generating Pie Charts ...")
    plot_pie_charts(df_raw)

    # -- Relationship Analysis --------------------------------
    print("\n[C] Relationship Analysis")
    print("  -> Generating Pair Plot ...")
    plot_pairplot(df_encoded)

    print("  -> Generating Scatter Plots ...")
    plot_scatter(df_encoded)

    print("  -> Generating Correlation Heatmap ...")
    plot_correlation_heatmap(df_encoded)

    print("\n[INFO] All EDA plots saved to outputs/plots/\n")


# =============================================================
# STEP 7 - MODEL BUILDING
# =============================================================
def build_models() -> dict:
    """
    Define all regression models to train and compare.

    Returns
    -------
    dict : Model name -> sklearn estimator instance.
    """
    return {
        "Linear Regression":         LinearRegression(),
        "Decision Tree Regressor":   DecisionTreeRegressor(random_state=RANDOM_STATE),
        "Random Forest Regressor":   RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
        "Gradient Boosting Regressor": GradientBoostingRegressor(random_state=RANDOM_STATE),
    }


def train_and_compare(models: dict, X_train, y_train, X_test, y_test) -> list:
    """
    Train every model and collect evaluation metrics.

    Returns
    -------
    list : List of metric dicts (one per model).
    """
    print("\n" + "=" * 65)
    print("  STEP 7 - MODEL BUILDING & COMPARISON")
    print("=" * 65)

    results = []
    for name, model in models.items():
        print(f"\n  Training : {name} ...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        metrics = compute_metrics(y_test, y_pred, model_name=name)
        results.append(metrics)

    return results


# =============================================================
# STEP 8 - HYPERPARAMETER TUNING (RandomizedSearchCV)
# =============================================================
def tune_random_forest(X_train, y_train) -> RandomForestRegressor:
    """
    Tune Random Forest using RandomizedSearchCV.

    Parameters searched:
      - n_estimators    : Number of trees
      - max_depth       : Maximum tree depth
      - min_samples_split : Min samples to split an internal node
      - min_samples_leaf  : Min samples at a leaf node
      - max_features    : Features to consider at each split

    Returns
    -------
    RandomForestRegressor : Best estimator after tuning.
    """
    print("\n" + "=" * 65)
    print("  STEP 8 - HYPERPARAMETER TUNING (RandomizedSearchCV)")
    print("=" * 65)

    param_grid = {
        "n_estimators":      [100, 200, 300, 400, 500],
        "max_depth":         [None, 5, 10, 15, 20, 25],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf":  [1, 2, 4],
        "max_features":      ["sqrt", "log2", 0.5, None],
    }

    rf_base = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)

    rs_cv = RandomizedSearchCV(
        estimator=rf_base,
        param_distributions=param_grid,
        n_iter=40,               # 40 random combinations
        cv=5,                    # 5-fold cross-validation
        scoring="r2",
        verbose=1,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\n  Running RandomizedSearchCV (40 iterations x 5-fold CV)...")
    rs_cv.fit(X_train, y_train)

    print(f"\n  Best Parameters Found:")
    for param, val in rs_cv.best_params_.items():
        print(f"    {param:<22} : {val}")
    print(f"\n  Best CV R2 Score : {rs_cv.best_score_:.4f}")

    return rs_cv.best_estimator_


# =============================================================
# STEP 9 - FINAL EVALUATION
# =============================================================
def evaluate_final_model(best_model, X_test, y_test, feature_names) -> None:
    """
    Full evaluation of the final tuned Random Forest model.
    Generates all diagnostic plots.
    """
    print("\n" + "=" * 65)
    print("  STEP 9 - FINAL MODEL EVALUATION")
    print("=" * 65)

    y_pred = best_model.predict(X_test)

    # Metrics
    metrics = compute_metrics(y_test, y_pred, model_name="Tuned Random Forest")

    # Plots
    print("\n  -> Generating Actual vs Predicted plot...")
    plot_actual_vs_predicted(y_test.values, y_pred, "Tuned Random Forest")

    print("  -> Generating Residual plot...")
    plot_residuals(y_test.values, y_pred, "Tuned Random Forest")

    print("  -> Generating Prediction Error plot...")
    plot_prediction_error(y_test.values, y_pred, "Tuned Random Forest")

    print("  -> Generating Feature Importance plot...")
    plot_feature_importance(feature_names, best_model.feature_importances_)

    # Result interpretation
    print(f"""
  ---------------------------------------------------------
  RESULT INTERPRETATION
  ---------------------------------------------------------
  R2   = {metrics['R2']:.4f}  -> Model explains {metrics['R2']*100:.1f}% of variance
         in Selling_Price. Above 0.85 is excellent.
  RMSE = {metrics['RMSE']:.4f} Rs. Lakhs -> Avg deviation from actual price.
  MAE  = {metrics['MAE']:.4f} Rs. Lakhs -> Median absolute prediction error.

  The Actual vs Predicted plot shows tight clustering around
  the diagonal - confirming strong predictive power.
  Residuals randomly scattered around 0 = no systematic bias.
  ---------------------------------------------------------
""")
    return y_pred


# =============================================================
# STEP 10 - SAVE MODEL
# =============================================================
def save_model(model, model_path: str, model_dir_path: str) -> None:
    """
    Persist the trained model using pickle.

    Saves to two locations:
      - model.pkl                      (root - used by Flask app)
      - models/random_forest_model.pkl (organised models/ folder)
    """
    print("\n" + "=" * 65)
    print("  STEP 10 - SAVING MODEL")
    print("=" * 65)

    for path in [model_path, model_dir_path]:
        with open(path, "wb") as f:
            pickle.dump(model, f)
        print(f"  [SAVED] {path}")

    print("\n  Model saved successfully. Ready for deployment.\n")


# =============================================================
# MAIN PIPELINE
# =============================================================
def main():
    # -- Step 1: Problem Definition ---------------------------
    print_problem_definition()

    # -- Steps 2 & 3: Data Loading + Preprocessing ------------
    print("=" * 65)
    print("  STEPS 2 & 3 - DATA COLLECTION & PREPROCESSING")
    print("=" * 65)

    (
        X_train, X_test, y_train, y_test,
        df_encoded, encoder
    ) = run_preprocessing_pipeline(
        filepath=DATA_PATH,
        encoder_save_path=ENCODER_PATH,
    )

    # Keep raw df (before encoding) for categorical EDA plots
    df_raw = load_data(DATA_PATH)
    df_raw = feature_engineering(df_raw)

    # -- Step 4: EDA ------------------------------------------
    run_eda(df_raw, df_encoded)

    # -- Steps 5 & 6 are embedded in preprocessing ------------
    print("=" * 65)
    print("  STEP 5 - FEATURE SET PREPARED")
    print(f"  Features : {list(X_train.columns)}")
    print("=" * 65)

    # -- Step 7: Train all models -----------------------------
    models = build_models()
    results = train_and_compare(models, X_train, y_train, X_test, y_test)

    # Compare table
    df_compare = compare_models(results)

    # -- Step 8: Hyperparameter Tuning -----------------------
    best_model = tune_random_forest(X_train, y_train)

    # -- Step 9: Evaluate final model -------------------------
    feature_names = list(X_train.columns)
    y_pred = evaluate_final_model(best_model, X_test, y_test, feature_names)

    # Save predictions CSV
    pred_df = pd.DataFrame({
        "Actual_Price":    y_test.values,
        "Predicted_Price": y_pred,
        "Error":           y_test.values - y_pred,
    })
    pred_df.to_csv(PREDICTIONS_PATH, index=False)
    print(f"  [SAVED] Predictions -> {PREDICTIONS_PATH}")

    # -- Step 10: Save model -----------------------------------
    save_model(best_model, MODEL_PATH, MODEL_DIR_PATH)

    print("\n" + "=" * 65)
    print("  PIPELINE COMPLETE")
    print("  Run 'python app.py' to launch the Flask web application.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
