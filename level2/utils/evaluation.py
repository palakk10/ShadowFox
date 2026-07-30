# =============================================================
# utils/evaluation.py
# Purpose: Model evaluation metrics and diagnostic plots.
#          Compares multiple models and visualises predictions.
# =============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PLOT_DIR = os.path.join("outputs", "plots")
REPORT_DIR = os.path.join("outputs", "reports")
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)

COLOR_MAIN = "#4F86C6"
COLOR_ACCENT = "#F4A261"
COLOR_TARGET = "#E76F51"


# ---------------------------------------------
# 1. Compute Regression Metrics
# ---------------------------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> dict:
    """
    Compute MAE, MSE, RMSE, and R2 Score.

    Parameters
    ----------
    y_true      : Actual target values.
    y_pred      : Predicted target values.
    model_name  : Label for display.

    Returns
    -------
    dict : Metrics dictionary.
    """
    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2   = r2_score(y_true, y_pred)

    metrics = {"Model": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

    print(f"\n{'-' * 50}")
    print(f"  Evaluation - {model_name}")
    print(f"{'-' * 50}")
    print(f"  MAE   : {mae:.4f}  (Rs. Lakhs - avg absolute error)")
    print(f"  MSE   : {mse:.4f}  (penalises large errors heavily)")
    print(f"  RMSE  : {rmse:.4f}  (same unit as target - Rs. Lakhs)")
    print(f"  R2    : {r2:.4f}  (% variance explained by model)")
    print(f"{'-' * 50}")
    return metrics


# ---------------------------------------------
# 2. Compare All Models
# ---------------------------------------------
def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Build a comparison table from a list of metric dicts.

    Parameters
    ----------
    results : List of dicts returned by compute_metrics().

    Returns
    -------
    pd.DataFrame : Sorted comparison table (best R2 first).
    """
    df_results = pd.DataFrame(results).set_index("Model")
    df_results = df_results.sort_values("R2", ascending=False)

    print("\n" + "=" * 60)
    print("  MODEL COMPARISON TABLE (Sorted by R2 - higher is better)")
    print("=" * 60)
    print(df_results.round(4).to_string())
    print("=" * 60)

    # Save to CSV
    report_path = os.path.join(REPORT_DIR, "model_comparison.csv")
    df_results.to_csv(report_path)
    print(f"\n[INFO] Comparison report saved -> {report_path}")

    # Bar chart
    _plot_model_comparison(df_results)

    return df_results


def _plot_model_comparison(df_results: pd.DataFrame) -> None:
    """Horizontal bar chart comparing R2 of all models."""
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = [COLOR_ACCENT if m == df_results.index[0] else COLOR_MAIN for m in df_results.index]
    bars = ax.barh(df_results.index, df_results["R2"], color=colors, edgecolor="white", height=0.5)

    for bar, val in zip(bars, df_results["R2"]):
        ax.text(bar.get_width() - 0.02, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", ha="right", color="white", fontweight="bold", fontsize=10)

    ax.set_xlabel("R2 Score (higher = better)", fontsize=11)
    ax.set_title("Model Comparison - R2 Score", fontsize=14, fontweight="bold")
    ax.set_xlim(0, 1.05)
    ax.axvline(0.8, color=COLOR_TARGET, linestyle="--", linewidth=1.5, label="0.80 threshold")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(PLOT_DIR, "10_model_comparison.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved -> {path}")


# ---------------------------------------------
# 3. Actual vs Predicted Plot
# ---------------------------------------------
def plot_actual_vs_predicted(y_test: np.ndarray, y_pred: np.ndarray,
                              model_name: str = "Model") -> None:
    """
    Scatter plot of actual vs predicted values.

    A perfect model would have all points on the red dashed diagonal.
    Points above the line = underestimation; below = overestimation.
    Our Random Forest model clusters tightly around the diagonal,
    indicating accurate predictions with few large errors.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    ax.scatter(y_test, y_pred, alpha=0.6, color=COLOR_MAIN,
               edgecolors="white", linewidth=0.4, s=60, label="Predictions")

    # Perfect prediction line
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            color=COLOR_TARGET, linewidth=2.5, linestyle="--", label="Perfect Fit")

    ax.set_xlabel("Actual Selling Price (Rs. Lakhs)", fontsize=12)
    ax.set_ylabel("Predicted Selling Price (Rs. Lakhs)", fontsize=12)
    ax.set_title(f"Actual vs Predicted - {model_name}", fontsize=14, fontweight="bold")
    ax.legend()

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "11_actual_vs_predicted.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved -> {path}")


# ---------------------------------------------
# 4. Residual Plot
# ---------------------------------------------
def plot_residuals(y_test: np.ndarray, y_pred: np.ndarray,
                   model_name: str = "Model") -> None:
    """
    Residual plot: residuals vs predicted values.

    Residual = Actual - Predicted.
    Ideally residuals should scatter randomly around 0 with no pattern.
    A funnel shape would indicate heteroscedasticity.
    Our model shows mostly random residuals with a few larger errors
    at high price points (luxury cars are harder to predict accurately).
    """
    residuals = np.array(y_test) - np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Residual Analysis - {model_name}", fontsize=14, fontweight="bold")

    # Scatter residuals vs predicted
    axes[0].scatter(y_pred, residuals, alpha=0.55, color=COLOR_MAIN,
                    edgecolors="white", linewidth=0.3, s=50)
    axes[0].axhline(0, color=COLOR_TARGET, linewidth=2, linestyle="--")
    axes[0].set_xlabel("Predicted Values", fontsize=11)
    axes[0].set_ylabel("Residuals (Actual - Predicted)", fontsize=11)
    axes[0].set_title("Residuals vs Predicted", fontsize=12, fontweight="bold")

    # Distribution of residuals
    sns.histplot(residuals, bins=25, kde=True, ax=axes[1],
                 color=COLOR_ACCENT, edgecolor="white", alpha=0.8)
    axes[1].axvline(0, color=COLOR_TARGET, linewidth=2, linestyle="--")
    axes[1].set_xlabel("Residual Value", fontsize=11)
    axes[1].set_ylabel("Frequency", fontsize=11)
    axes[1].set_title("Residual Distribution", fontsize=12, fontweight="bold")

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "12_residual_plot.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved -> {path}")


# ---------------------------------------------
# 5. Prediction Error Plot
# ---------------------------------------------
def plot_prediction_error(y_test: np.ndarray, y_pred: np.ndarray,
                           model_name: str = "Model") -> None:
    """
    Prediction error as percentage of actual value.

    Formula: Error% = |Actual - Predicted| / Actual x 100
    Helps see relative error regardless of scale.
    Most predictions should fall within ?15% for a good model.
    """
    y_test_arr = np.array(y_test)
    y_pred_arr = np.array(y_pred)

    # Avoid division by zero
    error_pct = np.abs(y_test_arr - y_pred_arr) / np.where(y_test_arr == 0, 1, y_test_arr) * 100

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(f"Prediction Error Analysis - {model_name}", fontsize=14, fontweight="bold")

    # Sample index vs error %
    indices = np.arange(len(error_pct))
    axes[0].bar(indices, error_pct, color=COLOR_MAIN, alpha=0.7, edgecolor="white")
    axes[0].axhline(15, color=COLOR_TARGET, linewidth=2, linestyle="--", label="15% threshold")
    axes[0].set_xlabel("Test Sample Index", fontsize=11)
    axes[0].set_ylabel("Prediction Error (%)", fontsize=11)
    axes[0].set_title("Prediction Error per Sample", fontsize=12, fontweight="bold")
    axes[0].legend()

    # Cumulative distribution of errors
    sorted_err = np.sort(error_pct)
    cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
    axes[1].plot(sorted_err, cdf, color=COLOR_ACCENT, linewidth=2.5)
    axes[1].axvline(15, color=COLOR_TARGET, linewidth=2, linestyle="--", label="15% threshold")
    axes[1].fill_between(sorted_err, cdf, alpha=0.2, color=COLOR_ACCENT)
    axes[1].set_xlabel("Prediction Error (%)", fontsize=11)
    axes[1].set_ylabel("Cumulative Fraction", fontsize=11)
    axes[1].set_title("CDF of Prediction Error", fontsize=12, fontweight="bold")
    axes[1].legend()

    within_15 = (error_pct <= 15).mean() * 100
    axes[1].text(0.65, 0.15, f"{within_15:.1f}% within 15%",
                 transform=axes[1].transAxes, fontsize=10, color="white",
                 bbox=dict(facecolor=COLOR_MAIN, alpha=0.8, boxstyle="round,pad=0.3"))

    plt.tight_layout()
    path = os.path.join(PLOT_DIR, "13_prediction_error.png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved -> {path}")
