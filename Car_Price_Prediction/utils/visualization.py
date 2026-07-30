# =============================================================
# utils/visualization.py
# Purpose: All EDA (Exploratory Data Analysis) and model
#          evaluation plots. Each function saves to outputs/plots/
# =============================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

# -- Global plot style ----------------------------------------
sns.set_theme(style="darkgrid", palette="muted", font_scale=1.1)
PLOT_DIR = os.path.join("outputs", "plots")
os.makedirs(PLOT_DIR, exist_ok=True)

COLOR_MAIN = "#4F86C6"
COLOR_ACCENT = "#F4A261"
COLOR_TARGET = "#E76F51"


def _save(fig: plt.Figure, filename: str) -> None:
    """Save figure to outputs/plots/ and close."""
    path = os.path.join(PLOT_DIR, filename)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  [PLOT] Saved -> {path}")


# =============================================================
# -- SECTION A: Numerical Analysis ----------------------------
# =============================================================

def plot_histograms(df: pd.DataFrame) -> None:
    """
    Histogram for every numerical column.

    What it tells us:
    - Shows the frequency distribution of each feature.
    - Skewed distributions (e.g., Selling_Price, Kms_Driven) indicate
      the data is not normally distributed; most cars cluster at lower
      values with a long right tail.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()
    fig.suptitle("Histograms - Numerical Features", fontsize=16, fontweight="bold", y=1.01)

    for i, col in enumerate(num_cols):
        axes[i].hist(df[col].dropna(), bins=30, color=COLOR_MAIN, edgecolor="white", alpha=0.85)
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Frequency")

    # Hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, "01_histograms.png")


def plot_distributions(df: pd.DataFrame) -> None:
    """
    KDE (Distribution) plot for numerical columns.

    What it tells us:
    - Smooth estimate of the probability density function.
    - Selling_Price distribution is right-skewed, meaning most used cars
      sell at lower prices, and only a few luxury/premium cars sell high.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()
    fig.suptitle("Distribution Plots (KDE) - Numerical Features", fontsize=16, fontweight="bold", y=1.01)

    for i, col in enumerate(num_cols):
        sns.kdeplot(df[col].dropna(), ax=axes[i], fill=True, color=COLOR_ACCENT, alpha=0.6, linewidth=2)
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].set_xlabel(col)
        axes[i].set_ylabel("Density")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, "02_distributions.png")


def plot_boxplots(df: pd.DataFrame) -> None:
    """
    Box plots for numerical columns to detect outliers.

    What it tells us:
    - Box plots reveal median, IQR, and outliers.
    - Kms_Driven and Present_Price show strong outliers, meaning
      some cars have extremely high values versus the typical range.
    - These outliers are real-world data (luxury cars, high-mileage vehicles)
      and are kept intentionally.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    n = len(num_cols)
    ncols = 3
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(16, nrows * 4))
    axes = axes.flatten()
    fig.suptitle("Box Plots - Outlier Detection", fontsize=16, fontweight="bold", y=1.01)

    for i, col in enumerate(num_cols):
        sns.boxplot(y=df[col].dropna(), ax=axes[i], color=COLOR_MAIN, width=0.4,
                    flierprops=dict(marker="o", markerfacecolor=COLOR_TARGET, markersize=4))
        axes[i].set_title(col, fontsize=12, fontweight="bold")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout()
    _save(fig, "03_boxplots.png")


# =============================================================
# -- SECTION B: Categorical Analysis --------------------------
# =============================================================

def plot_count_plots(df_raw: pd.DataFrame) -> None:
    """
    Count plots for categorical columns.

    What it tells us:
    - Fuel_Type  : Petrol dominates, followed by Diesel, then CNG.
    - Seller_Type: Most listings are from Dealers, not Individuals.
    - Transmission: Manual cars far outnumber Automatic.
    - Owner      : Majority of used cars have 0 previous owners (first owner).
    These insights guide feature importance understanding.
    """
    cat_cols = ["Fuel_Type", "Seller_Type", "Transmission", "Owner"]
    # Filter to columns that actually exist in the raw df
    cat_cols = [c for c in cat_cols if c in df_raw.columns]

    fig, axes = plt.subplots(1, len(cat_cols), figsize=(18, 5))
    if len(cat_cols) == 1:
        axes = [axes]
    fig.suptitle("Count Plots - Categorical Features", fontsize=16, fontweight="bold")

    palette = sns.color_palette("muted", 6)
    for i, col in enumerate(cat_cols):
        order = df_raw[col].value_counts().index
        sns.countplot(x=col, data=df_raw, ax=axes[i], order=order,
                      palette=palette, edgecolor="white")
        axes[i].set_title(col, fontsize=12, fontweight="bold")
        axes[i].set_xlabel("")
        axes[i].set_ylabel("Count")
        # Add value labels on bars
        for bar in axes[i].patches:
            axes[i].annotate(
                f"{int(bar.get_height())}",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                ha="center", va="bottom", fontsize=9, fontweight="bold",
            )

    plt.tight_layout()
    _save(fig, "04_count_plots.png")


def plot_pie_charts(df_raw: pd.DataFrame) -> None:
    """
    Pie charts for categorical columns.

    What it tells us:
    - Shows proportional share of each category.
    - Helps quickly see dominant classes (e.g., Petrol > Diesel).
    """
    cat_cols = ["Fuel_Type", "Seller_Type", "Transmission"]
    cat_cols = [c for c in cat_cols if c in df_raw.columns]

    fig, axes = plt.subplots(1, len(cat_cols), figsize=(16, 6))
    if len(cat_cols) == 1:
        axes = [axes]
    fig.suptitle("Pie Charts - Category Proportions", fontsize=16, fontweight="bold")

    palette = sns.color_palette("Set2", 6)
    for i, col in enumerate(cat_cols):
        counts = df_raw[col].value_counts()
        axes[i].pie(
            counts.values,
            labels=counts.index,
            autopct="%1.1f%%",
            startangle=140,
            colors=palette[: len(counts)],
            wedgeprops=dict(edgecolor="white", linewidth=1.5),
        )
        axes[i].set_title(col, fontsize=13, fontweight="bold")

    plt.tight_layout()
    _save(fig, "05_pie_charts.png")


# =============================================================
# -- SECTION C: Relationship Analysis -------------------------
# =============================================================

def plot_pairplot(df: pd.DataFrame) -> None:
    """
    Pair plot of numerical features.

    What it tells us:
    - Reveals pairwise relationships between all numeric features.
    - Strong positive correlation between Present_Price and Selling_Price.
    - Car_Age shows a negative relationship (older -> lower price).
    - Kms_Driven also shows slight negative correlation with price.
    """
    # Use a sample to keep the plot fast
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    # Limit to key columns for clarity
    key_cols = [c for c in ["Selling_Price", "Present_Price", "Kms_Driven", "Car_Age", "Owner"]
                if c in num_cols]

    fig = sns.pairplot(
        df[key_cols].sample(min(150, len(df)), random_state=42),
        diag_kind="kde",
        plot_kws={"alpha": 0.55, "color": COLOR_MAIN, "edgecolor": "none"},
        diag_kws={"fill": True, "color": COLOR_ACCENT},
    )
    fig.figure.suptitle("Pair Plot - Key Numerical Features", y=1.02, fontsize=14, fontweight="bold")

    _save(fig.figure, "06_pairplot.png")


def plot_scatter(df: pd.DataFrame) -> None:
    """
    Scatter plots: Selling_Price vs key features.

    What it tells us:
    - vs Present_Price: Strong linear/positive trend -> the market price
      of the showroom model is a very strong predictor of resale value.
    - vs Kms_Driven: Slight negative trend -> more driven = lower resale.
    - vs Car_Age: Clear negative trend -> newer cars fetch higher prices.
    """
    pairs = [
        ("Present_Price", "Selling_Price"),
        ("Kms_Driven", "Selling_Price"),
        ("Car_Age", "Selling_Price"),
    ]
    pairs = [(x, y) for (x, y) in pairs if x in df.columns and y in df.columns]

    fig, axes = plt.subplots(1, len(pairs), figsize=(18, 5))
    if len(pairs) == 1:
        axes = [axes]
    fig.suptitle("Scatter Plots - Selling Price vs Key Features", fontsize=15, fontweight="bold")

    for i, (x_col, y_col) in enumerate(pairs):
        axes[i].scatter(df[x_col], df[y_col], alpha=0.55, color=COLOR_MAIN,
                        edgecolors="white", linewidth=0.3, s=50)
        # Trend line
        z = np.polyfit(df[x_col], df[y_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[x_col].min(), df[x_col].max(), 200)
        axes[i].plot(x_line, p(x_line), color=COLOR_TARGET, linewidth=2, label="Trend")
        axes[i].set_xlabel(x_col, fontsize=11)
        axes[i].set_ylabel(y_col, fontsize=11)
        axes[i].set_title(f"{y_col} vs {x_col}", fontsize=12, fontweight="bold")
        axes[i].legend()

    plt.tight_layout()
    _save(fig, "07_scatter_plots.png")


def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Correlation heatmap of all numeric features.

    What it tells us:
    - Values close to +1 indicate strong positive correlation.
    - Values close to -1 indicate strong negative correlation.
    - Present_Price (0.88+) is the single strongest predictor.
    - Car_Age and Owner have moderate negative correlation with price.
    - Kms_Driven has a weak negative correlation.
    """
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()

    fig, ax = plt.subplots(figsize=(12, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool))  # Show lower triangle only

    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        center=0,
        linewidths=0.5,
        linecolor="white",
        square=True,
        ax=ax,
        cbar_kws={"shrink": 0.75},
    )
    ax.set_title("Correlation Heatmap - Numeric Features", fontsize=15, fontweight="bold", pad=15)
    plt.tight_layout()
    _save(fig, "08_correlation_heatmap.png")


def plot_feature_importance(feature_names: list, importances: np.ndarray) -> None:
    """
    Horizontal bar chart of Random Forest feature importances.

    What it tells us:
    - Quantifies how much each feature contributes to predictions.
    - Present_Price is typically the #1 feature (~50-60% importance).
    - Car_Age ranks second, confirming age significantly affects resale.
    - Kms_Driven contributes moderately.
    - Categorical encodings (Fuel_Type, Transmission) matter less overall.
    """
    sorted_idx = np.argsort(importances)
    sorted_feats = [feature_names[i] for i in sorted_idx]
    sorted_imps = importances[sorted_idx]

    palette = [COLOR_ACCENT if imp == sorted_imps.max() else COLOR_MAIN for imp in sorted_imps]

    fig, ax = plt.subplots(figsize=(10, max(6, len(feature_names) * 0.5)))
    bars = ax.barh(sorted_feats, sorted_imps, color=palette, edgecolor="white")

    # Value labels
    for bar, val in zip(bars, sorted_imps):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=9, color="white" if val == sorted_imps.max() else "black")

    ax.set_xlabel("Feature Importance Score", fontsize=11)
    ax.set_title("Random Forest - Feature Importance", fontsize=14, fontweight="bold")
    ax.set_xlim(0, sorted_imps.max() * 1.15)
    plt.tight_layout()
    _save(fig, "09_feature_importance.png")
