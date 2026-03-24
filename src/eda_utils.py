"""
eda_utils.py
------------
Visualization utilities for Credit Card Fraud EDA.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid", palette="muted")


def plot_class_distribution(df: pd.DataFrame, target: str = "Class", save_path: str = None):
    """Bar chart + pie chart of fraud vs legitimate transactions."""

    counts = df[target].value_counts().rename({0: "Legitimate", 1: "Fraud"})
    fraud_pct = (counts["Fraud"] / counts.sum()) * 100

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Bar plot
    counts.plot(kind="bar", color=["steelblue", "crimson"], ax=axes[0], edgecolor="white")
    axes[0].set_title("Transaction Class Count")
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis="x", rotation=0)

    for bar in axes[0].patches:
        axes[0].annotate(
            f"{int(bar.get_height()):,}",
            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
            ha="center",
            va="bottom",
        )

    # Pie chart
    axes[1].pie(
        counts,
        labels=counts.index,
        autopct="%1.2f%%",
        colors=["steelblue", "crimson"],
        startangle=90,
    )
    axes[1].set_title("Class Distribution (%)")

    plt.suptitle(
        f"Fraud vs Legitimate Transactions (Fraud = {fraud_pct:.3f}%)",
        fontsize=14,
        fontweight="bold",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")

    plt.show()


def plot_correlation_heatmap(df: pd.DataFrame, save_path: str = None):
    """Heatmap of feature correlations."""

    corr = df.corr()

    fig, ax = plt.subplots(figsize=(20, 16))

    mask = np.triu(np.ones_like(corr, dtype=bool))

    sns.heatmap(
        corr,
        mask=mask,
        cmap="coolwarm",
        center=0,
        annot=False,
        linewidths=0.3,
        ax=ax,
    )

    ax.set_title("Feature Correlation Heatmap", fontsize=16, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")

    plt.show()


def plot_amount_distribution(df: pd.DataFrame, save_path: str = None):
    """Transaction amount distribution for fraud vs legitimate."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    labels = {0: ("Legitimate", "steelblue"), 1: ("Fraud", "crimson")}

    for cls, (label, color) in labels.items():
        ax = axes[cls]

        data = df[df["Class"] == cls]["Amount"]

        ax.hist(data, bins=60, color=color, edgecolor="white", alpha=0.85)

        ax.set_title(f"Amount Distribution — {label}")
        ax.set_xlabel("Transaction Amount")
        ax.set_ylabel("Frequency")

        ax.annotate(
            f"Median: ${data.median():.2f}\nMax: ${data.max():.2f}",
            xy=(0.65, 0.85),
            xycoords="axes fraction",
            fontsize=9,
            bbox=dict(boxstyle="round", fc="white", alpha=0.7),
        )

    plt.suptitle("Transaction Amount by Class", fontsize=14, fontweight="bold")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")

    plt.show()


def plot_time_distribution(df: pd.DataFrame, save_path: str = None):
    """Transaction time distribution for fraud vs legitimate."""

    fig, ax = plt.subplots(figsize=(12, 5))

    for cls, (label, color) in {
        0: ("Legitimate", "steelblue"),
        1: ("Fraud", "crimson"),
    }.items():

        subset = df[df["Class"] == cls]["Time"]

        ax.hist(
            subset,
            bins=60,
            color=color,
            alpha=0.6,
            label=label,
            edgecolor="white",
        )

    ax.set_title("Transaction Time Distribution by Class", fontsize=14, fontweight="bold")
    ax.set_xlabel("Time (seconds from first transaction)")
    ax.set_ylabel("Frequency")
    ax.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved -> {save_path}")

    plt.show()


def plot_amount_boxplot(df: pd.DataFrame, save_path: str = None):
    """Boxplot comparing transaction amount by class."""

    plt.figure(figsize=(8, 5))

    sns.boxplot(x="Class", y="Amount", data=df)

    plt.title("Transaction Amount vs Fraud")
    plt.xlabel("Class (0 = Legitimate, 1 = Fraud)")
    plt.ylabel("Transaction Amount")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150)
        print(f"Saved -> {save_path}")

    plt.show()


def plot_top_correlations(df: pd.DataFrame, target: str = "Class", top_n: int = 10):
    """Show features most correlated with fraud."""

    corr = df.corr()[target].drop(target)

    top_corr = corr.abs().sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))

    sns.barplot(x=top_corr.values, y=top_corr.index)

    plt.title(f"Top {top_n} Features Correlated with Fraud")

    plt.xlabel("Correlation with Fraud")

    plt.tight_layout()

    plt.show()