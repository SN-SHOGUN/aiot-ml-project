"""
src/viz.py
----------
All visualisation helpers. Each function returns a matplotlib Figure.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.tree import plot_tree
import pathlib, warnings

FIG_DIR = pathlib.Path(__file__).parent.parent / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

PALETTE = "viridis"
COLORS  = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860", "#DA8BC3"]
TIERS   = ["LOCAL", "EDGE", "CLOUD"]


def save(fig: plt.Figure, name: str):
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


# ── 1. Class distribution ─────────────────────────────────────────────────────

def fig_class_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 4))
    counts = df["Machine failure"].value_counts().sort_index()
    bars = ax.bar(["No Failure (0)", "Failure (1)"], counts.values,
                  color=["#4C72B0", "#C44E52"], edgecolor="white", linewidth=1.2)
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
                f"{int(b.get_height())}", ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax.set_title("Class Distribution — Machine Failure", fontsize=14, fontweight="bold")
    ax.set_ylabel("Count")
    ax.set_xlabel("Class")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "01_class_distribution")
    return fig


# ── 2. Correlation heatmap ────────────────────────────────────────────────────

def fig_correlation_heatmap(df: pd.DataFrame) -> plt.Figure:
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    corr = df[num_cols].corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 7})
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "02_correlation_heatmap")
    return fig


# ── 3. Boxplots by failure class ──────────────────────────────────────────────

def fig_boxplots(df: pd.DataFrame) -> plt.Figure:
    features = ["Torque [Nm]", "Rotational speed [rpm]", "Tool wear [min]"]
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    for ax, feat in zip(axes, features):
        data = [df.loc[df["Machine failure"] == 0, feat].dropna(),
                df.loc[df["Machine failure"] == 1, feat].dropna()]
        bp = ax.boxplot(data,
                        patch_artist=True, widths=0.5)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(["No Failure", "Failure"])
        for patch, color in zip(bp["boxes"], ["#4C72B0", "#C44E52"]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_ylabel(feat)
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle("Feature Distributions by Failure Class", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "03_boxplots")
    return fig


# ── 4. Model comparison ───────────────────────────────────────────────────────

def fig_model_comparison(metrics_df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 5))
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(metrics_df))
    width = 0.18
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, metrics_df[metric].values, width,
               label=metric, color=COLORS[i], alpha=0.85)
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(metrics_df.index, rotation=20, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "04_model_comparison")
    return fig


# ── 5. Confusion matrix ───────────────────────────────────────────────────────

def fig_confusion_matrix(cm: np.ndarray, model_name: str = "Best Model") -> plt.Figure:
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No Failure", "Failure"],
                yticklabels=["No Failure", "Failure"],
                linewidths=1, linecolor="white")
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("Actual", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save(fig, "05_confusion_matrix")
    return fig


# ── 6. ROC curves ─────────────────────────────────────────────────────────────

def fig_roc_curves(roc_data: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, (fpr, tpr, auc_val)) in enumerate(roc_data.items()):
        ax.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})",
                color=COLORS[i % len(COLORS)], linewidth=1.8)
    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves — All Classifiers", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "06_roc_curves")
    return fig


# ── 7. Feature importance + Decision Tree ─────────────────────────────────────

def fig_feature_importance(rf_model, feature_names: list[str]) -> plt.Figure:
    importances = rf_model.feature_importances_
    idx = np.argsort(importances)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.bar(range(len(importances)),
                  importances[idx],
                  color=plt.cm.viridis(np.linspace(0.2, 0.8, len(importances))))
    ax.set_xticks(range(len(importances)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Importance")
    ax.set_title("Random Forest Feature Importances", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save(fig, "07a_feature_importance")
    return fig


def fig_decision_tree(dt_model, feature_names: list[str]) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(16, 7))
    plot_tree(dt_model, feature_names=feature_names,
              class_names=["No Failure", "Failure"],
              filled=True, rounded=True, fontsize=8, ax=ax,
              impurity=False, proportion=False)
    ax.set_title("Decision Tree (depth ≤ 3)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "07b_decision_tree")
    return fig


# ── 8. PCA clustering scatter ─────────────────────────────────────────────────

def fig_clustering(cluster_results: dict) -> plt.Figure:
    n = len(cluster_results)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    cmap = plt.cm.tab10
    for ax, (name, res) in zip(axes, cluster_results.items()):
        X_pca = res["X"]
        labels = res["labels"]
        unique = np.unique(labels)
        for lbl in unique:
            mask = labels == lbl
            color = "gray" if lbl == -1 else cmap(lbl / max(1, len(unique) - 1))
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=[color], s=5, alpha=0.6, label=f"C{lbl}" if lbl != -1 else "Noise")
        sil = res["silhouette"]
        ax.set_title(f"{name}\nSilhouette={sil:.3f}" if not np.isnan(sil)
                     else f"{name}\nSilhouette=N/A", fontsize=11, fontweight="bold")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.grid(alpha=0.2)
    fig.suptitle("PCA 2D Clustering Visualisation", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "08_clustering_pca")
    return fig


# ── 9. DL training curves ─────────────────────────────────────────────────────

def fig_dl_curves(ann_history: dict, lstm_history: dict) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    pairs = [
        (ann_history,  "ANN"),
        (lstm_history, "LSTM"),
    ]
    for col, (hist, label) in enumerate(pairs):
        epochs = range(1, len(hist["loss"]) + 1)
        axes[0, col].plot(epochs, hist["loss"],     label="Train Loss",     color="#4C72B0")
        axes[0, col].plot(epochs, hist["val_loss"], label="Val Loss",       color="#C44E52", linestyle="--")
        axes[0, col].set_title(f"{label} — Loss", fontweight="bold")
        axes[0, col].set_xlabel("Epoch"); axes[0, col].set_ylabel("Loss")
        axes[0, col].legend(); axes[0, col].grid(alpha=0.3)

        axes[1, col].plot(epochs, hist["accuracy"],     label="Train Acc", color="#55A868")
        axes[1, col].plot(epochs, hist["val_accuracy"], label="Val Acc",   color="#DD8452", linestyle="--")
        axes[1, col].set_title(f"{label} — Accuracy", fontweight="bold")
        axes[1, col].set_xlabel("Epoch"); axes[1, col].set_ylabel("Accuracy")
        axes[1, col].legend(); axes[1, col].grid(alpha=0.3)

    fig.suptitle("Deep Learning Training Curves", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save(fig, "09_dl_curves")
    return fig


# ── 10. Offloading Lab charts ─────────────────────────────────────────────────

def fig_offloading_pie(sim_results: dict, policy: str = "threshold") -> plt.Figure:
    records = sim_results[policy]
    tiers = [r["tier"] for r in records]
    counts = {t: tiers.count(t) for t in TIERS}
    # Filter out zero-count tiers for pie chart
    labels_nonzero = [t for t, c in counts.items() if c > 0]
    values_nonzero = [c for t, c in counts.items() if c > 0]
    colors_all = {"LOCAL": "#4C72B0", "EDGE": "#55A868", "CLOUD": "#C44E52"}
    colors_nonzero = [colors_all[t] for t in labels_nonzero]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie(
        values_nonzero,
        labels=labels_nonzero,
        autopct="%1.1f%%",
        colors=colors_nonzero,
        startangle=140,
        wedgeprops={"edgecolor": "white", "linewidth": 1.5},
    )
    ax.set_title(f"Tier Distribution — {policy.title()} Policy",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "10a_offloading_pie")
    return fig


def fig_latency_histogram(sim_results: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5))
    colors_map = {"threshold": "#4C72B0", "qlearning": "#55A868",
                  "always_cloud": "#C44E52", "always_local": "#DD8452"}
    for policy, records in sim_results.items():
        lats = [r["latency_ms"] for r in records]
        ax.hist(lats, bins=40, alpha=0.5, label=policy.replace("_", " ").title(),
                color=colors_map.get(policy, "gray"), density=True)
    ax.axvline(x=100, color="black", linestyle="--", linewidth=1.5, label="100 ms budget")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Density")
    ax.set_title("Latency Distribution by Policy", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save(fig, "10b_latency_histogram")
    return fig


def fig_budget_met(sim_results: dict) -> plt.Figure:
    policies = list(sim_results.keys())
    pct_met = [
        100 * np.mean([r["met_budget"] for r in sim_results[p]])
        for p in policies
    ]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(policies, pct_met, color=["#4C72B0", "#55A868", "#C44E52", "#DD8452"])
    for bar, pct in zip(bars, pct_met):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontweight="bold")
    ax.set_xlim(0, 110)
    ax.set_xlabel("% Requests Meeting Budget")
    ax.set_title("Budget Met — Policy Comparison", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "10c_budget_met")
    return fig


# ── 11. A* Gantt-style chart ──────────────────────────────────────────────────

def fig_astar_gantt(assignments: list[dict]) -> plt.Figure:
    tier_colors = {"LOCAL": "#4C72B0", "EDGE": "#55A868", "CLOUD": "#C44E52"}
    fig, ax = plt.subplots(figsize=(12, max(4, len(assignments) * 0.4)))
    cum_time = 0.0
    for i, task in enumerate(assignments):
        lat = task["actual_latency"]
        color = tier_colors[task["tier"]]
        ax.barh(i, lat, left=cum_time, color=color, edgecolor="white",
                linewidth=0.5, alpha=0.85, height=0.7)
        # Budget marker
        ax.plot(cum_time + task["budget_ms"], i, "k|", markersize=12, linewidth=2)
        label_x = cum_time + lat / 2
        ax.text(label_x, i, task["tier"][:1], ha="center", va="center",
                fontsize=7, color="white", fontweight="bold")
        cum_time += lat

    tier_patches = [mpatches.Patch(color=c, label=t) for t, c in tier_colors.items()]
    tier_patches.append(mpatches.Patch(color="none", label="| = budget"))
    ax.legend(handles=tier_patches, loc="upper right", fontsize=9)
    ax.set_yticks(range(len(assignments)))
    ax.set_yticklabels([f"T{a['task_id']}" for a in assignments], fontsize=7)
    ax.set_xlabel("Cumulative Time (ms)")
    ax.set_title("A* Scheduler — Task-to-Tier Assignment (Gantt)", fontsize=13, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    save(fig, "11_astar_gantt")
    return fig
