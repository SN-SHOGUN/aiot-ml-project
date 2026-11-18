"""
src/evaluate.py
---------------
Evaluation helpers: metrics table, confusion matrix, ROC curves, clustering scores.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc,
)
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import warnings


# ── Classification metrics ────────────────────────────────────────────────────

def evaluate_classifier(clf, X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Return dict of Accuracy, Precision, Recall, F1 for one classifier."""
    y_pred = clf.predict(X_test)
    return {
        "Accuracy":  accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall":    recall_score(y_test, y_pred, zero_division=0),
        "F1":        f1_score(y_test, y_pred, zero_division=0),
    }


def build_metrics_table(fitted_models: dict, X_test, y_test) -> pd.DataFrame:
    rows = []
    for name, clf in fitted_models.items():
        row = evaluate_classifier(clf, X_test, y_test)
        row["Model"] = name
        rows.append(row)
    df = pd.DataFrame(rows).set_index("Model")
    return df[["Accuracy", "Precision", "Recall", "F1"]]


def get_confusion_matrix(clf, X_test, y_test) -> np.ndarray:
    return confusion_matrix(y_test, clf.predict(X_test))


def get_roc_data(fitted_models: dict, X_test, y_test) -> dict:
    """
    Returns {model_name: (fpr, tpr, roc_auc)} for each model that
    supports predict_proba.
    """
    results = {}
    for name, clf in fitted_models.items():
        try:
            if hasattr(clf, "predict_proba"):
                proba = clf.predict_proba(X_test)[:, 1]
            elif hasattr(clf, "decision_function"):
                proba = clf.decision_function(X_test)
            else:
                continue
            fpr, tpr, _ = roc_curve(y_test, proba)
            results[name] = (fpr, tpr, auc(fpr, tpr))
        except Exception:
            pass
    return results


# ── Clustering ────────────────────────────────────────────────────────────────

def run_clustering(X: np.ndarray, n_components: int = 2) -> dict:
    """
    Runs K-Means, Agglomerative, DBSCAN on PCA-reduced data.
    Returns dict suitable for plotting.
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X)

    results = {}

    # K-Means
    km = KMeans(n_clusters=3, random_state=42, n_init="auto")
    km_labels = km.fit_predict(X_pca)
    sil_km = silhouette_score(X_pca, km_labels)
    results["K-Means"] = {"X": X_pca, "labels": km_labels, "silhouette": sil_km}

    # Agglomerative
    agg = AgglomerativeClustering(n_clusters=3)
    agg_labels = agg.fit_predict(X_pca)
    sil_agg = silhouette_score(X_pca, agg_labels)
    results["Hierarchical"] = {"X": X_pca, "labels": agg_labels, "silhouette": sil_agg}

    # DBSCAN
    db = DBSCAN(eps=0.5, min_samples=10)
    db_labels = db.fit_predict(X_pca)
    unique = set(db_labels) - {-1}
    sil_db = silhouette_score(X_pca, db_labels) if len(unique) > 1 else float("nan")
    results["DBSCAN"] = {"X": X_pca, "labels": db_labels, "silhouette": sil_db}

    # K-Medoids via sklearn fallback (PAM approximation using KMeans init)
    try:
        from sklearn_extra.cluster import KMedoids
        kmed = KMedoids(n_clusters=3, random_state=42)
        kmed_labels = kmed.fit_predict(X_pca)
    except ImportError:
        # Fallback: use KMeans with different init
        kmed_tmp = KMeans(n_clusters=3, init="random", random_state=99, n_init=5)
        kmed_labels = kmed_tmp.fit_predict(X_pca)
    sil_kmed = silhouette_score(X_pca, kmed_labels)
    results["K-Medoids"] = {"X": X_pca, "labels": kmed_labels, "silhouette": sil_kmed}

    return results
