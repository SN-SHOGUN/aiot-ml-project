"""
ml_pipeline.py
--------------
Auto-detect task type (classification vs regression), preprocess, train a suite
of models, evaluate them, and return structured results.
Optimized for high speed on larger datasets (subsampling during training, optimized n_estimators, n_jobs=-1).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    AdaBoostClassifier,
    AdaBoostRegressor,
    GradientBoostingClassifier,
    GradientBoostingRegressor,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    LinearRegression,
    LogisticRegression,
    Ridge,
    Lasso,
)
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.svm import LinearSVC, LinearSVR, SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

SEED = 42


# ── Task detection ─────────────────────────────────────────────────────────────

def detect_task(y: pd.Series) -> str:
    """Return 'classification' or 'regression'."""
    if pd.api.types.is_bool_dtype(y):
        return "classification"
    if pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
        return "classification"
    n_unique = int(y.nunique(dropna=True))
    if n_unique <= 15:
        return "classification"
    return "regression"


# ── Preprocessing ──────────────────────────────────────────────────────────────

def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X.columns if c not in numeric]
    parts = []
    if numeric:
        parts.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc",  StandardScaler()),
        ]), numeric))
    if categorical:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh",  OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=25)),
        ]), categorical))
    if not parts:
        raise ValueError("No usable feature columns found.")
    return ColumnTransformer(parts)


def prepare_data(df: pd.DataFrame, target: str, test_size: float = 0.2):
    """Split, preprocess, and return packed dict."""
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found.")
    work = df.dropna(subset=[target]).copy()
    if len(work) < 10:
        raise ValueError("Need at least 10 non-null rows in the target column.")
    y_raw = work[target]
    task = detect_task(y_raw)

    le = None
    if task == "classification":
        le = LabelEncoder()
        y = pd.Series(le.fit_transform(y_raw.astype(str)), name=target)
    else:
        y = y_raw.astype(float)

    X = work.drop(columns=[target])
    # Drop columns with >50% missing or constant
    X = X[[c for c in X.columns
            if X[c].isna().mean() <= 0.5 and X[c].nunique(dropna=True) > 1]]
    if X.shape[1] == 0:
        raise ValueError("No usable feature columns remain after cleaning.")

    strat = y if task == "classification" and int(y.value_counts().min()) >= 2 else None
    try:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size,
                                                   random_state=SEED, stratify=strat)
    except ValueError:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=SEED)

    pre = build_preprocessor(X_tr)
    pre.fit(X_tr)
    return {
        "task": task,
        "X_tr_raw": X_tr, "X_te_raw": X_te,
        "X_tr": pre.transform(X_tr), "X_te": pre.transform(X_te),
        "y_tr": y_tr.values, "y_te": y_te.values,
        "preprocessor": pre,
        "label_encoder": le,
        "feature_names": list(X.columns),
        "target": target,
    }


# ── Fast Model catalogues ───────────────────────────────────────────────────────

def _clf_models():
    return {
        "Logistic Regression":    LogisticRegression(max_iter=300, random_state=SEED),
        "Decision Tree":          DecisionTreeClassifier(max_depth=6, random_state=SEED),
        "Random Forest":          RandomForestClassifier(n_estimators=60, max_depth=10, random_state=SEED, n_jobs=-1),
        "AdaBoost":               AdaBoostClassifier(n_estimators=40, random_state=SEED),
        "Gradient Boosting":      GradientBoostingClassifier(n_estimators=40, max_depth=4, random_state=SEED),
        "Neural Network (MLP)":   MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=150, random_state=SEED),
        "KNN":                    KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM":                    LinearSVC(dual="auto", max_iter=1000, random_state=SEED),
        "Naive Bayes":            GaussianNB(),
    }


def _reg_models():
    return {
        "Linear Regression":      LinearRegression(),
        "Ridge Regression":       Ridge(alpha=1.0),
        "Lasso Regression":       Lasso(alpha=0.01, max_iter=1000),
        "Decision Tree":          DecisionTreeRegressor(max_depth=6, random_state=SEED),
        "Random Forest":          RandomForestRegressor(n_estimators=60, max_depth=10, random_state=SEED, n_jobs=-1),
        "AdaBoost":               AdaBoostRegressor(n_estimators=40, random_state=SEED),
        "Gradient Boosting":      GradientBoostingRegressor(n_estimators=40, max_depth=4, random_state=SEED),
        "Neural Network (MLP)":   MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=150, random_state=SEED),
        "KNN":                    KNeighborsRegressor(n_neighbors=5, n_jobs=-1),
        "SVR":                    LinearSVR(dual="auto", max_iter=1000, random_state=SEED),
    }


# ── Evaluation ─────────────────────────────────────────────────────────────────

def _eval_clf(name, model, X_te, y_te):
    pred = model.predict(X_te)
    binary = len(np.unique(y_te)) == 2
    avg = "binary" if binary else "macro"
    row = {
        "Model": name,
        "Accuracy": round(accuracy_score(y_te, pred), 4),
        "Recall":   round(recall_score(y_te, pred, average=avg, zero_division=0), 4),
        "F1":       round(f1_score(y_te, pred, average=avg, zero_division=0), 4),
    }
    if binary and hasattr(model, "predict_proba"):
        try:
            row["ROC-AUC"] = round(roc_auc_score(y_te, model.predict_proba(X_te)[:, 1]), 4)
        except Exception:
            row["ROC-AUC"] = float("nan")
    return row, pred


def _eval_reg(name, model, X_te, y_te):
    pred = model.predict(X_te)
    row = {
        "Model": name,
        "R²":    round(r2_score(y_te, pred), 4),
        "MAE":   round(mean_absolute_error(y_te, pred), 4),
        "RMSE":  round(np.sqrt(mean_squared_error(y_te, pred)), 4),
    }
    return row, pred


# ── Main training function (with speed optimizations) ──────────────────────────

def train_all_models(packed: dict) -> dict:
    """Train every selected model and return metrics + fitted models fast."""
    task = packed["task"]
    X_tr_full, X_te = packed["X_tr"], packed["X_te"]
    y_tr_full, y_te = packed["y_tr"], packed["y_te"]

    # Speed optimization: if training data exceeds 15,000 rows, sample for model fitting
    # to make training 10x faster while preserving statistical integrity
    if len(X_tr_full) > 15_000:
        idx = np.random.RandomState(SEED).choice(len(X_tr_full), size=15_000, replace=False)
        X_tr = X_tr_full[idx]
        y_tr = y_tr_full[idx]
    else:
        X_tr = X_tr_full
        y_tr = y_tr_full

    catalogue = _clf_models() if task == "classification" else _reg_models()
    rows, fitted, preds = [], {}, {}

    for name, model in catalogue.items():
        try:
            model.fit(X_tr, y_tr)
            if task == "classification":
                row, pred = _eval_clf(name, model, X_te, y_te)
            else:
                row, pred = _eval_reg(name, model, X_te, y_te)
            rows.append(row)
            fitted[name] = model
            preds[name] = pred
        except Exception as exc:
            rows.append({"Model": name, "Error": str(exc)})

    metrics = pd.DataFrame(rows).set_index("Model")
    # pick best model
    sort_col = "Accuracy" if task == "classification" else "R²"
    if sort_col in metrics.columns:
        best_name = metrics[sort_col].astype(float).idxmax()
    else:
        best_name = list(fitted.keys())[0] if fitted else None

    # confusion matrix for best clf
    cm = None
    if task == "classification" and best_name in fitted:
        best_pred = preds[best_name]
        cm = confusion_matrix(y_te, best_pred)

    return {
        "task": task,
        "metrics": metrics,
        "fitted": fitted,
        "predictions": preds,
        "best_name": best_name,
        "confusion_matrix": cm,
        "y_te": y_te,
        "label_encoder": packed["label_encoder"],
    }


# ── Prediction on new input ────────────────────────────────────────────────────

def predict_single(packed: dict, fitted: dict, model_name: str, user_input: dict):
    model = fitted[model_name]
    pre   = packed["preprocessor"]
    le    = packed["label_encoder"]
    feat  = packed["feature_names"]

    row = pd.DataFrame([{c: user_input.get(c, np.nan) for c in feat}])
    X   = pre.transform(row)
    pred = model.predict(X)[0]

    proba = None
    if hasattr(model, "predict_proba"):
        try:
            proba = model.predict_proba(X)[0]
        except Exception:
            pass

    label = le.inverse_transform([int(pred)])[0] if le is not None else pred
    return {"raw": pred, "label": label, "proba": proba,
            "classes": le.classes_ if le is not None else None}
