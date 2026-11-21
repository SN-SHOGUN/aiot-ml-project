"""Dataset-agnostic split, preprocess, and classical training.

Split first. Fit the imputer and scaler on the training rows only.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

SEED = 42


def _id_like(name: str) -> bool:
    text = str(name).lower().replace("-", "_")
    return bool(re.search(r"(^|_)(id|uuid|guid)$", text) or text.endswith(" id") or "uuid" in text)


def suggest_target(df: pd.DataFrame) -> str | None:
    """Pick a likely label column, or None."""
    keys = ("target", "label", "failure", "winner", "churn", "outcome", "class")
    for col in df.columns:
        if any(k in str(col).lower() for k in keys):
            return str(col)
    for col in reversed(list(df.columns)):
        nunique = int(df[col].nunique(dropna=True))
        if 2 <= nunique <= 8 and nunique < max(len(df) * 0.5, 2):
            return str(col)
    return None


def suggest_drop_cols(df: pd.DataFrame, target: str | None = None) -> list[str]:
    """ID-like, constant, and mostly-missing columns."""
    drops: list[str] = []
    n = max(len(df), 1)
    for col in df.columns:
        if col == target:
            continue
        series = df[col]
        nunique = int(series.nunique(dropna=True))
        if nunique <= 1 or float(series.isna().mean()) > 0.30 or _id_like(col):
            drops.append(str(col))
        elif n > 10 and nunique / n > 0.95:
            drops.append(str(col))
    return drops


def _is_classification(y: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
        return True
    return int(y.nunique(dropna=True)) <= 12


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = list(X.select_dtypes(include="number").columns)
    categorical = [c for c in X.columns if c not in numeric]
    parts = []
    if numeric:
        parts.append(("num", Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
        ]), numeric))
    if categorical:
        parts.append(("cat", Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
        ]), categorical))
    if not parts:
        raise ValueError("No usable feature columns.")
    return ColumnTransformer(parts)


def build_dataset(
    df: pd.DataFrame,
    target: str,
    drop_cols: list[str] | None = None,
    group_col: str | None = None,
    test_size: float = 0.2,
) -> dict:
    """Train/test matrices. The preprocessor is fit on the train split only."""
    if target not in df.columns:
        raise ValueError(f"Unknown target column: {target}")
    drop_cols = list(drop_cols or [])
    extra = [group_col] if group_col else []
    unknown = [c for c in drop_cols + extra if c not in df.columns]
    if unknown:
        raise ValueError(f"Unknown columns: {unknown}")
    if group_col and group_col not in drop_cols:
        drop_cols.append(group_col)
    work = df.dropna(subset=[target]).copy()
    if len(work) < 8:
        raise ValueError("Need at least 8 rows with a target value.")
    y = work[target]
    if not _is_classification(y):
        raise ValueError(f"'{target}' looks continuous. Choose a categorical target.")
    if int(y.nunique()) < 2:
        raise ValueError("The target has only one class.")
    X = work.drop(columns=[target] + [c for c in drop_cols if c in work.columns])
    if X.shape[1] == 0:
        raise ValueError("No feature columns left after drops.")
    if group_col:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
        train_pos, test_pos = next(splitter.split(X, y, work[group_col]))
        X_train, X_test = X.iloc[train_pos], X.iloc[test_pos]
        y_train, y_test = y.iloc[train_pos], y.iloc[test_pos]
    else:
        strat = y if int(y.value_counts().min()) >= 2 else None
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=SEED, stratify=strat,
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=SEED,
            )
    pre = _preprocessor(X_train)
    pre.fit(X_train)
    return {
        "X_train": pre.transform(X_train),
        "X_test": pre.transform(X_test),
        "y_train": y_train,
        "y_test": y_test,
        "preprocessor": pre,
        "train_index": X_train.index,
        "test_index": X_test.index,
    }


def get_models(quick: bool = False) -> dict:
    """Classical classifiers. Quick mode keeps logistic regression and a forest."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=400, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=60, max_depth=8, random_state=SEED, n_jobs=1,
        ),
    }
    if not quick:
        models["Decision Tree"] = DecisionTreeClassifier(max_depth=4, random_state=SEED)
    return models


def _metric_row(name: str, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    binary = int(pd.Series(y_test).nunique()) == 2
    average = "binary" if binary else "macro"
    pr_auc = np.nan
    if binary and hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        positive = classes[-1]
        scores = model.predict_proba(X_test)[:, classes.index(positive)]
        truth = (pd.Series(y_test).to_numpy() == positive).astype(int)
        pr_auc = float(average_precision_score(truth, scores))
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "recall": float(recall_score(y_test, pred, average=average, zero_division=0)),
        "f1": float(f1_score(y_test, pred, average=average, zero_division=0)),
        "pr_auc": pr_auc,
    }


def single_feature_check(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Holdout accuracy of a one-column tree. Flags near-perfect shortcuts."""
    columns = ["feature", "holdout_accuracy", "majority_baseline", "shortcut"]
    if target not in df.columns or len(df) < 8 or int(df[target].nunique(dropna=True)) < 2:
        return pd.DataFrame(columns=columns)
    y = df[target]
    try:
        strat = y if int(y.value_counts().min()) >= 2 else None
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=strat)
    except ValueError:
        train_idx, test_idx = train_test_split(df.index, test_size=0.2, random_state=SEED)
    y_train, y_test = y.loc[train_idx], y.loc[test_idx]
    majority = float(y_test.value_counts(normalize=True).iloc[0])
    rows = []
    for col in [c for c in df.columns if c != target][:15]:
        if pd.api.types.is_numeric_dtype(df[col]):
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("m", DecisionTreeClassifier(max_depth=2, random_state=SEED)),
            ])
        else:
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)),
                ("m", DecisionTreeClassifier(max_depth=2, random_state=SEED)),
            ])
        try:
            pipe.fit(df.loc[train_idx, [col]], y_train)
            acc = float(accuracy_score(y_test, pipe.predict(df.loc[test_idx, [col]])))
        except Exception:
            continue
        rows.append({
            "feature": col,
            "holdout_accuracy": round(acc, 4),
            "majority_baseline": round(majority, 4),
            "shortcut": bool(acc >= 0.95 and acc - majority >= 0.05),
        })
    return pd.DataFrame(rows, columns=columns)


def _fit_eval(df, target, drop_cols, group_col, quick: bool, cv: bool) -> dict:
    packed = build_dataset(df, target, drop_cols=drop_cols, group_col=group_col)
    y_train, y_test = packed["y_train"], packed["y_test"]
    baseline = float(accuracy_score(y_test, np.full(len(y_test), y_train.mode().iloc[0])))
    rows, fitted = [], {}
    for name, model in get_models(quick=quick).items():
        model.fit(packed["X_train"], y_train)
        row = _metric_row(name, model, packed["X_test"], y_test)
        if cv:
            try:
                scores = cross_val_score(model, packed["X_train"], y_train, cv=3, scoring="accuracy")
                row["cv_accuracy"] = float(np.mean(scores))
            except Exception:
                row["cv_accuracy"] = np.nan
        rows.append(row)
        fitted[name] = model
    table = pd.DataFrame(rows)
    best_name = str(table.sort_values(["f1", "accuracy"], ascending=False).iloc[0]["model"])
    best = fitted[best_name]
    labels = list(best.classes_)
    matrix = confusion_matrix(y_test, best.predict(packed["X_test"]), labels=labels)
    return {
        "comparison": table,
        "baseline_accuracy": baseline,
        "best_name": best_name,
        "confusion": matrix,
        "labels": labels,
    }


def train_and_evaluate(df, target, drop_cols=None, group_col=None, cv: bool = False) -> dict:
    """Fit the default models and score the held-out split."""
    return _fit_eval(df, target, drop_cols, group_col, quick=False, cv=cv)


def quick_model_eval(df, target) -> dict:
    """Logistic regression and random forest, no CV, plus a plain-language verdict."""
    result = _fit_eval(df, target, None, None, quick=True, cv=False)
    gain = (float(result["comparison"]["accuracy"].max()) - result["baseline_accuracy"]) * 100
    if gain < 5:
        result["verdict"] = "The model barely beats guessing"
    else:
        result["verdict"] = f"The model beats guessing by {gain:.1f} points"
    return result
