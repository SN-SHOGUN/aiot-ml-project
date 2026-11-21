"""Dataset-agnostic split, preprocess, and classical training.

The split happens first. Imputer and scaler fit on the training rows only.
"""
from __future__ import annotations

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
        name = str(col).lower()
        nunique = int(series.nunique(dropna=True))
        if nunique <= 1 or series.isna().mean() > 0.30:
            drops.append(str(col))
        elif n > 10 and nunique / n > 0.95:
            drops.append(str(col))
        elif any(token in name for token in ("id", "uuid", "guid")):
            drops.append(str(col))
    return drops


def _classification(y: pd.Series) -> bool:
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_object_dtype(y) or pd.api.types.is_string_dtype(y):
        return True
    return int(y.nunique(dropna=True)) <= 12


def _preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include="number").columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    parts = []
    if numeric:
        parts.append((
            "num",
            Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("sc", StandardScaler()),
            ]),
            numeric,
        ))
    if categorical:
        parts.append((
            "cat",
            Pipeline([
                ("imp", SimpleImputer(strategy="most_frequent")),
                ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)),
            ]),
            categorical,
        ))
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
    """Return train/test matrices. Preprocessor is fit on train only."""
    if target not in df.columns:
        raise ValueError(f"Unknown target column: {target}")
    drop_cols = list(drop_cols or [])
    unknown = [c for c in drop_cols + ([group_col] if group_col else []) if c not in df.columns]
    if unknown:
        raise ValueError(f"Unknown columns: {unknown}")
    if group_col and group_col not in drop_cols:
        drop_cols.append(group_col)
    work = df.dropna(subset=[target]).copy()
    if len(work) < 8:
        raise ValueError("Need at least 8 rows with a target value.")
    y = work[target]
    if not _classification(y):
        raise ValueError(f"'{target}' looks continuous. Choose a categorical or low-cardinality target.")
    if int(y.nunique()) < 2:
        raise ValueError("The target has only one class.")
    X = work.drop(columns=[target] + [c for c in drop_cols if c in work.columns])
    if X.shape[1] == 0:
        raise ValueError("No feature columns left after drops.")
    if group_col:
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
        train_idx, test_idx = next(splitter.split(X, y, work[group_col]))
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        strat = y if y.value_counts().min() >= 2 else None
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
    """Default classical classifiers. Quick mode is logistic regression and a forest."""
    models = {
        "Logistic Regression": LogisticRegression(max_iter=400, random_state=SEED),
        "Random Forest": RandomForestClassifier(
            n_estimators=80, max_depth=8, random_state=SEED, n_jobs=1,
        ),
    }
    if not quick:
        models["Decision Tree"] = DecisionTreeClassifier(max_depth=4, random_state=SEED)
    return models


def _binary(y: pd.Series) -> bool:
    return int(pd.Series(y).nunique()) == 2


def _metric_row(name: str, model, X_test, y_test) -> dict:
    pred = model.predict(X_test)
    binary = _binary(y_test)
    average = "binary" if binary else "macro"
    pr_auc = np.nan
    if binary and hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        positive = classes[-1]
        scores = model.predict_proba(X_test)[:, classes.index(positive)]
        pr_auc = float(average_precision_score((pd.Series(y_test) == positive).astype(int), scores))
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
    if target not in df.columns or len(df) < 8:
        return pd.DataFrame(columns=columns)
    y = df[target]
    if int(y.nunique(dropna=True)) < 2:
        return pd.DataFrame(columns=columns)
    try:
        X_dummy = df.drop(columns=[target])
        strat = y if y.value_counts().min() >= 2 else None
        tr, te = train_test_split(df.index, test_size=0.2, random_state=SEED, stratify=strat)
    except ValueError:
        tr, te = train_test_split(df.index, test_size=0.2, random_state=SEED)
    y_tr, y_te = y.loc[tr], y.loc[te]
    majority = float(y_te.value_counts(normalize=True).iloc[0])
    rows = []
    for col in list(X_dummy.columns)[:15]:
        model = DecisionTreeClassifier(max_depth=2, random_state=SEED)
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="most_frequent")),
            ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=20)
             if not pd.api.types.is_numeric_dtype(df[col]) else "passthrough"),
            ("m", model),
        ])
        if pd.api.types.is_numeric_dtype(df[col]):
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("m", model),
            ])
        try:
            pipe.fit(df.loc[tr, [col]], y_tr)
            acc = float(accuracy_score(y_te, pipe.predict(df.loc[te, [col]])))
        except Exception:
            continue
        rows.append({
            "feature": col,
            "holdout_accuracy": round(acc, 4),
            "majority_baseline": round(majority, 4),
            "shortcut": bool(acc >= 0.95 and acc - majority >= 0.05),
        })
    return pd.DataFrame(rows, columns=columns)


def _fit_eval(df: pd.DataFrame, target: str, drop_cols, group_col, quick: bool, cv: bool) -> dict:
    packed = build_dataset(df, target, drop_cols=drop_cols, group_col=group_col)
    y_train, y_test = packed["y_train"], packed["y_test"]
    majority_label = y_train.mode().iloc[0]
    baseline = float(accuracy_score(y_test, np.full(len(y_test), majority_label)))
    rows = []
    fitted = {}
    for name, model in get_models(quick=quick).items():
        model.fit(packed["X_train"], y_train)
        row = _metric_row(name, model, packed["X_test"], y_test)
        if cv:
            try:
                scores = cross_val_score(model, packed["X_train"], y_train, cv=3, scoring="accuracy")
                row["cv_accuracy"] = float(scores.mean())
            except Exception:
                row["cv_accuracy"] = np.nan
        rows.append(row)
        fitted[name] = model
    table = pd.DataFrame(rows)
    best_name = str(table.sort_values(["f1", "accuracy"], ascending=False).iloc[0]["model"])
    best = fitted[best_name]
    labels = list(best.classes_)
    cm = confusion_matrix(y_test, best.predict(packed["X_test"]), labels=labels)
    return {
        "comparison": table,
        "baseline_accuracy": baseline,
        "best_name": best_name,
        "best_model": best,
        "confusion": cm,
        "labels": labels,
        "packed": packed,
    }


def train_and_evaluate(
    df: pd.DataFrame,
    target: str,
    drop_cols: list[str] | None = None,
    group_col: str | None = None,
    cv: bool = False,
) -> dict:
    """Fit the default models and score them on the held-out split."""
    return _fit_eval(df, target, drop_cols, group_col, quick=False, cv=cv)


def quick_model_eval(df: pd.DataFrame, target: str) -> dict:
    """Logistic regression and random forest, no cross-validation, plus a plain verdict."""
    result = _fit_eval(df, target, None, None, quick=True, cv=False)
    best_acc = float(result["comparison"]["accuracy"].max())
    gain = (best_acc - result["baseline_accuracy"]) * 100
    if gain < 5:
        verdict = "The model barely beats guessing"
    else:
        verdict = f"The model beats guessing by {gain:.1f} points"
    result["verdict"] = verdict
    return result
