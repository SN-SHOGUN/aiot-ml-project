"""
src/generic.py
--------------
Dataset-agnostic pipeline: works on any CSV once the user picks a target column.

Leakage rules enforced here
  * split happens FIRST; imputation, category lumping and scaling are fit on train only
  * optional group split (e.g. by film_id) so the same entity never sits in both sets
  * group column is never used as a feature
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GroupShuffleSplit, cross_validate, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             average_precision_score, roc_auc_score)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

SEED = 42
MAX_ROWS = 100_000


# ── helpers ───────────────────────────────────────────────────────────────────
def suggest_drop_cols(df: pd.DataFrame, target: str) -> list[str]:
    """ID-like (>95% unique text/int), constant, and *_id columns."""
    out = []
    n = len(df)
    for c in df.columns:
        if c == target:
            continue
        nu = df[c].nunique(dropna=True)
        is_text = df[c].dtype == object or str(df[c].dtype).startswith(("str", "string", "category"))
        if nu <= 1 or (nu > 0.95 * n and (is_text or pd.api.types.is_integer_dtype(df[c]))) \
                or str(c).lower().endswith(("_id", " id")) or str(c).lower() == "id":
            out.append(c)
    return out


def suggest_target(df: pd.DataFrame) -> str:
    for name in ("machine failure", "outcome", "target", "label", "class", "y"):
        for c in df.columns:
            if str(c).lower() == name:
                return c
    return df.columns[-1]


def parse_body(s) -> str:
    """'Academy Award for Best Actor' -> 'Academy Award'; 'Golden Bear: x' -> 'Golden Bear'."""
    if not isinstance(s, str):
        return "Unknown"
    for sep in (" for ", ": ", " - "):
        if sep in s:
            return s.split(sep)[0].strip()
    return s.strip()


def _is_cat(s: pd.Series) -> bool:
    return not pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)


# ── main builder ──────────────────────────────────────────────────────────────
def build_dataset(df: pd.DataFrame, target: str, drop_cols: list[str] | None = None,
                  group_col: str | None = None, test_size: float = 0.2, top_n: int = 30,
                  parse_text: bool = False, seed: int = SEED) -> dict:
    drop_cols = [c for c in (drop_cols or []) if c in df.columns and c != target]
    df = df.dropna(subset=[target]).reset_index(drop=True)
    if len(df) > MAX_ROWS:
        df = df.sample(MAX_ROWS, random_state=seed).reset_index(drop=True)

    le = LabelEncoder()
    y_all = le.fit_transform(df[target].astype(str))
    class_names = [str(c) for c in le.classes_]
    if len(class_names) < 2:
        raise ValueError("Target has fewer than 2 classes.")
    if len(class_names) > 30:
        raise ValueError(f"Target has {len(class_names)} classes - this looks like a numeric/regression target.")

    groups = df[group_col].values if group_col else None
    feat_drop = set(drop_cols) | {target} | ({group_col} if group_col else set())
    X = df.drop(columns=[c for c in feat_drop if c in df.columns]).copy()
    X = X.drop(columns=[c for c in X.columns if X[c].nunique(dropna=True) <= 1])

    if parse_text:
        for c in [c for c in X.columns if _is_cat(X[c]) and X[c].nunique() > top_n]:
            X[c + "_body"] = X[c].map(parse_body)
            X = X.drop(columns=[c])

    # split FIRST
    idx = np.arange(len(df))
    if groups is not None:
        tr, te = next(GroupShuffleSplit(1, test_size=test_size, random_state=seed).split(idx, groups=groups))
    else:
        strat = y_all if np.bincount(y_all).min() >= 2 else None
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed, stratify=strat)
    Xtr, Xte = X.iloc[tr].copy(), X.iloc[te].copy()
    ytr, yte = y_all[tr], y_all[te]

    # categorical: lump to top-N (train only), one-hot
    cat_cols = [c for c in X.columns if _is_cat(X[c])]
    for c in cat_cols:
        top = set(Xtr[c].astype(str).value_counts().head(top_n).index)
        Xtr[c] = Xtr[c].astype(str).where(Xtr[c].astype(str).isin(top), "Other")
        Xte[c] = Xte[c].astype(str).where(Xte[c].astype(str).isin(top), "Other")
    # numeric: missing flags + train median
    num_cols = [c for c in X.columns if c not in cat_cols]
    for c in num_cols:
        Xtr[c] = pd.to_numeric(Xtr[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        Xte[c] = pd.to_numeric(Xte[c], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if Xtr[c].isna().any():
            Xtr[c + "_missing"] = Xtr[c].isna().astype(int)
            Xte[c + "_missing"] = Xte[c].isna().astype(int)
        med = Xtr[c].median()
        Xtr[c] = Xtr[c].fillna(med)
        Xte[c] = Xte[c].fillna(med)
    both = pd.get_dummies(pd.concat([Xtr, Xte], keys=["tr", "te"]), columns=cat_cols, dtype=float)
    Xtr_d, Xte_d = both.loc["tr"], both.loc["te"]
    feats = list(Xtr_d.columns)
    scaler = StandardScaler().fit(Xtr_d.values.astype(float))
    counts = np.bincount(y_all)
    return dict(Xtr=scaler.transform(Xtr_d.values.astype(float)), Xte=scaler.transform(Xte_d.values.astype(float)),
                ytr=ytr, yte=yte, features=feats, class_names=class_names,
                pos=int(np.argmin(counts)) if len(class_names) == 2 else None,  # minority = positive
                raw_tr=Xtr, raw_te=Xte, n_rows=len(df), group_split=groups is not None,
                baseline_acc=float(counts.max() / counts.sum()))


# ── models & evaluation ───────────────────────────────────────────────────────
def get_models(n_train: int) -> dict:
    m = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, class_weight="balanced", random_state=SEED),
        "Random Forest": RandomForestClassifier(200, class_weight="balanced", n_jobs=-1, random_state=SEED),
        "Bagging": BaggingClassifier(DecisionTreeClassifier(max_depth=8), n_estimators=50, n_jobs=-1, random_state=SEED),
        "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=SEED),
        "kNN": KNeighborsClassifier(7, n_jobs=-1),
        "Naive Bayes": GaussianNB(),
    }
    if n_train <= 10_000:
        m["SVM (RBF)"] = SVC(probability=True, class_weight="balanced", random_state=SEED)
    return m


def _scores(y, pred, proba, pos, n_classes):
    if n_classes == 2:
        out = dict(Accuracy=accuracy_score(y, pred),
                   Precision=precision_score(y, pred, pos_label=pos, zero_division=0),
                   Recall=recall_score(y, pred, pos_label=pos, zero_division=0),
                   F1=f1_score(y, pred, pos_label=pos, zero_division=0))
        out["PR-AUC"] = average_precision_score(y == pos, proba[:, pos]) if proba is not None else np.nan
    else:
        out = dict(Accuracy=accuracy_score(y, pred),
                   Precision=precision_score(y, pred, average="macro", zero_division=0),
                   Recall=recall_score(y, pred, average="macro", zero_division=0),
                   F1=f1_score(y, pred, average="macro", zero_division=0))
        try:
            out["PR-AUC"] = roc_auc_score(y, proba, multi_class="ovr", average="macro")  # ROC-AUC for multiclass
        except Exception:
            out["PR-AUC"] = np.nan
    return out


def train_and_evaluate(b: dict, do_cv: bool = False) -> tuple[pd.DataFrame, dict]:
    Xtr, Xte, ytr, yte = b["Xtr"], b["Xte"], b["ytr"], b["yte"]
    k = len(b["class_names"]); pos = b["pos"]
    rows, fitted = [], {}
    for name, clf in get_models(len(ytr)).items():
        try:
            clf.fit(Xtr, ytr)
            pred = clf.predict(Xte)
            proba = clf.predict_proba(Xte) if hasattr(clf, "predict_proba") else None
            r = _scores(yte, pred, proba, pos, k)
            if do_cv:
                scoring = "average_precision" if k == 2 else "f1_macro"
                y_cv = (ytr == pos).astype(int) if k == 2 else ytr
                cv = cross_validate(clf, Xtr, y_cv, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                    scoring=scoring, n_jobs=1)
                r["CV score (mean)"] = cv["test_score"].mean(); r["CV std"] = cv["test_score"].std()
            r["Model"] = name
            rows.append(r); fitted[name] = clf
        except Exception as e:
            rows.append({"Model": name, "Accuracy": np.nan, "Precision": np.nan, "Recall": np.nan,
                         "F1": np.nan, "PR-AUC": np.nan})
    df = pd.DataFrame(rows).set_index("Model").sort_values("F1", ascending=False)
    return df, fitted


def single_feature_check(b: dict, min_gain: float = 0.05) -> pd.DataFrame:
    """Shortcut detector: how well does ONE raw column alone predict the target on the test set?"""
    Xtr, Xte, ytr, yte = b["raw_tr"], b["raw_te"], b["ytr"], b["yte"]
    glob = np.bincount(ytr).argmax()
    rows = []
    for c in Xtr.columns:
        if c.endswith("_missing"):
            continue
        s_tr, s_te = Xtr[c], Xte[c]
        if pd.api.types.is_numeric_dtype(s_tr) and s_tr.nunique() > 20:
            try:
                bins = np.unique(np.quantile(s_tr, np.linspace(0, 1, 21)))
                s_tr = pd.cut(s_tr, bins, include_lowest=True).astype(str)
                s_te = pd.cut(s_te, bins, include_lowest=True).astype(str)
            except Exception:
                continue
        maj = pd.DataFrame({"c": s_tr.astype(str).values, "y": ytr}).groupby("c")["y"].agg(lambda v: v.value_counts().idxmax())
        pred = s_te.astype(str).map(maj).fillna(glob).astype(int).values
        rows.append({"Column": c, "Accuracy using only this column": accuracy_score(yte, pred),
                     "Gain over majority baseline": accuracy_score(yte, pred) - b["baseline_acc"]})
    out = pd.DataFrame(rows).sort_values("Gain over majority baseline", ascending=False)
    out["Flag"] = np.where(out["Gain over majority baseline"] > min_gain, "possible shortcut", "")
    return out.reset_index(drop=True)
