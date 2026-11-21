"""Whitelisted data tools. Each returns table, figure, and short text.

Python computes every number. The model never receives the figure object.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.ensemble import IsolationForest

from src.generic import (
    quick_model_eval,
    single_feature_check,
    suggest_drop_cols,
    suggest_target,
    train_and_evaluate,
)


def _ok(table=None, figure=None, text: str = "") -> dict:
    return {"table": table, "figure": figure, "text": text}


def _err(text: str) -> dict:
    return _ok(text=text)


def _missing(df: pd.DataFrame, cols: list) -> str | None:
    bad = [c for c in cols if c not in df.columns]
    if bad:
        return f"Unknown columns {bad}. Available columns: {list(map(str, df.columns))}"
    return None


def _as_list(value):
    if value is None or value == "":
        return None
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return list(value)


def _as_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def profile(df: pd.DataFrame) -> dict:
    """Shape, types, missing percent, duplicates, and obvious data issues."""
    rows, issues = [], []
    n = max(len(df), 1)
    for col in df.columns:
        series = df[col]
        missing = round(100 * float(series.isna().mean()), 2)
        unique = int(series.nunique(dropna=True))
        rows.append({"column": str(col), "dtype": str(series.dtype), "missing_pct": missing, "unique": unique})
        if unique <= 1:
            issues.append(f"{col} is constant")
        elif n > 10 and unique / n > 0.95:
            issues.append(f"{col} looks like an ID")
        elif missing > 30:
            issues.append(f"{col} is {missing}% missing")
    text = (
        f"shape={df.shape[0]}x{df.shape[1]}; duplicate_rows={int(df.duplicated().sum())}; "
        f"issues={'; '.join(issues) if issues else 'none'}"
    )
    return _ok(pd.DataFrame(rows), text=text)


def summarize(df: pd.DataFrame, column: str) -> dict:
    """describe() for a numeric column, otherwise the top value counts."""
    err = _missing(df, [column])
    if err:
        return _err(err)
    series = df[column]
    if pd.api.types.is_numeric_dtype(series):
        table = series.describe().rename("value").rename_axis("stat").reset_index()
        text = f"{column}: mean={series.mean():.4g}, missing={int(series.isna().sum())}"
    else:
        counts = series.astype(str).value_counts(dropna=False).head(15)
        table = counts.rename_axis(column).reset_index(name="count")
        text = f"{column}: {int(series.nunique(dropna=True))} unique values"
    return _ok(table, text=text)


def group_aggregate(df: pd.DataFrame, group_by: str, metric: str, agg: str = "mean", top_n: int = 15) -> dict:
    """Aggregate a metric by a group and draw a bar chart."""
    err = _missing(df, [group_by, metric])
    if err:
        return _err(err)
    allowed = {"mean", "sum", "count", "median", "min", "max"}
    if agg not in allowed:
        return _err(f"agg must be one of {sorted(allowed)}")
    if agg != "count" and not pd.api.types.is_numeric_dtype(df[metric]):
        return _err(f"{metric} is not numeric, so {agg} is not valid.")
    top_n = max(int(top_n or 15), 1)
    grouped = getattr(df.groupby(group_by, dropna=False)[metric], agg)()
    table = grouped.sort_values(ascending=False).head(top_n).reset_index()
    fig = px.bar(table, x=str(group_by), y=str(metric), title=f"{agg} of {metric} by {group_by}")
    fig.update_layout(height=420)
    return _ok(table, fig, text=f"{agg} of {metric} by {group_by}")


def correlate(df: pd.DataFrame, columns: list[str] | None = None) -> dict:
    """Correlation heatmap and the strongest numeric pairs."""
    chosen = _as_list(columns)
    if chosen and _missing(df, chosen):
        return _err(_missing(df, chosen))
    frame = df if not chosen else df[chosen]
    nums = frame.select_dtypes(include="number")
    if nums.shape[1] < 2:
        return _err("Need at least two numeric columns to correlate.")
    corr = nums.corr(numeric_only=True)
    pairs = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().reset_index()
    pairs.columns = ["left", "right", "correlation"]
    pairs["abs"] = pairs["correlation"].abs()
    top = pairs.sort_values("abs", ascending=False).head(8).drop(columns="abs")
    fig = px.imshow(corr.round(3), text_auto=True, title="Correlation", aspect="auto")
    fig.update_layout(height=480)
    strongest = "; ".join(f"{a}~{b}={c:.2f}" for a, b, c in top.head(3).to_numpy())
    return _ok(top.round(4), fig, text=f"strongest pairs: {strongest or 'none'}")


def plot(df: pd.DataFrame, kind: str, x: str, y: str | None = None, color: str | None = None) -> dict:
    """One chart: hist, box, scatter, line, or bar."""
    allowed = {"hist", "box", "scatter", "line", "bar"}
    if kind not in allowed:
        return _err(f"kind must be one of {sorted(allowed)}")
    needed = [c for c in (x, y, color) if c]
    err = _missing(df, needed)
    if err:
        return _err(err)
    if kind in {"scatter", "line"} and not y:
        return _err(f"{kind} needs both x and y.")
    view = df.sample(5000, random_state=42) if len(df) > 5000 else df
    if kind == "hist":
        fig = px.histogram(view, x=x, color=color, title=f"Histogram of {x}")
    elif kind == "box":
        fig = px.box(view, x=x, y=y, color=color, title=f"Box plot of {x}")
    elif kind == "scatter":
        fig = px.scatter(view, x=x, y=y, color=color, title=f"{y} vs {x}")
    elif kind == "line":
        fig = px.line(view, x=x, y=y, color=color, title=f"{y} over {x}")
    elif y:
        fig = px.bar(view, x=x, y=y, color=color, title=f"{y} by {x}")
    else:
        counts = view[x].astype(str).value_counts().head(20).rename_axis(x).reset_index(name="count")
        fig = px.bar(counts, x=x, y="count", title=f"Counts of {x}")
    fig.update_layout(height=420)
    return _ok(None, fig, text=f"{kind} chart of {x}" + (f" vs {y}" if y else ""))


def detect_anomalies(df: pd.DataFrame, column: str | None = None) -> dict:
    """IsolationForest when there are enough rows, otherwise an IQR fence."""
    if column and _missing(df, [column]):
        return _err(_missing(df, [column]))
    nums = df[[column]].select_dtypes(include="number") if column else df.select_dtypes(include="number")
    if nums.shape[1] == 0:
        return _err("No numeric column is available for anomaly detection.")
    filled = nums.fillna(nums.median(numeric_only=True))
    flags = np.zeros(len(filled), dtype=bool)
    if len(filled) >= 30:
        model = IsolationForest(random_state=42, contamination=0.08, n_estimators=40)
        flags = model.fit_predict(filled) == -1
        method = "IsolationForest"
    else:
        method = "IQR"
        for col in filled.columns:
            q1, q3 = filled[col].quantile(0.25), filled[col].quantile(0.75)
            fence = 1.5 * (q3 - q1)
            flags = flags | ((filled[col] < q1 - fence) | (filled[col] > q3 + fence)).to_numpy()
    count = int(flags.sum())
    top = df.loc[flags].head(15).copy()
    top.insert(0, "anomaly", True)
    chart_col = column or str(filled.columns[0])
    plot_df = pd.DataFrame({chart_col: filled[chart_col], "anomaly": np.where(flags, "anomaly", "typical")})
    fig = px.box(plot_df, y=chart_col, color="anomaly", title=f"Anomalies in {chart_col} ({method})")
    fig.update_layout(height=420)
    return _ok(top, fig, text=f"{method} flagged {count} of {len(df)} rows")


def trend(df: pd.DataFrame, time_column: str, value_column: str, freq: str = "D") -> dict:
    """Resample a numeric value on a datetime column and draw a line."""
    err = _missing(df, [time_column, value_column])
    if err:
        return _err(err)
    if not pd.api.types.is_numeric_dtype(df[value_column]):
        return _err(f"{value_column} is not numeric.")
    stamps = pd.to_datetime(df[time_column], errors="coerce")
    if float(stamps.notna().mean()) < 0.8:
        return _err(f"{time_column} does not parse as datetime.")
    rule = {"D": "D", "W": "W", "M": "ME", "H": "h", "ME": "ME"}.get(str(freq).upper())
    if rule is None:
        return _err("freq must be one of D, W, M, H.")
    frame = pd.DataFrame({"time": stamps, value_column: df[value_column]}).dropna()
    table = frame.set_index("time").resample(rule)[value_column].mean().dropna().reset_index()
    if table.empty:
        return _err("Resample produced no rows. Try another freq.")
    fig = px.line(table, x="time", y=value_column, title=f"{value_column} by {freq}")
    fig.update_layout(height=420)
    return _ok(table, fig, text=f"resampled {value_column} by {freq}: {len(table)} points")


def _confusion_figure(matrix, labels):
    fig = px.imshow(
        matrix,
        x=[str(x) for x in labels],
        y=[str(x) for x in labels],
        text_auto=True,
        labels={"x": "Predicted", "y": "Actual"},
        title="Confusion matrix (best model)",
        aspect="auto",
    )
    fig.update_layout(height=420)
    return fig


def train_models(
    df: pd.DataFrame,
    target: str,
    drop_cols: list[str] | None = None,
    group_col: str | None = None,
    cv: bool = False,
) -> dict:
    """Train the shared models. Returns metrics, the majority baseline, and shortcut checks."""
    try:
        trained = train_and_evaluate(
            df, target, drop_cols=_as_list(drop_cols), group_col=group_col or None, cv=_as_bool(cv),
        )
        checks = single_feature_check(df, target)
    except (ValueError, TypeError) as exc:
        return _err(str(exc))
    names = []
    if not checks.empty:
        names = checks.loc[checks["shortcut"], "feature"].astype(str).tolist()
    text = (
        f"majority_baseline_accuracy={trained['baseline_accuracy']:.4f}; best={trained['best_name']}; "
        f"SHORTCUT WARNING columns={names or ['none']}. "
        "Compare recall and pr_auc with the baseline, not accuracy alone."
    )
    return _ok(trained["comparison"].round(4), _confusion_figure(trained["confusion"], trained["labels"]), text=text)


def quick_model(df: pd.DataFrame, target: str) -> dict:
    """Fast logistic regression and random forest. Python writes the verdict; the model only rephrases it."""
    try:
        trained = quick_model_eval(df, target)
    except (ValueError, TypeError) as exc:
        return _err(str(exc))
    table = trained["comparison"][["model", "accuracy", "recall", "f1"]].round(4)
    base = float(trained["baseline_accuracy"])
    chart = pd.concat([
        pd.DataFrame([{"model": "Majority baseline", "accuracy": round(base, 4)}]),
        table[["model", "accuracy"]],
    ], ignore_index=True)
    fig = px.bar(chart, x="model", y="accuracy", title="Accuracy versus majority baseline")
    fig.update_layout(height=400)
    return _ok(table, fig, text=f"baseline_accuracy={base:.4f}. {trained['verdict']}.")


def data_quality_report(df: pd.DataFrame) -> dict:
    """Duplicates, missingness, outlier columns, and leakage suspects."""
    health = profile(df)
    target = suggest_target(df)
    drops = suggest_drop_cols(df, target)
    checks = single_feature_check(df, target) if target else pd.DataFrame()
    shortcuts = []
    if not checks.empty:
        shortcuts = checks.loc[checks["shortcut"], "feature"].astype(str).tolist()
    outlier_cols = 0
    for col in df.select_dtypes(include="number").columns:
        series = df[col].dropna()
        if series.empty:
            continue
        q1, q3 = series.quantile(0.25), series.quantile(0.75)
        fence = 1.5 * (q3 - q1)
        if ((series < q1 - fence) | (series > q3 + fence)).any():
            outlier_cols += 1
    text = (
        f"{health['text']}; suggested_drop={drops or ['none']}; likely_target={target}; "
        f"numeric_columns_with_iqr_outliers={outlier_cols}; "
        f"SHORTCUT WARNING columns={shortcuts or ['none']}"
    )
    return _ok(checks if not checks.empty else health["table"], text=text)


REGISTRY = {
    "profile": profile,
    "summarize": summarize,
    "group_aggregate": group_aggregate,
    "correlate": correlate,
    "plot": plot,
    "detect_anomalies": detect_anomalies,
    "trend": trend,
    "train_models": train_models,
    "quick_model": quick_model,
    "data_quality_report": data_quality_report,
}


def df_token(df: pd.DataFrame) -> str:
    """Cheap dataset identity for the tool cache."""
    sample = df.head(40).astype(str)
    digest = int(pd.util.hash_pandas_object(sample, index=False).sum())
    return f"{df.shape[0]}x{df.shape[1]}:{digest}"


def run_tool(df: pd.DataFrame, name: str, args: dict | None, cache: dict | None = None) -> dict:
    """Run one whitelisted tool. Identical calls on the same table reuse the cache."""
    clean = {k: v for k, v in (args or {}).items() if v is not None}
    if name not in REGISTRY:
        return _err(f"Unknown tool {name}. Options: {list(REGISTRY)}")
    key = None
    if cache is not None:
        key = json.dumps({"t": name, "a": clean, "d": df_token(df)}, sort_keys=True, default=str)
        if key in cache:
            return cache[key]
    try:
        out = REGISTRY[name](df, **clean)
    except TypeError as exc:
        out = _err(f"Bad arguments for {name}: {exc}")
    if not isinstance(out, dict):
        out = _err(str(out))
    out.setdefault("table", None)
    out.setdefault("figure", None)
    out.setdefault("text", "")
    if cache is not None and key is not None:
        cache[key] = out
    return out
