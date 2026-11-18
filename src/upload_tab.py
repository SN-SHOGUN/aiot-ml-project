"""Streamlit tab: upload ANY csv -> choose target -> leakage-aware pipeline -> compare models."""
from __future__ import annotations
import pathlib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from sklearn.metrics import confusion_matrix, precision_recall_curve

from src.generic import (build_dataset, train_and_evaluate, single_feature_check,
                         suggest_drop_cols, suggest_target, MAX_ROWS)

SAMPLE = pathlib.Path(__file__).parent.parent / "data" / "ai4i2020.csv"


def render():
    st.header("📁 Upload & Analyze your own dataset")
    st.caption("Any CSV. Pick the target, the app builds a leakage-aware pipeline and compares models.")

    up = st.file_uploader("Upload a CSV", type=["csv"], key="up_csv")
    use_sample = st.checkbox("…or use the bundled AI4I sample", value=False, key="up_sample")
    if up is None and not use_sample:
        st.info("Upload a CSV to begin.")
        return
    try:
        df = pd.read_csv(up) if up is not None else pd.read_csv(SAMPLE)
    except Exception as e:
        st.error(f"Could not read the file as CSV: {e}")
        return
    if df.shape[1] < 2 or len(df) < 50:
        st.error("Need at least 2 columns and 50 rows.")
        return
    if len(df) > MAX_ROWS:
        st.warning(f"{len(df):,} rows - a random sample of {MAX_ROWS:,} will be used (cloud memory limit).")

    st.subheader("1 · Data overview")
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}"); c2.metric("Columns", df.shape[1]); c3.metric("Duplicate rows", int(df.duplicated().sum()))
    st.dataframe(df.head(10), use_container_width=True)
    info = pd.DataFrame({"dtype": df.dtypes.astype(str), "unique": df.nunique(), "missing %": (df.isna().mean() * 100).round(1)})
    st.dataframe(info, use_container_width=True)

    st.subheader("2 · Configure")
    cols = list(df.columns)
    target = st.selectbox("Target column (what to predict)", cols, index=cols.index(suggest_target(df)), key="up_target")
    vc = df[target].astype(str).value_counts(normalize=True)
    if df[target].nunique() > 30:
        st.error("This target has more than 30 distinct values - it looks like a number to regress, not classes. Pick another column.")
        return
    st.plotly_chart(px.bar(vc.reset_index(), x=target, y="proportion", title="Target class distribution"), use_container_width=True)
    st.caption(f"Majority-class baseline accuracy: **{vc.max():.1%}** - a model must beat this to have learned anything.")

    sugg = suggest_drop_cols(df, target)
    drop = st.multiselect("Columns to drop (IDs, leaky or useless) - suggestions pre-filled", [c for c in cols if c != target],
                          default=sugg, key="up_drop")
    gcol = st.selectbox("Group column (optional): keeps the same entity out of both train and test",
                        ["(none)"] + [c for c in cols if c != target], key="up_group")
    gcol = None if gcol == "(none)" else gcol
    c1, c2, c3 = st.columns(3)
    test_size = c1.slider("Test size", 0.1, 0.4, 0.2, 0.05, key="up_ts")
    top_n = c2.slider("Keep top-N categories per text column", 5, 100, 30, key="up_topn")
    parse = c3.checkbox("Split long text like 'Academy Award for Best Actor' into body", value=False, key="up_parse")
    do_cv = st.checkbox("Also run 5-fold cross-validation (slower, shows stability)", value=False, key="up_cv")

    if not st.button("🚀 Build pipeline & train all models", type="primary", key="up_go"):
        return
    try:
        with st.spinner("Preprocessing..."):
            b = build_dataset(df, target, drop, gcol, test_size, top_n, parse)
    except Exception as e:
        st.error(f"Preprocessing failed: {e}")
        return
    binary = len(b["class_names"]) == 2
    st.success(f"{len(b['features'])} features · train {len(b['ytr']):,} / test {len(b['yte']):,}"
               + (" · group-aware split" if b["group_split"] else "")
               + (f" · positive class = '{b['class_names'][b['pos']]}' (minority)" if binary else " · multiclass, macro metrics"))

    st.subheader("3 · Shortcut / leakage check")
    chk = single_feature_check(b)
    st.dataframe(chk.style.format({"Accuracy using only this column": "{:.3f}", "Gain over majority baseline": "{:+.3f}"}),
                 use_container_width=True)
    if (chk["Flag"] != "").any():
        st.warning("A single column predicts the target well above the baseline. Check whether it is known at prediction time "
                   "or just reflects how the data was collected. Re-run without it and compare.")

    st.subheader("4 · Model comparison")
    with st.spinner("Training models..."):
        res, fitted = train_and_evaluate(b, do_cv)
    base = pd.DataFrame({"Accuracy": [b["baseline_acc"]]}, index=["(majority-class baseline)"])
    st.dataframe(pd.concat([res, base]).style.format("{:.3f}", na_rep="-"), use_container_width=True)
    st.caption("Sorted by F1" + (" on the minority class" if binary else " (macro)") + ". Accuracy alone can mislead on imbalanced data. "
               + ("PR-AUC is the area under the precision-recall curve." if binary else "The last metric column is macro ROC-AUC."))
    long = res.reset_index().melt(id_vars="Model", value_vars=["Precision", "Recall", "F1"], var_name="Metric")
    st.plotly_chart(px.bar(long, x="Model", y="value", color="Metric", barmode="group", title="Precision / Recall / F1"),
                    use_container_width=True)

    best = res["F1"].idxmax()
    sel = st.selectbox("Inspect model", list(fitted), index=list(fitted).index(best), key="up_inspect")
    clf = fitted[sel]
    pred = clf.predict(b["Xte"])
    cm = confusion_matrix(b["yte"], pred)
    st.plotly_chart(px.imshow(cm, text_auto=True, x=b["class_names"], y=b["class_names"], color_continuous_scale="Blues",
                              labels=dict(x="Predicted", y="Actual"), title=f"Confusion matrix - {sel}"), use_container_width=True)
    if binary and hasattr(clf, "predict_proba"):
        p, r, _ = precision_recall_curve(b["yte"] == b["pos"], clf.predict_proba(b["Xte"])[:, b["pos"]])
        st.plotly_chart(px.line(x=r, y=p, labels=dict(x="Recall", y="Precision"), title=f"Precision-recall curve - {sel}"),
                        use_container_width=True)
    rf = fitted.get("Random Forest")
    if rf is not None:
        imp = pd.Series(rf.feature_importances_, index=b["features"]).sort_values().tail(15)
        st.plotly_chart(px.bar(imp, orientation="h", title="Random Forest - top 15 feature importances",
                               labels=dict(value="importance", index="")), use_container_width=True)
