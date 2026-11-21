"""Train on the table loaded in Ask Your Data. No second copy of the pipeline."""
from __future__ import annotations

import inspect

import pandas as pd
import streamlit as st

from src.generic import suggest_target
from src.tools import run_tool


def _kw(widget) -> dict:
    if "width" in inspect.signature(widget).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


def render_upload() -> None:
    """Local training on the shared session table."""
    st.header("Upload & Analyze")
    st.caption("Uses the table from Ask Your Data. Training stays on this machine.")
    df = st.session_state.get("ask_df")
    if not isinstance(df, pd.DataFrame):
        st.info("Load a file in Ask Your Data. The same table is reused here.")
        return
    st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
    guess = suggest_target(df)
    index = list(df.columns).index(guess) if guess in df.columns else 0
    target = st.selectbox("Target", list(df.columns), index=index, key="upload_target")
    if st.button("Train models", key="upload_train"):
        st.session_state.setdefault("tool_cache", {})
        with st.spinner("Training…"):
            st.session_state["upload_result"] = run_tool(
                df, "train_models", {"target": target}, st.session_state["tool_cache"],
            )
    result = st.session_state.get("upload_result")
    if not result:
        return
    st.write(result.get("text") or "")
    if result.get("table") is not None:
        st.dataframe(result["table"], **_kw(st.dataframe))
    if result.get("figure") is not None:
        st.plotly_chart(result["figure"], **_kw(st.plotly_chart))
