"""Ask Your Data tab. Numbers come from tools; Gemini only explains them."""
from __future__ import annotations

import inspect

import pandas as pd
import streamlit as st

from src.generic import suggest_target
from src.ingest import limit_rows, load_example, parse_text, read_uploaded
from src.llm import ask_question, make_client, schema_text
from src.modes import MODES, PRIVACY_NOTE, questions_for, resolve_model
from src.report_md import to_markdown
from src.tools import profile, run_tool


def _kw(widget) -> dict:
    """Full-width argument that matches the installed Streamlit."""
    if "width" in inspect.signature(widget).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}


def _init() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("quick_log", [])
    st.session_state.setdefault("tool_cache", {})
    st.session_state.setdefault("token_total", 0)
    st.session_state.setdefault("token_budget", 50_000)
    st.session_state.setdefault("ask_df", None)
    st.session_state.setdefault("ask_note", "")


def _api_key() -> str:
    typed = (st.session_state.get("gemini_api_key_input") or "").strip()
    if typed:
        return typed
    try:
        return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        return ""


def _store(df: pd.DataFrame, note: str = "") -> None:
    frame, warn = limit_rows(df)
    st.session_state["ask_df"] = frame
    st.session_state["ask_note"] = " ".join(part for part in (warn, note) if part)
    st.session_state["tool_cache"] = {}
    st.session_state["messages"] = []
    st.session_state["quick_log"] = []


def _pairs(messages: list[dict]) -> list[dict]:
    pairs, pending = [], None
    for msg in messages:
        if msg.get("role") == "user":
            pending = msg.get("content")
        elif msg.get("role") == "assistant" and pending:
            pairs.append({"q": pending, "a": msg.get("content") or ""})
            pending = None
    return pairs


def _show_evidence(items: list[dict]) -> None:
    for ev in items:
        if ev.get("figure") is not None:
            st.plotly_chart(ev["figure"], **_kw(st.plotly_chart))
    if not items:
        return
    with st.expander("Evidence"):
        for ev in items:
            st.markdown(f"**{ev.get('name')}** `{ev.get('args')}`")
            if ev.get("text"):
                st.write(ev["text"])
            if ev.get("table") is not None:
                st.dataframe(ev["table"], **_kw(st.dataframe))


def _handle(question: str, mode_name: str) -> None:
    messages = st.session_state["messages"]
    messages.append({"role": "user", "content": question})
    df = st.session_state.get("ask_df")
    if df is None:
        messages.append({"role": "assistant", "content": "Load a dataset first. I have nothing to measure.", "evidence": [], "tokens": 0})
        return
    key = _api_key()
    if not key:
        messages.append({
            "role": "assistant",
            "content": "Add a Gemini API key in the sidebar, or set GEMINI_API_KEY in .streamlit/secrets.toml.",
            "evidence": [],
            "tokens": 0,
        })
        return
    mode = MODES[mode_name]
    send_samples = bool(st.session_state.get("send_samples")) if mode_name == "researcher" else False

    def execute(name, args):
        if name == "train_models" and st.session_state.get("use_cv"):
            args = {**args, "cv": True}
        return run_tool(df, name, args, st.session_state["tool_cache"])

    try:
        client = make_client(key)
    except Exception as exc:
        from src.llm import friendly_error
        messages.append({"role": "assistant", "content": friendly_error(exc), "evidence": [], "tokens": 0})
        return
    result = ask_question(
        client,
        question,
        schema=schema_text(df, send_samples=send_samples),
        history=_pairs(messages[:-1]),
        mode=mode,
        model=resolve_model(mode_name, bool(st.session_state.get("deeper_model"))),
        execute=execute,
    )
    st.session_state["token_total"] = int(st.session_state["token_total"]) + int(result.tokens)
    messages.append({
        "role": "assistant",
        "content": result.error or result.text,
        "evidence": result.evidence,
        "tokens": result.tokens,
    })


def _quick(name: str, args: dict | None = None) -> None:
    df = st.session_state.get("ask_df")
    if df is None:
        st.session_state["quick_log"].append({
            "name": name, "args": args or {}, "table": None, "figure": None, "text": "Load a dataset first.",
        })
        return
    if name == "class_balance":
        target = suggest_target(df)
        if not target:
            result = {"table": None, "figure": None, "text": "No obvious class column to balance."}
            result.update(name="summarize", args={})
        else:
            result = run_tool(df, "summarize", {"column": target}, st.session_state["tool_cache"])
            result.update(name="summarize", args={"column": target})
    else:
        result = run_tool(df, name, args or {}, st.session_state["tool_cache"])
        result.update(name=name, args=args or {})
    st.session_state["quick_log"].append(result)


def render_sidebar() -> None:
    """API key, mode, budget, and the privacy note. Key stays in session state."""
    _init()
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Ask Your Data")
    mode = st.sidebar.radio(
        "Mode",
        ["student", "researcher"],
        format_func=lambda key: MODES[key]["label"],
        key="mode_radio",
    )
    if st.session_state.get("_mode_applied") != mode:
        if "_mode_applied" in st.session_state:
            st.session_state["messages"] = []
            st.session_state["quick_log"] = []
        st.session_state["_mode_applied"] = mode
    st.sidebar.caption(MODES[mode]["caption"])
    st.sidebar.text_input("Gemini API key", type="password", key="gemini_api_key_input")
    if mode == "researcher":
        st.sidebar.checkbox("Use deeper model", key="deeper_model", help="Pro-tier model. It costs more tokens.")
        st.sidebar.checkbox("Send 5 sample rows", key="send_samples")
        st.sidebar.checkbox("Cross-validate train_models", key="use_cv")
    st.sidebar.number_input("Session token budget", min_value=1000, max_value=2_000_000, step=1000, key="token_budget")
    st.sidebar.caption(f"Session total: {int(st.session_state['token_total'])} tokens")
    if st.sidebar.button("Clear chat", key="clear_chat"):
        st.session_state["messages"] = []
        st.session_state["quick_log"] = []
    st.sidebar.info(PRIVACY_NOTE)


def render_ask() -> None:
    """Upload, quick actions, and the chat. Quick actions do not call Gemini."""
    _init()
    mode = st.session_state.get("mode_radio", "student")
    st.header("Ask Your Data")
    st.caption("Python measures the table. Gemini only explains those measurements.")
    uploaded = st.file_uploader("Upload a table", type=["csv", "tsv", "txt", "xlsx", "json"], key="ask_upload")
    paste = st.text_area("Paste data here", height=120, key="ask_paste", placeholder="col_a,col_b\n1,2")
    c1, c2, c3 = st.columns(3)
    if c1.button("Use uploaded file", key="use_upload") and uploaded is not None:
        try:
            _store(read_uploaded(uploaded))
        except Exception as exc:
            st.error(f"Could not read that file: {exc}")
    if c2.button("Use pasted data", key="use_paste"):
        try:
            _store(parse_text(paste))
        except Exception as exc:
            st.error(str(exc))
    if c3.button("Load example", key="load_example"):
        try:
            _store(load_example(), "Example: AI4I 2020.")
        except Exception as exc:
            st.error(str(exc))
    samples = mode == "researcher" and bool(st.session_state.get("send_samples"))
    st.info(PRIVACY_NOTE + (" Sample rows are included." if samples else " Sample rows are not sent."))
    df = st.session_state.get("ask_df")
    if isinstance(df, pd.DataFrame):
        if st.session_state.get("ask_note"):
            st.warning(st.session_state["ask_note"])
        st.caption(f"{df.shape[0]:,} rows × {df.shape[1]} columns")
        st.dataframe(df.head(5), **_kw(st.dataframe))
        health = profile(df)
        st.markdown("**Data health**")
        st.write(health["text"])
    else:
        st.info("Upload a file, paste a table, or load the example to start.")
    st.markdown("**Quick actions** — no API call")
    q1, q2, q3, q4 = st.columns(4)
    clicked = None
    if q1.button("Data health", key="qa_health"):
        clicked = ("profile", {})
    if q2.button("Class balance", key="qa_balance"):
        clicked = ("class_balance", {})
    if q3.button("Correlation heatmap", key="qa_corr"):
        clicked = ("correlate", {})
    if q4.button("Missing values", key="qa_missing"):
        clicked = ("profile", {})
    if clicked:
        _quick(*clicked)
        st.rerun()
    total = int(st.session_state["token_total"])
    over = total >= int(st.session_state["token_budget"])
    for item in st.session_state["quick_log"]:
        with st.chat_message("assistant"):
            st.markdown(item.get("text") or item.get("name"))
            _show_evidence([item])
            st.caption(f"this answer: 0 tokens | session total: {total} tokens | quick action, no API call")
    for msg in st.session_state["messages"]:
        with st.chat_message(msg["role"]):
            st.markdown(msg.get("content") or "")
            if msg["role"] == "assistant":
                _show_evidence(msg.get("evidence") or [])
                st.caption(f"this answer: {msg.get('tokens', 0)} tokens | session total: {total} tokens")
    if mode == "researcher" and (st.session_state["messages"] or st.session_state["quick_log"]):
        st.download_button(
            "Export report",
            data=to_markdown(st.session_state["messages"], st.session_state["quick_log"]),
            file_name="ask_report.md",
            mime="text/markdown",
            key="export_report",
        )
    st.markdown("**Try a question**")
    asked = None
    for i, question in enumerate(questions_for(mode, df if isinstance(df, pd.DataFrame) else None)):
        if st.button(question, key=f"sug_{mode}_{i}", disabled=over):
            asked = question
    if over:
        st.warning("Session token budget reached. Quick actions still work. Raise the budget in the sidebar to ask more.")
    try:
        typed = st.chat_input("Ask about your data", disabled=over, key="ask_input")
    except TypeError:
        typed = None if over else st.chat_input("Ask about your data", key="ask_input")
    if asked or typed:
        _handle(asked or typed, mode)
        st.rerun()
