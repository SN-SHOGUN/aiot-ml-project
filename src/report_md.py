"""Markdown export of the session. PNG charts only when kaleido is installed."""
from __future__ import annotations

import base64

import pandas as pd


def _figure_md(fig) -> str:
    try:
        import kaleido  # noqa: F401
        raw = fig.to_image(format="png")
        encoded = base64.b64encode(raw).decode("ascii")
        return f"![chart](data:image/png;base64,{encoded})"
    except Exception:
        title = ""
        try:
            title = fig.layout.title.text or ""
        except Exception:
            title = ""
        traces = len(getattr(fig, "data", []) or [])
        return f"Chart spec: {title} (traces={traces})"


def _block(title: str, body: str, evidence: list | None) -> list[str]:
    lines = [f"## {title}", body or "", ""]
    for item in evidence or []:
        lines.append(f"### {item.get('name')} `{item.get('args')}`")
        if item.get("text"):
            lines.append(str(item["text"]))
        table = item.get("table")
        if isinstance(table, pd.DataFrame):
            lines.append(table.head(30).to_csv(index=False))
        if item.get("figure") is not None:
            lines.append(_figure_md(item["figure"]))
        lines.append("")
    return lines


def to_markdown(messages: list[dict], quick_log: list[dict] | None = None) -> str:
    """Questions, answers, tables, and charts from this session."""
    lines = ["# Ask Your Data report", ""]
    for msg in messages:
        title = "You" if msg.get("role") == "user" else "Assistant"
        lines.extend(_block(title, msg.get("content") or "", msg.get("evidence")))
    if quick_log:
        lines.append("## Quick actions (no API)")
        for item in quick_log:
            lines.extend(_block(item.get("name", "tool"), item.get("text") or "", [item]))
    return "\n".join(lines)
