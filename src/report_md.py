"""Markdown export of one session. PNG only if kaleido is installed."""
from __future__ import annotations

import base64


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
        return f"Chart spec: {title} traces={traces}"


def _block(lines: list[str], item: dict) -> None:
    lines.append(f"### {item.get('name')} `{item.get('args')}`")
    if item.get("text"):
        lines.append(str(item["text"]))
    table = item.get("table")
    if table is not None:
        lines.append(table.head(30).to_csv(index=False))
    if item.get("figure") is not None:
        lines.append(_figure_md(item["figure"]))


def to_markdown(messages: list[dict], quick_log: list[dict]) -> str:
    """Questions, answers, tables, and charts from this session."""
    lines = ["# Ask Your Data report", ""]
    for item in quick_log or []:
        lines.append("## Quick action (no API call)")
        _block(lines, item)
        lines.append("")
    for msg in messages or []:
        who = "You" if msg.get("role") == "user" else "Assistant"
        lines.append(f"## {who}")
        lines.append(str(msg.get("content") or ""))
        for ev in msg.get("evidence") or []:
            _block(lines, ev)
        if msg.get("role") == "assistant":
            lines.append(f"Tokens this answer: {msg.get('tokens', 0)}")
        lines.append("")
    return "\n".join(lines)
