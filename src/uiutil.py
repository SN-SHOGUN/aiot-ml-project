"""Widget width kwargs that match the installed Streamlit."""
from __future__ import annotations

import inspect


def full_width(widget) -> dict:
    if "width" in inspect.signature(widget).parameters:
        return {"width": "stretch"}
    return {"use_container_width": True}
