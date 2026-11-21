"""Gemini tool loop. The model chooses a tool; Python runs it. No exec or eval.

Schema text belongs in the system instruction, not in every user turn.
"""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass, field

import pandas as pd

from src.modes import MODES

MAX_RESULT_CHARS = 1200
MAX_RESULT_ROWS = 15

try:
    from google.genai import types as _types
except ImportError:  # tests can run before the SDK is installed
    class _Part:
        def __init__(self, text=None, function_response=None):
            self.text = text
            self.function_response = function_response

        @staticmethod
        def from_text(text):
            return _Part(text=text)

        @staticmethod
        def from_function_response(name, response):
            return _Part(function_response={"name": name, "response": response})

    class _Content:
        def __init__(self, role, parts):
            self.role = role
            self.parts = parts

    class _Fn:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class _types:
        Part = _Part
        Content = _Content
        FunctionDeclaration = _Fn
        Tool = _Fn
        GenerateContentConfig = _Fn
        AutomaticFunctionCallingConfig = _Fn
        HttpOptions = _Fn


@dataclass
class AskResult:
    text: str
    evidence: list = field(default_factory=list)
    tokens: int = 0
    error: str | None = None


def friendly_error(exc: Exception) -> str:
    """Map API failures to a short message. Never include the key."""
    msg = str(exc).lower()
    if any(token in msg for token in ("429", "quota", "resource_exhausted", "rate limit", "rate-limit")):
        return (
            "Gemini is rate-limiting this key (quota or 429). Wait a minute and try again. "
            "Your table is still loaded."
        )
    if any(token in msg for token in ("api key", "api_key", "unauthorized", "permission", "401", "403")):
        return (
            "Gemini rejected the API key. Paste a valid key in the sidebar, or set GEMINI_API_KEY "
            "in .streamlit/secrets.toml. The key stays in this session only."
        )
    if any(token in msg for token in ("timeout", "timed out", "connection", "network", "unavailable", "503")):
        return "Could not reach Gemini (network or timeout). Check your connection and try again."
    return "Gemini did not return a usable answer. Try a simpler question."


def _retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(token in msg for token in ("429", "quota", "resource_exhausted", "rate limit", "rate-limit"))


def format_tool_result(result: dict) -> str:
    """Compact CSV for the model: at most 15 rows and 1,200 characters. No figure."""
    chunks: list[str] = []
    if result.get("text"):
        chunks.append(str(result["text"]))
    table = result.get("table")
    if isinstance(table, pd.DataFrame):
        chunks.append(table.head(MAX_RESULT_ROWS).to_csv(index=False))
    raw = "\n".join(chunks).strip() or "No result."
    if len(raw) > MAX_RESULT_CHARS:
        raw = raw[: MAX_RESULT_CHARS - 3] + "..."
    return raw


def _schema_for(annotation) -> dict:
    text = str(annotation)
    if annotation is int or text == "int":
        return {"type": "integer"}
    if annotation is float or text == "float":
        return {"type": "number"}
    if annotation is bool or text == "bool":
        return {"type": "boolean"}
    if "list" in text.lower():
        return {"type": "array", "items": {"type": "string"}}
    return {"type": "string"}


def tool_declarations(names: list[str]):
    """Build Gemini declarations from the tool functions. One registry, no second list of schemas."""
    from src.tools import REGISTRY

    decls = []
    for name in names:
        fn = REGISTRY[name]
        props = {}
        required = []
        doc = inspect.getdoc(fn) or name
        for pname, param in inspect.signature(fn).parameters.items():
            if pname == "df":
                continue
            props[pname] = _schema_for(param.annotation)
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        decls.append(_types.FunctionDeclaration(
            name=name,
            description=doc.split("\n")[0][:180],
            parameters_json_schema={"type": "object", "properties": props, "required": required},
        ))
    return decls


def _token_count(response) -> int:
    meta = getattr(response, "usage_metadata", None)
    if meta is None:
        return 0
    if isinstance(meta, dict):
        total = meta.get("total_token_count")
        prompt = meta.get("prompt_token_count") or 0
        cand = meta.get("candidates_token_count") or 0
    else:
        total = getattr(meta, "total_token_count", None)
        prompt = getattr(meta, "prompt_token_count", 0) or 0
        cand = getattr(meta, "candidates_token_count", 0) or 0
    if total:
        return int(total)
    return int(prompt) + int(cand)


def _args(call) -> dict:
    raw = getattr(call, "args", {}) or {}
    if isinstance(raw, dict):
        return dict(raw)
    return {key: raw[key] for key in raw}


def _history_contents(history: list[dict], turns: int, question: str):
    contents = []
    for turn in history[-turns:]:
        contents.append(_types.Content(role="user", parts=[_types.Part.from_text(text=turn["q"])]))
        contents.append(_types.Content(role="model", parts=[_types.Part.from_text(text=turn["a"])]))
    contents.append(_types.Content(role="user", parts=[_types.Part.from_text(text=question)]))
    return contents


def _generate(client, payload: dict):
    try:
        return client.models.generate_content(**payload)
    except Exception as exc:
        if not _retryable(exc):
            raise
        time.sleep(1)
        return client.models.generate_content(**payload)


def ask_question(client, question: str, *, schema: str, history: list[dict], mode: str, model: str, execute) -> AskResult:
    """Call Gemini until it answers or the mode's tool cap is hit."""
    cfg = MODES[mode]
    system = cfg["system"] + "\n\nDATA SCHEMA (computed locally; do not invent columns):\n" + schema
    contents = _history_contents(history, cfg["history_turns"], question)
    config = _types.GenerateContentConfig(
        system_instruction=system,
        tools=[_types.Tool(function_declarations=tool_declarations(cfg["tools"]))],
        max_output_tokens=cfg["max_output_tokens"],
        temperature=0.2,
        automatic_function_calling=_types.AutomaticFunctionCallingConfig(disable=True),
    )
    evidence: list[dict] = []
    tokens = 0
    calls = 0
    try:
        while calls < cfg["max_tool_calls"]:
            response = _generate(client, {"model": model, "contents": contents, "config": config})
            tokens += _token_count(response)
            function_calls = list(getattr(response, "function_calls", None) or [])
            if not function_calls:
                text = (getattr(response, "text", None) or "").strip()
                if not text:
                    return AskResult(
                        "Gemini returned an empty response. Try rephrasing the question.",
                        evidence, tokens, error="Gemini returned an empty response. Try rephrasing the question.",
                    )
                return AskResult(text, evidence, tokens)
            candidate = getattr(response, "candidates", None)
            if candidate:
                contents.append(candidate[0].content)
            for call in function_calls:
                if calls >= cfg["max_tool_calls"]:
                    break
                name = call.name
                args = _args(call)
                if name not in cfg["tools"]:
                    result = {"table": None, "figure": None, "text": f"{name} is not available in this mode."}
                else:
                    result = execute(name, args)
                evidence.append({
                    "name": name,
                    "args": args,
                    "table": result.get("table"),
                    "figure": result.get("figure"),
                    "text": result.get("text") or "",
                })
                calls += 1
                contents.append(_types.Content(
                    role="tool",
                    parts=[_types.Part.from_function_response(
                        name=name, response={"result": format_tool_result(result)},
                    )],
                ))
        return AskResult(
            "I stopped at the tool-call limit for this mode. The evidence section has every number I computed.",
            evidence,
            tokens,
        )
    except Exception as exc:
        return AskResult("", evidence, tokens, error=friendly_error(exc))


def make_client(api_key: str):
    """Official client. 30 second timeout. The key is not logged."""
    from google import genai

    return genai.Client(api_key=api_key, http_options=_types.HttpOptions(timeout=30_000))
