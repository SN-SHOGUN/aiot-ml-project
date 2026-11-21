"""Gemini tool loop. Model text is never executed and never treated as a measurement."""
from __future__ import annotations

import inspect
import time
from dataclasses import dataclass

import pandas as pd

from src.tools import REGISTRY

MAX_CHARS = 1200
MAX_ROWS = 15

try:
    from google.genai import types
except ImportError:  # unit tests can run before the SDK is installed
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

    class _Bag:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class types:  # type: ignore
        Part = _Part
        Content = _Bag
        FunctionDeclaration = _Bag
        Tool = _Bag
        AutomaticFunctionCallingConfig = _Bag
        GenerateContentConfig = _Bag
        HttpOptions = _Bag


@dataclass
class AskResult:
    text: str
    evidence: list
    tokens: int
    error: str | None = None


def format_tool_result(result: dict) -> str:
    """Compact CSV the model is allowed to see. At most 15 rows and 1,200 characters."""
    chunks = []
    if result.get("text"):
        chunks.append(str(result["text"]))
    table = result.get("table")
    if isinstance(table, pd.DataFrame):
        chunks.append(table.head(MAX_ROWS).to_csv(index=False))
    raw = "\n".join(chunks).strip() or "No result."
    if len(raw) > MAX_CHARS:
        raw = raw[: MAX_CHARS - 3] + "..."
    return raw


def schema_text(df: pd.DataFrame, send_samples: bool = False) -> str:
    """One line per column. Sample rows are optional and off by default."""
    lines = [f"rows={len(df)} cols={df.shape[1]}"]
    for col in df.columns:
        series = df[col]
        miss = 100 * float(series.isna().mean())
        unique = int(series.nunique(dropna=True))
        if pd.api.types.is_numeric_dtype(series):
            extra = f"min={series.min()}|max={series.max()}"
        else:
            top = ",".join(series.astype(str).value_counts().head(3).index.tolist())
            extra = f"top={top}"
        lines.append(f"{col}|{series.dtype}|miss={miss:.1f}%|unique={unique}|{extra}")
    if send_samples:
        lines.append("sample_rows:")
        lines.append(df.head(5).to_csv(index=False))
    text = "\n".join(lines)
    return text[:6000] + ("\n...schema truncated" if len(text) > 6000 else "")


def _json_schema(annotation) -> dict:
    text = str(annotation)
    if annotation is int or text == "int":
        return {"type": "integer"}
    if annotation is float:
        return {"type": "number"}
    if annotation is bool or text == "bool":
        return {"type": "boolean"}
    if "list" in text.lower():
        return {"type": "array", "items": {"type": "string"}}
    return {"type": "string"}


def tool_declarations(names: list[str]):
    """Build Gemini declarations from the tool functions. One place to add a tool."""
    decls = []
    for name in names:
        fn = REGISTRY[name]
        props, required = {}, []
        for pname, param in inspect.signature(fn).parameters.items():
            if pname == "df":
                continue
            props[pname] = _json_schema(param.annotation)
            if param.default is inspect.Parameter.empty:
                required.append(pname)
        schema = {"type": "object", "properties": props, "required": required}
        doc = ((inspect.getdoc(fn) or name).split("\n")[0])[:180]
        try:
            decls.append(types.FunctionDeclaration(
                name=name, description=doc, parameters_json_schema=schema,
            ))
        except TypeError:
            decls.append(types.FunctionDeclaration(name=name, description=doc, parameters=schema))
    return decls


def friendly_error(exc: Exception) -> str:
    """Map API failures to a short message. Never include the key."""
    msg = str(exc).lower()
    if any(s in msg for s in ("429", "quota", "resource_exhausted", "rate limit", "rate-limit")):
        return "Gemini is rate-limiting this key (quota or 429). Wait a minute and try again. Your table is still loaded."
    if any(s in msg for s in ("api key", "api_key", "unauthorized", "permission", "401", "403", "invalid")):
        return (
            "Gemini rejected the API key. Paste a valid key in the sidebar, or set "
            "GEMINI_API_KEY in .streamlit/secrets.toml. The key stays in this session only."
        )
    if any(s in msg for s in ("timeout", "timed out", "connection", "network", "unavailable", "503", "502")):
        return "Could not reach Gemini (network or timeout). Check your connection and try again."
    return "Gemini did not return a usable answer. Try a simpler question."


def _retryable(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(s in msg for s in ("429", "quota", "resource_exhausted", "rate limit", "rate-limit"))


def _tokens(resp) -> int:
    meta = getattr(resp, "usage_metadata", None)
    if meta is None:
        return 0
    if isinstance(meta, dict):
        total = meta.get("total_token_count")
        if total:
            return int(total)
        return int(meta.get("prompt_token_count") or 0) + int(meta.get("candidates_token_count") or 0)
    total = getattr(meta, "total_token_count", None)
    if total:
        return int(total)
    prompt = int(getattr(meta, "prompt_token_count", 0) or 0)
    cand = int(getattr(meta, "candidates_token_count", 0) or 0)
    return prompt + cand


def _args(fc) -> dict:
    raw = getattr(fc, "args", {}) or {}
    if isinstance(raw, dict):
        return dict(raw)
    try:
        return {key: raw[key] for key in raw}
    except Exception:
        return {}


def _call(client, payload):
    """One retry on quota errors, then stop. Never loop."""
    try:
        return client.models.generate_content(**payload)
    except Exception as exc:
        if _retryable(exc):
            time.sleep(1)
            return client.models.generate_content(**payload)
        raise


def _trim_history(history: list[dict], turns: int) -> list[dict]:
    clean = []
    for item in history or []:
        question, answer = item.get("q"), item.get("a")
        if question and answer:
            clean.append({"q": str(question), "a": str(answer)})
    return clean[-int(turns):]


def ask_question(client, question: str, *, schema: str, history: list[dict], mode: dict, model: str, execute) -> AskResult:
    """Call tools until the model writes an answer or the mode cap is hit."""
    allowed = set(mode["tools"])
    system = mode["system"] + "\n\nDATA SCHEMA (computed locally; do not invent columns):\n" + schema
    contents = []
    for turn in _trim_history(history, mode["history_turns"]):
        contents.append(types.Content(role="user", parts=[types.Part.from_text(text=turn["q"])]))
        contents.append(types.Content(role="model", parts=[types.Part.from_text(text=turn["a"])]))
    contents.append(types.Content(role="user", parts=[types.Part.from_text(text=question)]))
    config = types.GenerateContentConfig(
        system_instruction=system,
        tools=[types.Tool(function_declarations=tool_declarations(list(mode["tools"])))],
        max_output_tokens=int(mode["max_output_tokens"]),
        temperature=0.2,
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
    )
    evidence, tokens = [], 0
    cap = int(mode["max_tool_calls"])
    try:
        while len(evidence) < cap:
            resp = _call(client, {"model": model, "contents": contents, "config": config})
            tokens += _tokens(resp)
            calls = list(getattr(resp, "function_calls", None) or [])
            if not calls:
                text = (getattr(resp, "text", None) or "").strip()
                if not text:
                    text = "I could not determine an answer from the tools. Try a more specific question."
                return AskResult(text=text, evidence=evidence, tokens=tokens)
            candidates = getattr(resp, "candidates", None) or []
            if candidates and getattr(candidates[0], "content", None) is not None:
                contents.append(candidates[0].content)
            parts = []
            for fc in calls:
                if len(evidence) >= cap:
                    break
                name, args = fc.name, _args(fc)
                if name not in allowed:
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
                parts.append(types.Part.from_function_response(
                    name=name, response={"result": format_tool_result(result)},
                ))
            contents.append(types.Content(role="tool", parts=parts))
        return AskResult(
            text="I stopped at the tool-call limit. The evidence below is everything Python measured.",
            evidence=evidence,
            tokens=tokens,
        )
    except Exception as exc:
        return AskResult(text="", evidence=evidence, tokens=tokens, error=friendly_error(exc))


def make_client(api_key: str):
    """Official client. Timeout is 30 seconds. The key is not logged."""
    from google import genai
    return genai.Client(api_key=api_key, http_options=types.HttpOptions(timeout=30_000))
