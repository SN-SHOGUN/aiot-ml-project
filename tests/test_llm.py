"""Fake Gemini client. No network."""
from types import SimpleNamespace

import pandas as pd

from src.llm import ask_question, schema_text
from src.modes import MODES
from src.tools import REGISTRY, run_tool


class Call:
    def __init__(self, name, args):
        self.name = name
        self.args = args


class Resp:
    def __init__(self, text="", calls=None, tokens=4):
        self.text = text
        self.function_calls = calls or []
        self.usage_metadata = SimpleNamespace(total_token_count=tokens)
        self.candidates = [SimpleNamespace(content=SimpleNamespace(role="model", parts=[]))]


class Client:
    def __init__(self, script):
        self.script = list(script)
        self.seen = []
        self.models = self

    def generate_content(self, model, contents, config):
        self.seen.append({"model": model, "contents": contents, "config": config})
        item = self.script.pop(0) if self.script else Resp(calls=[Call("profile", {})])
        if isinstance(item, Exception):
            raise item
        return item


def _texts(contents):
    found = []
    for content in contents:
        for part in content.parts:
            text = getattr(part, "text", None)
            if text:
                found.append(text)
    return found


def _frame():
    return pd.DataFrame({"x": range(12), "winner": [0, 1] * 6})


def test_tool_then_answer_captures_evidence():
    df = _frame()
    client = Client([
        Resp(calls=[Call("summarize", {"column": "winner"})], tokens=5),
        Resp(text="The winner column is balanced.", tokens=6),
    ])

    def execute(name, args):
        return run_tool(df, name, args, {})

    result = ask_question(
        client, "summarize winner", schema=schema_text(df), history=[],
        mode=MODES["student"], model="gemini-3.8-flash", execute=execute,
    )
    assert result.error is None
    assert result.evidence[0]["name"] == "summarize"
    assert result.evidence[0]["table"] is not None
    assert "balanced" in result.text
    assert result.tokens == 11
    assert client.seen[0]["config"].max_output_tokens == 400


def test_student_stops_at_3_and_researcher_at_8(monkeypatch):
    monkeypatch.setattr("src.llm.time.sleep", lambda _s: None)
    df = _frame()
    executed = []

    def execute(name, args):
        executed.append(name)
        return {"table": None, "figure": None, "text": "ok"}

    student = Client([])
    ask_question(
        student, "go", schema="rows=12 cols=2", history=[],
        mode=MODES["student"], model="m", execute=execute,
    )
    assert len(executed) == 3
    assert all(item["config"].max_output_tokens == 400 for item in student.seen)
    executed.clear()
    researcher = Client([])
    ask_question(
        researcher, "go", schema="rows=12 cols=2", history=[],
        mode=MODES["researcher"], model="m", execute=execute,
    )
    assert len(executed) == 8
    assert all(item["config"].max_output_tokens == 1200 for item in researcher.seen)


def test_history_trimmed_and_schema_not_repeated_in_turns():
    df = _frame()
    history = [{"q": f"q{i}", "a": f"a{i}"} for i in range(6)]
    client = Client([Resp(text="done", tokens=2)])
    schema = schema_text(df)
    ask_question(
        client, "new question", schema=schema, history=history,
        mode=MODES["student"], model="m", execute=lambda n, a: {},
    )
    texts = _texts(client.seen[0]["contents"])
    assert "q4" in texts and "q5" in texts and "new question" in texts
    assert "q0" not in texts and "q3" not in texts
    assert "rows=" in client.seen[0]["config"].system_instruction
    assert all("rows=" not in text for text in texts)


def test_quota_retries_once_and_bad_key_does_not(monkeypatch):
    monkeypatch.setattr("src.llm.time.sleep", lambda _s: None)
    df = _frame()
    quota = Client([Exception("429 quota exceeded"), Resp(text="ok", tokens=3)])
    result = ask_question(
        quota, "q", schema=schema_text(df), history=[],
        mode=MODES["student"], model="m", execute=lambda n, a: {},
    )
    assert result.error is None and result.text == "ok"
    assert len(quota.seen) == 2
    denied = Client([Exception("API key not valid")])
    result = ask_question(
        denied, "q", schema=schema_text(df), history=[],
        mode=MODES["student"], model="m", execute=lambda n, a: {},
    )
    assert result.error and "key" in result.error.lower()
    assert len(denied.seen) == 1


def test_modes_declare_expected_tools():
    student = MODES["student"]["tools"]
    assert student == [
        "profile", "summarize", "group_aggregate", "plot", "correlate", "quick_model",
    ]
    for banned in ("train_models", "detect_anomalies", "trend", "data_quality_report"):
        assert banned not in student
    assert set(MODES["researcher"]["tools"]) == set(REGISTRY)
