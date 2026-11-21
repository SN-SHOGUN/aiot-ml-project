"""Tool tests on a tiny frame. No network."""
import numpy as np
import pandas as pd

from src.generic import build_dataset
from src.ingest import parse_text
from src.llm import format_tool_result
from src.tools import REGISTRY, run_tool


def toy():
    rng = np.random.default_rng(0)
    n = 40
    frame = pd.DataFrame({
        "film_id": [f"f{i}" for i in range(n)],
        "group": ["A", "B"] * (n // 2),
        "x": rng.normal(size=n),
        "y": rng.normal(size=n),
        "day": pd.date_range("2024-01-01", periods=n, freq="D").astype(str),
        "winner": [0] * 20 + [1] * 20,
        "award": [0] * 20 + [1] * 20,
    })
    frame.loc[0, "x"] = np.nan
    return frame


def test_every_tool_and_bad_columns():
    df = toy()
    good = {
        "profile": {},
        "summarize": {"column": "x"},
        "group_aggregate": {"group_by": "group", "metric": "x", "agg": "mean"},
        "correlate": {},
        "plot": {"kind": "scatter", "x": "x", "y": "y"},
        "detect_anomalies": {"column": "y"},
        "trend": {"time_column": "day", "value_column": "y", "freq": "W"},
        "quick_model": {"target": "winner"},
        "train_models": {"target": "winner"},
        "data_quality_report": {},
    }
    for name, args in good.items():
        out = run_tool(df, name, args, cache=None)
        assert "text" in out and "Unknown columns" not in out["text"], name
    bad = {
        "summarize": {"column": "missing"},
        "group_aggregate": {"group_by": "nope", "metric": "x"},
        "correlate": {"columns": ["nope"]},
        "plot": {"kind": "hist", "x": "nope"},
        "detect_anomalies": {"column": "nope"},
        "trend": {"time_column": "nope", "value_column": "x"},
        "quick_model": {"target": "nope"},
        "train_models": {"target": "nope"},
    }
    for name, args in bad.items():
        out = REGISTRY[name](df, **args)
        assert out["figure"] is None
        assert "Unknown" in out["text"]


def test_shortcut_and_baseline_language():
    out = run_tool(toy(), "train_models", {"target": "winner"})
    assert "majority_baseline_accuracy" in out["text"]
    assert "award" in out["text"]
    assert "SHORTCUT WARNING" in out["text"]
    assert {"accuracy", "recall", "f1", "pr_auc"} <= set(out["table"].columns)
    quick = run_tool(toy(), "quick_model", {"target": "winner"})
    assert "baseline_accuracy=" in quick["text"]
    assert "beats guessing" in quick["text"]


def test_group_split_is_disjoint():
    df = toy()
    packed = build_dataset(df, "winner", group_col="group")
    train_groups = set(df.loc[packed["train_index"], "group"])
    test_groups = set(df.loc[packed["test_index"], "group"])
    assert train_groups.isdisjoint(test_groups)


def test_trim_wide_table():
    frame = pd.DataFrame({"a": ["x" * 400] * 10_000})
    text = format_tool_result({"table": frame, "figure": None, "text": ""})
    assert len(text) <= 1200
    assert text.count("\n") <= 16


def test_cache_runs_once():
    df = toy()
    cache, calls = {}, {"n": 0}
    original = REGISTRY["profile"]

    def spy(frame):
        calls["n"] += 1
        return original(frame)

    REGISTRY["profile"] = spy
    try:
        run_tool(df, "profile", {}, cache)
        run_tool(df, "profile", {}, cache)
    finally:
        REGISTRY["profile"] = original
    assert calls["n"] == 1


def test_parse_csv_tsv_json():
    csv_df = parse_text("a,b\n1,2\n3,4\n")
    assert list(csv_df.columns) == ["a", "b"]
    tsv_df = parse_text("a\tb\n1\t2\n")
    assert tsv_df.shape == (1, 2)
    json_df = parse_text('[{"a": 1, "b": 2}]')
    assert int(json_df.loc[0, "a"]) == 1


def test_plot_rejects_unknown_kind():
    out = REGISTRY["plot"](toy(), kind="pie", x="x")
    assert "kind" in out["text"]
