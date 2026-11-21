"""App smoke tests. Legacy model training is the existing dashboard and is not re-run here."""
import os
from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

os.environ["AIOT_SKIP_LEGACY"] = "1"
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_both_modes_load():
    app = AppTest.from_file(APP)
    app.run(timeout=60)
    assert not app.exception
    app.sidebar.radio[0].set_value("researcher").run(timeout=60)
    assert not app.exception
    app.sidebar.radio[0].set_value("student").run(timeout=60)
    assert not app.exception


def test_budget_disables_chat_and_quick_action_runs():
    app = AppTest.from_file(APP)
    app.session_state["token_total"] = 90_000
    app.session_state["token_budget"] = 50_000
    app.session_state["ask_df"] = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0]})
    app.run(timeout=60)
    assert not app.exception
    assert app.chat_input[0].disabled
    app.button(key="qa_health").click().run(timeout=60)
    assert not app.exception
    assert app.session_state["quick_log"]
