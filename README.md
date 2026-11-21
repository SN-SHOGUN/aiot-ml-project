# AIoT ML Pipeline — Predictive Maintenance + Intelligent Offloading

## Overview
End-to-end ML/DL pipeline on the **AI4I 2020 Predictive Maintenance Dataset**, extended with an
**Intelligent Offloading Agent** that decides per-request whether inference runs LOCAL / EDGE / CLOUD
based on latency budget, battery level, RSSI and round-trip-time.

## Project Structure
```
aiot_ml_project/
├── data/               # ai4i2020.csv (generated or downloaded)
├── scripts/
│   └── gen_data.py     # Downloads or synthesises the dataset
├── src/
│   ├── pipeline.py     # Data loading + preprocessing
│   ├── models.py       # Classical ML classifiers
│   ├── evaluate.py     # Metrics, confusion matrix, ROC
│   ├── dl_models.py    # ANN + LSTM (TF fallback to MLP)
│   ├── offload_agent.py# Threshold, Q-learning, A* scheduler
│   └── viz.py          # All plotting helpers
├── app.py              # Streamlit 5-tab dashboard
├── notebooks/
│   └── experiments.ipynb
├── report/
│   ├── REPORT.md
│   ├── VIVA_SCRIPT.md
│   └── figures/        # PNGs exported by the notebook
├── requirements.txt
├── README.md
└── NOTES.md
```

## Quick Start
```bash
pip install -r requirements.txt
python scripts/gen_data.py          # generate data/ai4i2020.csv
streamlit run app.py                # launch dashboard
jupyter nbconvert --to notebook --execute notebooks/experiments.ipynb --output notebooks/experiments_executed.ipynb
```

## Tabs
| Tab | Contents |
|-----|----------|
| Ask Your Data | Upload or paste a table, ask in plain language, or run quick actions with no API call |
| Upload & Analyze | Train on the same table |
| Legacy AI4I demos | The original Data Explorer, Classical ML, Deep Learning, Visual Analytics, and Offloading Lab |

## Ask Your Data

```bash
pip install -r requirements.txt
streamlit run app.py
```

Put the key in `.streamlit/secrets.toml` (gitignored):

```toml
GEMINI_API_KEY = "your-key"
```

Or type it in the sidebar. It stays in `st.session_state` and is never written to disk.

**What is sent:** one line per column (name, dtype, missing %, unique count, min/max or top values). Five sample rows only if Researcher mode has "Send 5 sample rows" on. The full table is not sent. Python computes every number. The model cannot run code.

**Modes:** Student uses six tools, 3 tool calls, 400 output tokens, and 2 history turns. Researcher uses every tool, 8 calls, 1,200 output tokens, and 4 history turns. The deeper model is off until you tick it. Quick actions (data health, class balance, correlation, missing values) make no API call. The meter shows tokens for the last answer and the session. At the budget (default 50,000) the chat box stops and quick actions keep working.

**Add a tool:** write a function in `src/tools.py` that returns `{table, figure, text}`, register it in `REGISTRY`. Researcher mode picks it up. Add the name to `STUDENT_TOOLS` in `src/modes.py` only if students should see it.

**Limits:** the assistant can only use those tools. Small samples, imbalance, and leakage still need a human review. This is not a substitute for domain review. Real Gemini token counts can only be checked with your API key.

### Manual checks
- `data/ai4i2020.csv`: ask what drives machine failure. The evidence should show recall and PR-AUC next to the majority baseline, not accuracy alone.
- A messy awards CSV with `film_id` and a column that copies the winner: ask if winners can be predicted. The shortcut warning should name that column.
- Paste about 20 CSV rows and ask for a summary.
- A wrong API key should show a friendly error and the app should stay up.
