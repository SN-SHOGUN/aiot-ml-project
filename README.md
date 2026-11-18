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
| # | Tab | Contents |
|---|-----|----------|
| 1 | Data Explorer | Dataset overview, class distribution, correlations |
| 2 | Classical ML | Train + compare RF, DT, KNN, SVM, Logistic Reg |
| 3 | Deep Learning | ANN / LSTM (or MLP fallback) |
| 4 | Visual Analytics | All 11 visual aids |
| 5 | Offloading Lab | Threshold / Q-learning / A* policies |
