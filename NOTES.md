# NOTES — Implementation Decisions

## Dataset
- Primary source: UCI AI4I 2020 Predictive Maintenance Dataset (synthetic CNC machine data)
- Fallback: `scripts/gen_data.py` synthesises 10 000 rows with matching schema if download fails

## Offloading Agent
- **Threshold policy**: if `latency_budget_ms < 50` → LOCAL; elif RSSI > -70 AND RTT < 30 → EDGE; else CLOUD
- **Q-learning**: state = (battery_bin, rssi_bin, budget_bin); action = {LOCAL, EDGE, CLOUD}; reward = +1 if met budget, -1 if missed, -0.5 if battery depleted
- **A\* scheduler**: nodes are (task_id, tier); heuristic = estimated remaining latency; produces an ordered list of assignments

## Deep Learning
- TensorFlow 2.x used for ANN and LSTM when available
- Falls back to `sklearn.neural_network.MLPClassifier` if TF not installed
- LSTM uses a sliding-window approach (window=5) to capture temporal patterns

## Clustering
- K-Means, Agglomerative, DBSCAN, K-Medoids (sklearn_extra) on PCA-reduced features
- Silhouette score computed for each

## Reproducibility
- Random seed = 42 everywhere
- All paths relative via `pathlib.Path`
