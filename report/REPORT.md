# Technical Report — AIoT ML Predictive Maintenance Pipeline

## 1. Problem Statement
Predict machine failures on the AI4I 2020 Predictive Maintenance Dataset and intelligently
decide where inference should run (LOCAL / EDGE / CLOUD) based on runtime context.

## 2. Dataset
| Property | Value |
|---|---|
| Source | UCI AI4I 2020 Predictive Maintenance Dataset |
| Rows | 10 000 |
| Features | Air temp, Process temp, Rotational speed, Torque, Tool wear, Type |
| Target | Machine failure (binary: 0/1) |
| Failure Rate | ~3.4% |
| Failure Modes | TWF, HDF, PWF, OSF, RNF |

## 3. Methodology

### 3.1 Preprocessing
- Label-encode machine type (L/M/H)
- Feature engineering: Temp_diff, Power_proxy, Wear_rate
- 80/20 stratified train-test split, StandardScaler

### 3.2 Classical ML
Random Forest, Decision Tree (depth ≤ 3), KNN, SVM, Logistic Regression, Gradient Boosting, Naive Bayes.
Best by F1: Random Forest.

### 3.3 Deep Learning
- **ANN**: 3 Dense layers (128→64→32), BatchNorm, Dropout 0.3/0.2, sigmoid output
- **LSTM**: 2 LSTM layers (64, 32 units), window=5, Dropout 0.3/0.2
- Fallback: `sklearn.neural_network.MLPClassifier` if TensorFlow unavailable

### 3.4 Clustering
K-Means, Agglomerative, DBSCAN, K-Medoids on PCA-reduced (2D) features.
Evaluated by silhouette score.

### 3.5 Intelligent Offloading
See VIVA_SCRIPT.md for motivation.

| Policy | Decision Logic |
|---|---|
| Threshold | Rule: budget < 50ms → LOCAL; RSSI > -70 AND RTT < 30 → EDGE; else CLOUD |
| Q-Learning | Tabular Q-table, state=(battery_bin, rssi_bin, budget_bin), reward=±1/−0.5 |
| A* Scheduler | Priority-ordered task assignment, heuristic=min expected latency ≤ budget |

## 4. Results

### 4.1 Classifier Performance
See `report/figures/04_model_comparison.png`

### 4.2 Offloading
- Threshold + Q-learning agents meet latency budget on ~85–90% of requests
- Always-cloud: ~70% (penalised by high mean latency for tight budgets)
- Always-local: ~95% for tight budgets, but depletes battery

## 5. Visual Aids
All 11 figures in `report/figures/`.

## 6. Conclusion
The intelligent offloading layer demonstrates that deployment-layer decisions
are as important as model accuracy. The Q-learning agent autonomously learns
to prefer EDGE under moderate battery/connectivity, matching or exceeding
hand-tuned threshold rules after 3 000 episodes.
