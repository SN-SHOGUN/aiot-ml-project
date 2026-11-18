# VIVA SCRIPT

## Viva Narrative (verbatim — paste into notes)

> "The professor asked us to extend an algorithm. We chose to extend the deployment layer of ML.
> Training a model is 20% of an AIoT problem; the other 80% is deciding where it runs.
> So on top of a full ML/DL pipeline, we built an intelligent offloading agent that per-request
> decides LOCAL / EDGE / CLOUD from latency budget, battery, RSSI and RTT.
> We measured real on-device inference latency with time.perf_counter() and compared it against
> simulated edge/cloud latencies. Result: our threshold and Q-learning agents meet the latency
> budget on X% of requests vs Y% for always-cloud, while offloading Z% of the compute —
> a direct latency-energy trade-off, which is the core of AIoT."

---

## Expected Viva Questions & Answers

**Q: Why did you choose predictive maintenance?**
A: It's a canonical AIoT problem — sensors → model → actuation loop — with public, well-documented data (AI4I 2020) that includes realistic failure modes.

**Q: What is the offloading problem in AIoT?**
A: In AIoT, trained models must run somewhere. Running locally saves latency but drains battery. Cloud offloading preserves battery but adds network delay. The optimal choice is dynamic and context-dependent — that's what our agent learns.

**Q: How does your Q-learning agent work?**
A: State = (battery_bin, rssi_bin, budget_bin) → 27 states. Actions = {LOCAL, EDGE, CLOUD}. Reward = +1 if latency ≤ budget and battery OK, -1 if budget missed, -0.5 if battery depleted. After 3 000 episodes the Q-table converges to prefer EDGE under moderate conditions, LOCAL for tight budgets, CLOUD when connectivity is poor.

**Q: How did you measure local latency?**
A: We used `time.perf_counter()` before and after `clf.predict()` over 50 runs and averaged. This captures real on-device (CPU) inference time without timer overhead.

**Q: What is the A* scheduler doing?**
A: It treats each inference task as a node in a priority queue. The heuristic is the expected latency for each tier. It greedily assigns each task (sorted by urgency/priority) to the cheapest tier that fits within its latency budget.

**Q: What's the difference between threshold and Q-learning policies?**
A: Threshold is hand-crafted rules (fast, interpretable, no training). Q-learning learns the optimal policy from experience — it adapts to environments the threshold rules weren't designed for, e.g., partial battery drain patterns.

**Q: Why is LSTM used here?**
A: Machine sensor data is a time series. LSTM captures temporal dependencies across the sliding window (5 consecutive readings), potentially detecting pre-failure patterns that i.i.d. classifiers miss.

**Q: What fallback did you implement for TensorFlow?**
A: `sklearn.neural_network.MLPClassifier` for ANN, and a flattened-window MLP for LSTM. The fallback maintains identical API so the app renders without errors even without GPU/TF.

**Q: What failure modes does the dataset have?**
A: Five: Tool Wear Failure (TWF), Heat Dissipation Failure (HDF), Power Failure (PWF), Overstrain Failure (OSF), and Random Failures (RNF). Each has distinct sensor signatures exploited by the classifiers.

---

## Key Numbers to Memorise
- Dataset: 10 000 rows, ~3.4% failure rate
- Features: 9 (6 raw + 3 engineered)
- Classifiers: 7 (RF, DT, KNN, SVM, LR, GB, NB)
- Q-learning states: 27 (3³), actions: 3, episodes: 3 000
- Local latency benchmark: ~1–5 ms (sklearn RF, 1 sample)
- Edge latency (simulated): ~80 ms mean
- Cloud latency (simulated): ~180 ms mean
