"""Test script to verify all figures generate correctly."""
import sys, pathlib, warnings, random
warnings.filterwarnings('ignore')
ROOT = pathlib.Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
import matplotlib; matplotlib.use('Agg')

from src.pipeline import get_data
from src.models import train_all
from src.evaluate import build_metrics_table, get_confusion_matrix, get_roc_data, run_clustering
from src.viz import (
    fig_class_distribution, fig_correlation_heatmap, fig_boxplots,
    fig_model_comparison, fig_confusion_matrix, fig_roc_curves,
    fig_feature_importance, fig_decision_tree, fig_clustering,
    fig_dl_curves, fig_offloading_pie, fig_latency_histogram,
    fig_budget_met, fig_astar_gantt
)
from src.offload_agent import run_offloading_simulation, AStarScheduler
from src.dl_models import train_ann, train_lstm
from sklearn.model_selection import train_test_split

print("Loading data and training models...")
X_train, X_test, y_train, y_test, scaler, df, feat_names = get_data()
fitted = train_all(X_train, y_train)
mdf = build_metrics_table(fitted, X_test, y_test)
best = mdf["F1"].idxmax()
roc_data = get_roc_data(fitted, X_test, y_test)
cm = get_confusion_matrix(fitted[best], X_test, y_test)

print("Fig 1: class distribution")
f = fig_class_distribution(df)
print("Fig 2: correlation heatmap")
f = fig_correlation_heatmap(df)
print("Fig 3: boxplots")
f = fig_boxplots(df)
print("Fig 4: model comparison")
f = fig_model_comparison(mdf)
print("Fig 5: confusion matrix")
f = fig_confusion_matrix(cm, best)
print("Fig 6: ROC curves")
f = fig_roc_curves(roc_data)
print("Fig 7a: feature importance")
f = fig_feature_importance(fitted["Random Forest"], feat_names)
print("Fig 7b: decision tree")
f = fig_decision_tree(fitted["Decision Tree"], feat_names)
print("Fig 8: clustering PCA")
clust = run_clustering(X_test)
f = fig_clustering(clust)
for name, res in clust.items():
    print(f"  {name}: silhouette={res['silhouette']:.4f}")

print("Fig 9: DL curves (MLP fallback)")
Xtr, Xvl, ytr, yvl = train_test_split(X_train, y_train, test_size=0.125, random_state=42)
ann_model, ann_hist = train_ann(Xtr, ytr, Xvl, yvl, epochs=15)
lstm_model, lstm_hist = train_lstm(Xtr, ytr, Xvl, yvl, epochs=10)
f = fig_dl_curves(ann_hist, lstm_hist)

print("Fig 10: offloading simulation")
sim = run_offloading_simulation(n_requests=200)
for policy, records in sim.items():
    pct = 100.0 * sum(r["met_budget"] for r in records) / len(records)
    print(f"  {policy}: {pct:.1f}% budget met")
f = fig_offloading_pie(sim, "threshold")
f = fig_latency_histogram(sim)
f = fig_budget_met(sim)

print("Fig 11: A* Gantt")
random.seed(42)
tasks = [
    {"id": i + 1, "budget_ms": random.choice([40, 60, 80, 120, 200, 300]),
     "priority": random.randint(1, 5)}
    for i in range(15)
]
scheduler = AStarScheduler(tasks)
asgn = scheduler.schedule()
f = fig_astar_gantt(asgn)

figs = list((ROOT / "report" / "figures").glob("*.png"))
print(f"\nFigures saved: {len(figs)}")
for fig_path in sorted(figs):
    print(f"  {fig_path.name}")

print("\n=== ALL TESTS PASSED ===")
