"""
app.py
------
Ask Your Data, plus the legacy AI4I dashboard (five demos under one tab).

Run: streamlit run app.py
"""
import os, pathlib, sys, warnings, subprocess
warnings.filterwarnings("ignore")

# ── ensure src is importable ──────────────────────────────────────────────────
ROOT = pathlib.Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Auto-generate dataset if missing (for cloud deployment) ───────────────────
_csv = ROOT / "data" / "ai4i2020.csv"
if not _csv.exists():
    (ROOT / "data").mkdir(exist_ok=True)
    subprocess.run([sys.executable, str(ROOT / "scripts" / "gen_data.py")], check=False)


import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AIoT ML Dashboard",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .main { background: #0f1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        background: #1e2130; border-radius: 8px; padding: 8px 16px;
        color: #9ca3af; font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2130 0%, #252d3d 100%);
        border: 1px solid #334155; border-radius: 12px;
        padding: 16px; margin: 6px 0;
    }
    .stDataFrame { border-radius: 10px; }
    h1, h2, h3 { color: #e2e8f0; }
    .stSelectbox label, .stSlider label { color: #94a3b8; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 AIoT ML Pipeline")
    st.markdown("**Predictive Maintenance + Intelligent Offloading**")
    st.markdown("---")
    st.markdown("### Dataset")
    data_path_input = st.text_input(
        "CSV path",
        value=str(ROOT / "data" / "ai4i2020.csv"),
    )
    st.markdown("---")
    st.markdown("### About")
    st.info(
        "AI4I 2020 Predictive Maintenance Dataset — "
        "10 000 synthetic CNC machine records with 5 failure modes."
    )
    st.markdown("**Model seed:** 42")

# ── Load data (cached) ────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading dataset…")
def load_data(path: str):
    from src.pipeline import load_raw, preprocess
    df = load_raw(pathlib.Path(path))
    X_train, X_test, y_train, y_test, scaler, df_proc, feat_names = preprocess(df)
    return df, X_train, X_test, y_train, y_test, scaler, df_proc, feat_names


@st.cache_resource(show_spinner="Training classifiers…")
def get_trained_models(path: str):
    from src.pipeline import load_raw, preprocess
    from src.models import train_all
    df = load_raw(pathlib.Path(path))
    X_train, X_test, y_train, y_test, scaler, df_proc, feat_names = preprocess(df)
    fitted = train_all(X_train, y_train)
    return fitted


from src.ask_tab import render_ask, render_sidebar
from src.upload_tab import render_upload

render_sidebar()

ask_tab, upload_tab, legacy_tab = st.tabs([
    "💬 Ask Your Data",
    "Upload & Analyze",
    "Legacy AI4I demos",
])

with ask_tab:
    render_ask()

with upload_tab:
    render_upload()

with legacy_tab:
    if os.environ.get("AIOT_SKIP_LEGACY") == "1":
        st.info("Legacy AI4I demos are unchanged. This smoke path skips the heavy training.")
    else:
        try:
            df, X_train, X_test, y_train, y_test, scaler, df_proc, feat_names = load_data(data_path_input)
            data_loaded = True
        except FileNotFoundError as e:
            st.error(f"⚠️ {e}")
            st.info("Run `python scripts/gen_data.py` to generate the dataset, then refresh.")
            data_loaded = False
            st.stop()

        # ══════════════════════════════════════════════════════════════════════════════
        #  TAB DEFINITIONS
        # ══════════════════════════════════════════════════════════════════════════════
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "📊 Data Explorer",
            "🧠 Classical ML",
            "🔬 Deep Learning",
            "📈 Visual Analytics",
            "🚀 Offloading Lab",
        ])

        # ══════════════════════════════════════════════════════════════════════════════
        #  TAB 1 — DATA EXPLORER
        # ══════════════════════════════════════════════════════════════════════════════
        with tab1:
            st.header("📊 Data Explorer")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Records", f"{len(df):,}")
            c2.metric("Features", len(feat_names))
            c3.metric("Failure Rate", f"{df['Machine failure'].mean()*100:.2f}%")
            c4.metric("Failure Count", f"{df['Machine failure'].sum():,}")

            st.subheader("Raw Data Preview")
            st.dataframe(df.head(100), use_container_width=True)

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Class Distribution")
                from src.viz import fig_class_distribution
                st.pyplot(fig_class_distribution(df))

            with col2:
                st.subheader("Descriptive Statistics")
                st.dataframe(df.describe().T.round(3), use_container_width=True)

            st.subheader("Feature Correlation Heatmap")
            from src.viz import fig_correlation_heatmap
            st.pyplot(fig_correlation_heatmap(df))

            st.subheader("Feature Distributions by Failure Class")
            from src.viz import fig_boxplots
            st.pyplot(fig_boxplots(df))

            st.subheader("Failure Mode Breakdown")
            failure_modes = ["TWF", "HDF", "PWF", "OSF", "RNF"]
            mode_counts = df[failure_modes].sum().reset_index()
            mode_counts.columns = ["Failure Mode", "Count"]
            mode_counts["Description"] = [
                "Tool Wear Failure",
                "Heat Dissipation Failure",
                "Power Failure",
                "Overstrain Failure",
                "Random Failures",
            ]
            st.dataframe(mode_counts, use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════════════
        #  TAB 2 — CLASSICAL ML
        # ══════════════════════════════════════════════════════════════════════════════
        with tab2:
            st.header("🧠 Classical ML Classifiers")

            if st.button("🚀 Train All Classifiers", type="primary", key="train_btn"):
                with st.spinner("Training…"):
                    st.cache_resource.clear()

            fitted_models = get_trained_models(data_path_input)

            from src.evaluate import build_metrics_table, get_confusion_matrix, get_roc_data
            from src.viz import fig_model_comparison, fig_confusion_matrix, fig_roc_curves

            st.subheader("📋 Performance Metrics")
            metrics_df = build_metrics_table(fitted_models, X_test, y_test)
            st.dataframe(
                metrics_df.style.highlight_max(axis=0, color="#2d4a3e").format("{:.4f}"),
                use_container_width=True,
            )

            best_model_name = metrics_df["F1"].idxmax()
            st.success(f"🏆 Best model by F1: **{best_model_name}** (F1={metrics_df.loc[best_model_name,'F1']:.4f})")

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("📊 Grouped Bar Chart")
                st.pyplot(fig_model_comparison(metrics_df))

            with col2:
                st.subheader("🔢 Confusion Matrix")
                selected_model = st.selectbox("Select model", list(fitted_models.keys()), key="cm_select")
                cm = get_confusion_matrix(fitted_models[selected_model], X_test, y_test)
                st.pyplot(fig_confusion_matrix(cm, selected_model))

            st.subheader("📉 ROC Curves (all classifiers)")
            roc_data = get_roc_data(fitted_models, X_test, y_test)
            st.pyplot(fig_roc_curves(roc_data))

            st.subheader("🌲 Feature Importance & Decision Tree")
            col3, col4 = st.columns(2)
            from src.viz import fig_feature_importance, fig_decision_tree
            with col3:
                rf = fitted_models.get("Random Forest")
                if rf:
                    st.pyplot(fig_feature_importance(rf, feat_names))
            with col4:
                dt = fitted_models.get("Decision Tree")
                if dt:
                    st.pyplot(fig_decision_tree(dt, feat_names))

        # ══════════════════════════════════════════════════════════════════════════════
        #  TAB 3 — DEEP LEARNING
        # ══════════════════════════════════════════════════════════════════════════════
        with tab3:
            st.header("🔬 Deep Learning Models")

            # Check TF availability
            _tf_msg = ""
            try:
                import tensorflow as tf
                _tf_msg = f"✅ TensorFlow {tf.__version__} available"
                _tf_ok = True
            except ImportError:
                _tf_msg = "⚠️ TensorFlow not installed — using sklearn MLP fallback"
                _tf_ok = False

            st.info(_tf_msg)

            col_ann, col_lstm = st.columns(2)

            with col_ann:
                ann_epochs = st.slider("ANN epochs", 10, 100, 40, key="ann_ep")
            with col_lstm:
                lstm_epochs = st.slider("LSTM epochs", 10, 60, 25, key="lstm_ep")

            if st.button("🚀 Train ANN + LSTM", type="primary", key="dl_btn"):
                from src.dl_models import train_ann, train_lstm, predict_dl
                from sklearn.model_selection import train_test_split
                from sklearn.metrics import accuracy_score, f1_score

                # Use 80% for train, 10% val, 10% test
                Xtr, Xvl, ytr, yvl = train_test_split(X_train, y_train, test_size=0.125, random_state=42)

                with st.spinner("Training ANN…"):
                    ann_model, ann_hist = train_ann(Xtr, ytr, Xvl, yvl, epochs=ann_epochs)

                with st.spinner("Training LSTM…"):
                    lstm_model, lstm_hist = train_lstm(Xtr, ytr, Xvl, yvl, epochs=lstm_epochs)

                # Store in session
                st.session_state["ann_hist"]   = ann_hist
                st.session_state["lstm_hist"]  = lstm_hist
                st.session_state["ann_model"]  = ann_model
                st.session_state["lstm_model"] = lstm_model

                # Evaluate
                ann_pred = predict_dl(ann_model, X_test, is_lstm=False)
                lst_pred = predict_dl(lstm_model, X_test, window=5, is_lstm=True)
                # LSTM shortens by window
                y_test_lstm = y_test[5:]

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ANN Accuracy", f"{accuracy_score(y_test, ann_pred):.4f}")
                c2.metric("ANN F1", f"{f1_score(y_test, ann_pred, zero_division=0):.4f}")
                c3.metric("LSTM Accuracy", f"{accuracy_score(y_test_lstm, lst_pred):.4f}")
                c4.metric("LSTM F1", f"{f1_score(y_test_lstm, lst_pred, zero_division=0):.4f}")

            if "ann_hist" in st.session_state:
                from src.viz import fig_dl_curves
                st.subheader("📉 Training Curves")
                st.pyplot(fig_dl_curves(st.session_state["ann_hist"], st.session_state["lstm_hist"]))
            else:
                st.info("Click **Train ANN + LSTM** to see training curves.")

        # ══════════════════════════════════════════════════════════════════════════════
        #  TAB 4 — VISUAL ANALYTICS (all 11 figures)
        # ══════════════════════════════════════════════════════════════════════════════
        with tab4:
            st.header("📈 Visual Analytics — All 11 Figures")
            st.markdown("All figures are also saved to `report/figures/` as PNGs.")

            from src import viz
            from src.evaluate import build_metrics_table, get_roc_data, run_clustering, get_confusion_matrix

            fitted_models_v = get_trained_models(data_path_input)
            metrics_df_v = build_metrics_table(fitted_models_v, X_test, y_test)
            best_name_v  = metrics_df_v["F1"].idxmax()
            roc_data_v   = get_roc_data(fitted_models_v, X_test, y_test)
            cm_v         = get_confusion_matrix(fitted_models_v[best_name_v], X_test, y_test)

            fig_items = []

            # 1
            with st.expander("1️⃣ Class Distribution", expanded=True):
                f = viz.fig_class_distribution(df); st.pyplot(f)
                fig_items.append(("01_class_distribution", f))

            # 2
            with st.expander("2️⃣ Correlation Heatmap"):
                f = viz.fig_correlation_heatmap(df); st.pyplot(f)

            # 3
            with st.expander("3️⃣ Boxplots — Torque / Speed / Wear by Class"):
                f = viz.fig_boxplots(df); st.pyplot(f)

            # 4
            with st.expander("4️⃣ Model Comparison"):
                f = viz.fig_model_comparison(metrics_df_v); st.pyplot(f)
                st.dataframe(metrics_df_v.round(4), use_container_width=True)

            # 5
            with st.expander("5️⃣ Confusion Matrix — Best Model"):
                f = viz.fig_confusion_matrix(cm_v, best_name_v); st.pyplot(f)

            # 6
            with st.expander("6️⃣ Overlaid ROC Curves"):
                f = viz.fig_roc_curves(roc_data_v); st.pyplot(f)

            # 7
            with st.expander("7️⃣ Feature Importance + Decision Tree"):
                col_a, col_b = st.columns(2)
                rf_v = fitted_models_v.get("Random Forest")
                dt_v = fitted_models_v.get("Decision Tree")
                if rf_v:
                    with col_a:
                        f = viz.fig_feature_importance(rf_v, feat_names); st.pyplot(f)
                if dt_v:
                    with col_b:
                        f = viz.fig_decision_tree(dt_v, feat_names); st.pyplot(f)

            # 8
            with st.expander("8️⃣ PCA Clustering — K-Means / Hierarchical / DBSCAN / K-Medoids"):
                with st.spinner("Running clustering…"):
                    clust_res = run_clustering(X_test)
                f = viz.fig_clustering(clust_res); st.pyplot(f)
                for name, res in clust_res.items():
                    sil = res["silhouette"]
                    st.metric(f"{name} Silhouette", f"{sil:.4f}" if not np.isnan(sil) else "N/A")

            # 9
            with st.expander("9️⃣ ANN + LSTM Loss / Accuracy Curves"):
                if "ann_hist" in st.session_state:
                    f = viz.fig_dl_curves(st.session_state["ann_hist"], st.session_state["lstm_hist"])
                    st.pyplot(f)
                else:
                    st.warning("Train models in Tab 3 first to see curves here.")

            # 10
            with st.expander("🔟 Offloading Lab — Pie / Latency Histogram / Budget Met"):
                with st.spinner("Running offloading simulation…"):
                    from src.offload_agent import run_offloading_simulation
                    sim = run_offloading_simulation(n_requests=300)
                col_p, col_h = st.columns(2)
                with col_p:
                    f = viz.fig_offloading_pie(sim, "threshold"); st.pyplot(f)
                with col_h:
                    f = viz.fig_latency_histogram(sim); st.pyplot(f)
                f = viz.fig_budget_met(sim); st.pyplot(f)

            # 11
            with st.expander("1️⃣1️⃣ A* Scheduler Gantt"):
                from src.offload_agent import AStarScheduler
                import random
                random.seed(42)
                tasks = [
                    {"id": i + 1, "budget_ms": random.choice([40, 60, 80, 120, 200, 300]),
                     "priority": random.randint(1, 5)}
                    for i in range(20)
                ]
                scheduler = AStarScheduler(tasks)
                asgn = scheduler.schedule()
                f = viz.fig_astar_gantt(asgn[:20]); st.pyplot(f)
                st.dataframe(pd.DataFrame(asgn), use_container_width=True)

        # ══════════════════════════════════════════════════════════════════════════════
        #  TAB 5 — OFFLOADING LAB
        # ══════════════════════════════════════════════════════════════════════════════
        with tab5:
            st.header("🚀 Offloading Lab — Intelligent Inference Routing")

            st.markdown("""
            > **Viva Insight**: Training a model is 20% of an AIoT problem.
            > The other 80% is *deciding where it runs*.
            > This lab measures LOCAL / EDGE / CLOUD decisions under real latency budgets.
            """)

            # ── Interactive single-request decision ───────────────────────────────────
            st.subheader("🎛️ Interactive Offloading Decision")
            col1, col2, col3, col4 = st.columns(4)
            budget_ms  = col1.slider("Latency Budget (ms)", 20, 500, 100, key="ol_budget")
            battery    = col2.slider("Battery (%)", 1, 100, 60,  key="ol_bat")
            rssi       = col3.slider("RSSI (dBm)", -100, -40, -65, key="ol_rssi")
            rtt        = col4.slider("RTT (ms)", 1, 150, 30, key="ol_rtt")

            from src.offload_agent import threshold_policy, QLearningOffloader

            t_decision = threshold_policy(budget_ms, battery, rssi, rtt)

            # Quick Q-learner (pre-trained on defaults)
            @st.cache_resource
            def get_qagent():
                agent = QLearningOffloader()
                agent.train(n_episodes=3000)
                return agent

            qagent = get_qagent()
            q_decision = qagent.decide(battery, rssi, budget_ms)

            col_a, col_b = st.columns(2)
            tier_emoji = {"LOCAL": "📱", "EDGE": "🌐", "CLOUD": "☁️"}
            col_a.metric("Threshold Policy", f"{tier_emoji.get(t_decision,'')} {t_decision}")
            col_b.metric("Q-Learning Policy", f"{tier_emoji.get(q_decision,'')} {q_decision}")

            st.markdown("---")

            # ── Simulation ────────────────────────────────────────────────────────────
            st.subheader("📊 Policy Simulation (500 requests)")
            n_sim = st.slider("Number of requests to simulate", 100, 1000, 500, step=100, key="n_sim")

            if st.button("▶️ Run Simulation", type="primary", key="sim_btn"):
                from src.offload_agent import run_offloading_simulation
                with st.spinner("Simulating…"):
                    sim_results = run_offloading_simulation(n_requests=n_sim)
                st.session_state["sim_results"] = sim_results

            if "sim_results" in st.session_state:
                sim_results = st.session_state["sim_results"]
                from src.viz import fig_offloading_pie, fig_latency_histogram, fig_budget_met

                st.subheader("📊 Tier Distribution")
                policy_tabs = st.tabs(list(sim_results.keys()))
                for ptab, policy in zip(policy_tabs, sim_results.keys()):
                    with ptab:
                        st.pyplot(fig_offloading_pie(sim_results, policy))

                st.subheader("⏱️ Latency Distribution")
                st.pyplot(fig_latency_histogram(sim_results))

                st.subheader("✅ Budget Met (%) by Policy")
                st.pyplot(fig_budget_met(sim_results))

                # Summary table
                rows = []
                for policy, records in sim_results.items():
                    tiers = [r["tier"] for r in records]
                    rows.append({
                        "Policy": policy.replace("_", " ").title(),
                        "% Budget Met": f"{100*np.mean([r['met_budget'] for r in records]):.1f}%",
                        "Avg Latency (ms)": f"{np.mean([r['latency_ms'] for r in records]):.1f}",
                        "LOCAL %": f"{100*tiers.count('LOCAL')/len(tiers):.1f}%",
                        "EDGE %":  f"{100*tiers.count('EDGE')/len(tiers):.1f}%",
                        "CLOUD %": f"{100*tiers.count('CLOUD')/len(tiers):.1f}%",
                    })
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

            st.markdown("---")

            # ── A* Scheduler ──────────────────────────────────────────────────────────
            st.subheader("🗓️ A* Task Scheduler")
            n_tasks = st.slider("Number of tasks", 5, 30, 15, key="n_tasks")

            if st.button("📋 Schedule Tasks (A*)", key="astar_btn"):
                import random as _rand
                _rand.seed(42)
                from src.offload_agent import AStarScheduler
                from src.viz import fig_astar_gantt

                tasks = [
                    {
                        "id": i + 1,
                        "budget_ms": _rand.choice([40, 60, 80, 120, 180, 250]),
                        "priority": _rand.randint(1, 5),
                    }
                    for i in range(n_tasks)
                ]
                scheduler = AStarScheduler(tasks)
                asgn = scheduler.schedule()

                st.pyplot(fig_astar_gantt(asgn))
                df_asgn = pd.DataFrame(asgn)
                df_asgn["met_budget"] = df_asgn["met_budget"].map({True: "✅", False: "❌"})
                st.dataframe(df_asgn, use_container_width=True)

                pct_met = 100 * sum(a["actual_latency"] <= a["budget_ms"] for a in asgn) / len(asgn)
                st.metric("Tasks Meeting Budget", f"{pct_met:.1f}%")

            # ── On-device latency benchmark ───────────────────────────────────────────
            st.markdown("---")
            st.subheader("⚡ On-Device Latency Benchmark")
            if st.button("🔬 Measure LOCAL inference latency", key="bench_btn"):
                from src.offload_agent import measure_local_latency
                rf = get_trained_models(data_path_input).get("Random Forest")
                if rf:
                    with st.spinner("Benchmarking (50 runs)…"):
                        lat = measure_local_latency(rf, X_test)
                    st.metric("Mean LOCAL latency (RF, 50 runs)", f"{lat:.2f} ms")
                    if lat < 50:
                        st.success("✅ Well within tight latency budget (< 50 ms)")
                    else:
                        st.warning("⚠️ Consider EDGE offloading for tight budgets")
