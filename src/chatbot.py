"""
chatbot.py
----------
High-IQ conversational AI for DataMind.
Understands broad, fuzzy, exploratory, and specific inquiries about:
- Missing values, data quality, outliers, distributions
- Algorithm comparisons, neural network architecture, model evaluation
- Concrete recommendations on how to proceed
- Plain text or advanced data science questions
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd


def _num(x) -> str:
    try:
        return f"{float(x):,.4g}"
    except Exception:
        return str(x)


def answer(question: str, df: pd.DataFrame | None, results: dict | None) -> str:
    raw_q = question.strip()
    q = raw_q.lower()

    if not q:
        return "👋 Ask me anything about your dataset, statistical distributions, algorithms, or predictions!"

    # ── No data loaded ────────────────────────────────────────────────────────
    if df is None:
        return (
            "📁 **No dataset is currently uploaded.**\n\n"
            "Upload any CSV, Excel, or JSON file in **Tab 1 (Upload Data)**. Once loaded, "
            "I can explain every column, analyze correlations, run outlier detections, and guide model tuning!"
        )

    n_rows, n_cols = df.shape
    num_cols = list(df.select_dtypes(include="number").columns)
    cat_cols = [c for c in df.columns if c not in num_cols]

    # ── Neural Network architecture / mapping question ────────────────────────
    if any(w in q for w in ["neural network", "neural net", "mlp", "architecture", "layers", "neurons", "network mapping"]):
        dense_desc = (
            f"🧠 **Neural Network Architecture (MLP)**:\n\n"
            f"1. **Input Layer**: `{len(num_cols) + len(cat_cols)}` primary feature inputs (preprocessed into standard-scaled and one-hot representations).\n"
            f"2. **Hidden Layer 1**: `64` neurons with ReLU activation function.\n"
            f"3. **Hidden Layer 2**: `32` neurons with ReLU activation function.\n"
            f"4. **Output Layer**: "
            + ("`1` neuron with linear activation for regression." if results and results.get("task") == "regression"
               else "Softmax activation mapping to target classes for classification.") + "\n\n"
            f"📌 *Tip: You can also inspect the visual interactive Neural Network Layer Mapping in **Tab 2**!*"
        )
        return dense_desc

    # ── Dataset dimensions / volume ───────────────────────────────────────────
    if any(w in q for w in ["how many rows", "size", "shape", "how big", "rows", "records", "samples", "dataset size"]):
        return (
            f"📊 **Dataset Scale & Dimensions**:\n\n"
            f"• Total Rows: **{n_rows:,}** observations\n"
            f"• Total Features: **{n_cols}** columns ({len(num_cols)} numeric, {len(cat_cols)} categorical)\n"
            f"• Total Data Points: **{n_rows * n_cols:,}** cells"
        )

    # ── Columns listing ───────────────────────────────────────────────────────
    if any(w in q for w in ["which columns", "what columns", "list columns", "features", "fields"]):
        c_list = ", ".join(f"`{c}`" for c in df.columns[:20])
        extra = f" *(+ {n_cols - 20} more)*" if n_cols > 20 else ""
        return (
            f"📑 **Dataset Columns ({n_cols} total)**:\n\n"
            f"{c_list}{extra}\n\n"
            f"• Numeric: **{len(num_cols)}**\n"
            f"• Categorical/Object: **{len(cat_cols)}**"
        )

    # ── Missing values / Data integrity ───────────────────────────────────────
    if any(w in q for w in ["missing", "null", "nan", "empty", "incomplete", "na"]):
        total_miss = int(df.isna().sum().sum())
        if total_miss == 0:
            return "✨ **Clean Data Notice**: There are **0 missing values** across your entire dataset!"
        
        miss_s = df.isna().sum()
        miss_cols = miss_s[miss_s > 0].sort_values(ascending=False)
        top_miss_lines = "\n".join(
            f"• `{c}`: **{v:,}** missing ({v/n_rows*100:.1f}%)" for c, v in miss_cols.head(5).items()
        )
        return (
            f"⚠️ **Missing Value Audit**:\n\n"
            f"Found **{total_miss:,} total nulls** across **{len(miss_cols)}** columns.\n\n"
            f"**Most Affected Columns:**\n{top_miss_lines}\n\n"
            f"💡 *All missing values are automatically imputed during preprocessing (medians for numbers, modes for categories).*"
        )

    # ── Duplicates ────────────────────────────────────────────────────────────
    if any(w in q for w in ["duplicate", "repeated", "redundant", "dupe"]):
        dups = int(df.duplicated().sum())
        if dups == 0:
            return "✅ **No duplicate rows detected.** Every single record in your dataset is unique."
        return (
            f"⚠️ **Duplicate Rows Detected**:\n\n"
            f"Found **{dups:,} duplicate rows** ({dups/n_rows*100:.2f}% of the dataset).\n"
            f"Removing identical duplicates can prevent data leakage and artificial model bias."
        )

    # ── Outliers & Extreme Values ─────────────────────────────────────────────
    if any(w in q for w in ["outlier", "extreme", "anomaly", "anomalies", "iqr"]):
        if not num_cols:
            return "No numeric columns found to compute IQR outlier fences."
        nums = df[num_cols]
        summary_outliers = []
        for col in num_cols:
            s = nums[col].dropna()
            q1, q3 = s.quantile(0.25), s.quantile(0.75)
            iqr = q3 - q1
            n_out = int(((s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)).sum())
            if n_out > 0:
                summary_outliers.append((col, n_out, n_out / len(s) * 100))
        
        if not summary_outliers:
            return "✅ **Outlier Check**: No severe IQR outliers detected across numeric columns."
        
        summary_outliers.sort(key=lambda x: x[1], reverse=True)
        lines = "\n".join(f"• `{c}`: **{cnt:,}** rows ({pct:.1f}%)" for c, cnt, pct in summary_outliers[:5])
        return (
            f"🚨 **Outlier Diagnostics (1.5 × IQR standard)**:\n\n"
            f"Columns containing notable outliers:\n{lines}\n\n"
            f"Tree-based models (Random Forest, Gradient Boosting) naturally handle these outliers well, "
            f"whereas Linear Regression and Neural Nets benefit from standard scaling (which we apply automatically)."
        )

    # ── Correlations & Associations ───────────────────────────────────────────
    if any(w in q for w in ["correlat", "relation", "association", "multicollinearity", "related", "dependency"]):
        if len(num_cols) < 2:
            return "At least 2 numeric columns are required to calculate Pearson correlation coefficients."
        corr = df[num_cols].corr().abs()
        corr_mat = corr.to_numpy(copy=True)
        np.fill_diagonal(corr_mat, 0)
        corr_clean = pd.DataFrame(corr_mat, index=corr.index, columns=corr.columns)
        stacked = corr_clean.unstack().sort_values(ascending=False)
        top_pair = stacked.index[0]
        top_val = stacked.iloc[0]
        return (
            f"🔗 **Correlation Analysis**:\n\n"
            f"• Strongest observed relationship: **`{top_pair[0]}`** ↔ **`{top_pair[1]}`** with |r| = **{top_val:.3f}**.\n\n"
            f"Explore the interactive **Correlation Matrix & Heatmap** in **Tab 3** to inspect full pairwise dependencies!"
        )

    # ── Model Results / Training Outcomes ─────────────────────────────────────
    if results is not None and any(w in q for w in ["model", "accuracy", "train", "best", "leaderboard", "f1", "r2", "rmse", "score", "performance"]):
        task = results.get("task", "classification")
        best = results.get("best_name", "")
        metrics = results.get("metrics")
        sort_col = "Accuracy" if task == "classification" else "R²"
        
        if metrics is not None and sort_col in metrics.columns:
            sorted_m = metrics.sort_values(sort_col, ascending=False)
            top_lines = "\n".join(
                f"{i+1}. **{idx}**: {sort_col} = `{row[sort_col]:.4f}`"
                for i, (idx, row) in enumerate(sorted_m.head(4).iterrows())
            )
            return (
                f"🏆 **Model Leaderboard Summary ({task.title()})**:\n\n"
                f"• Top Performer: **{best}**\n\n"
                f"**Rankings by {sort_col}:**\n{top_lines}\n\n"
                f"Head to **Tab 2 (Train Models)** for detailed confusion matrices, residual curves, and per-class metrics!"
            )

    # ── Question about a specific column in dataset ───────────────────────────
    for col in df.columns:
        if col.lower() in q:
            s = df[col]
            if pd.api.types.is_numeric_dtype(s):
                return (
                    f"📈 **Column Spotlight: `{col}`** (Numeric)\n\n"
                    f"• Mean: **{s.mean():.4g}** | Median: **{s.median():.4g}**\n"
                    f"• Min: **{s.min():.4g}** | Max: **{s.max():.4g}**\n"
                    f"• Std Dev: **{s.std():.4g}** | Skewness: **{s.skew():.2f}**\n"
                    f"• Missing: **{s.isna().sum()}** entries"
                )
            else:
                vc = s.value_counts().head(5)
                top_v = ", ".join(f"`{k}` ({v:,})" for k, v in vc.items())
                return (
                    f"🔤 **Column Spotlight: `{col}`** (Categorical)\n\n"
                    f"• Unique Categories: **{s.nunique()}**\n"
                    f"• Most Common Values: {top_v}\n"
                    f"• Missing: **{s.isna().sum()}** entries"
                )

    # ── How to improve accuracy / training guidance ───────────────────────────
    if any(w in q for w in ["improve", "better", "boost accuracy", "tune", "optimize"]):
        return (
            "🚀 **Techniques to Maximize Model Accuracy**:\n\n"
            "1. **Ensemble Models**: Random Forest & Gradient Boosting typically outperform single trees or linear models.\n"
            "2. **Feature Engineering**: Creating interaction features (e.g. ratios or differences between related numbers) often unlocks critical signal.\n"
            "3. **Address Outliers & Imbalance**: Check Tab 4 (Data Brief) for class distribution; heavily skewed targets require oversampling or class weighting.\n"
            "4. **Fine-Tuned Hyperparameters**: Try increasing tree depths or enabling cross-validation."
        )

    # ── Explaining algorithms ─────────────────────────────────────────────────
    if any(w in q for w in ["explain", "how does", "what is"]):
        if "random forest" in q:
            return (
                "🌲 **Random Forest**: An ensemble of numerous decision trees trained on random bootstrap subsets of data. "
                "Each tree votes on the final outcome, reducing variance and completely avoiding single-tree overfitting."
            )
        if "gradient boost" in q or "adaboost" in q or "boosting" in q:
            return (
                "⚡ **Boosting (AdaBoost / Gradient Boosting)**: Instead of training independent trees, boosting builds trees sequentially. "
                "Each new tree focuses on the errors made by previous iterations, producing an exceptionally strong predictor."
            )
        if "neural network" in q or "mlp" in q:
            return (
                "🧠 **Neural Network (Multi-Layer Perceptron)**: Uses interconnected layers of mathematical nodes with nonlinear activation functions. "
                "It excels at learning complex, multi-dimensional, nonlinear relationships across high-dimensional features."
            )
        if "decision tree" in q:
            return (
                "🌿 **Decision Tree**: Breaks down data by making sequential binary rule-based splits. Highly intuitive and interpretable, "
                "though individual trees can overfit if allowed to grow too deep."
            )

    # ── Comprehensive Fallback ────────────────────────────────────────────────
    return (
        f"🤖 **DataMind Intelligence System**:\n\n"
        f"I can analyze any aspect of your **{n_rows:,} row × {n_cols} column** dataset. Try asking:\n\n"
        f"• *\"Show me the neural network mapping\"*\n"
        f"• *\"Which model has the highest accuracy?\"*\n"
        f"• *\"Are there any severe outliers in the data?\"*\n"
        f"• *\"How many missing values are present?\"*\n"
        f"• *\"Which features correlate most strongly?\"*\n"
        f"• *\"Tell me about column `{df.columns[0]}`\"*"
    )
