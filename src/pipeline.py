"""
src/pipeline.py
---------------
Data loading and preprocessing for the AI4I 2020 Predictive Maintenance dataset.
"""
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

ROOT = pathlib.Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "ai4i2020.csv"

FEATURE_COLS = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
    "Type_encoded",
]
TARGET_COL = "Machine failure"


def load_raw(path: pathlib.Path = DATA_PATH) -> pd.DataFrame:
    """Load CSV, normalise column names, return raw DataFrame."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. "
            "Run `python scripts/gen_data.py` first."
        )
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def preprocess(df: pd.DataFrame):
    """
    Returns
    -------
    X_train, X_test, y_train, y_test : numpy arrays (scaled)
    scaler : fitted StandardScaler
    df_processed : DataFrame with engineered features
    """
    df = df.copy()
    le = LabelEncoder()
    df["Type_encoded"] = le.fit_transform(df["Type"].astype(str))

    # Feature engineering
    df["Temp_diff"] = df["Process temperature [K]"] - df["Air temperature [K]"]
    df["Power_proxy"] = df["Torque [Nm]"] * df["Rotational speed [rpm]"]
    df["Wear_rate"] = df["Tool wear [min]"] * df["Torque [Nm]"]

    extra_feats = ["Temp_diff", "Power_proxy", "Wear_rate"]
    all_feats = FEATURE_COLS + extra_feats

    X = df[all_feats].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.int32)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    return X_train, X_test, y_train, y_test, scaler, df, all_feats


def get_data(path: pathlib.Path = DATA_PATH):
    """Convenience wrapper: load → preprocess."""
    df_raw = load_raw(path)
    return preprocess(df_raw)
