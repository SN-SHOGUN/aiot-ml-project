"""
src/models.py
-------------
Classical ML classifiers for predictive maintenance.
"""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
import joblib, pathlib

SEED = 42
MODEL_DIR = pathlib.Path(__file__).parent.parent / "data" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_classifiers() -> dict:
    """Return a dict of {name: estimator} for all classical classifiers."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=None, random_state=SEED, n_jobs=-1
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=3, random_state=SEED
        ),
        "K-Nearest Neighbours": KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=SEED),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=SEED
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=SEED
        ),
        "Naive Bayes": GaussianNB(),
    }


def train_all(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Train all classifiers and return fitted models."""
    clfs = get_classifiers()
    fitted = {}
    for name, clf in clfs.items():
        clf.fit(X_train, y_train)
        fitted[name] = clf
    return fitted


def save_model(clf, name: str):
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    joblib.dump(clf, MODEL_DIR / f"{safe_name}.pkl")


def load_model(name: str):
    safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
    return joblib.load(MODEL_DIR / f"{safe_name}.pkl")
