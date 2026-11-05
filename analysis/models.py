"""The four classifiers plus a baseline, built exactly to the spec in AGENT_BRIEF.md.

Each function returns a fresh, unfitted pipeline. All preprocessing (encoding, imputing,
scaling) is a pipeline step, so it is learned from the training rows only.
analysis/evaluate.py and analysis/__main__.py fit, score and export these; this module only builds them.
"""

from __future__ import annotations

from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from analysis.data import encode


def baseline():
    """Always predicts the training prevalence; the floor every real model must beat."""
    return make_pipeline(DummyClassifier(strategy="prior"))


def logreg():
    return make_pipeline(
        encode(),
        SimpleImputer(strategy="median"),
        StandardScaler(),
        LogisticRegression(max_iter=5000),
    )


def knn():
    return make_pipeline(
        encode(),
        SimpleImputer(strategy="median"),
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=25),
    )


def rf():
    return make_pipeline(
        encode(),
        SimpleImputer(strategy="median"),
        RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1),
    )


def lgbm():
    # LightGBM handles missing values itself, so there is no imputer or scaler.
    return make_pipeline(
        encode(),
        LGBMClassifier(random_state=0, verbose=-1, n_jobs=8),
    )


MODELS = {"baseline": baseline, "logreg": logreg, "knn": knn, "rf": rf, "lgbm": lgbm}
