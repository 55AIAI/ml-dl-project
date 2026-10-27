"""Leakage-aware classification experiments on educational datasets."""
from dataclasses import dataclass
import csv
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.datasets import load_digits, make_classification
from sklearn.decomposition import PCA
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin",
            "BMI", "DiabetesPedigreeFunction", "Age"]
ZERO_MISSING = [1, 2, 3, 4, 5]


@dataclass
class Dataset:
    name: str
    X: np.ndarray
    y: np.ndarray
    features: list
    zero_missing: bool = False


def clean_zeros(X):
    result = np.asarray(X, dtype=float).copy()
    if result.ndim != 2 or result.shape[1] != len(FEATURES):
        raise ValueError("Diabetes input must have exactly eight features")
    for index in ZERO_MISSING:
        result[result[:, index] == 0, index] = np.nan
    return result


def load_dataset(name, csv_path=None, quick=False, seed=42):
    if name == "digits":
        data = load_digits()
        result = Dataset(name, data.data.copy(), data.target.copy(),
                         [f"pixel_{i}" for i in range(data.data.shape[1])])
    elif name == "synthetic":
        X, y = make_classification(n_samples=300, n_features=8, n_informative=5,
                                   n_redundant=1, random_state=seed)
        result = Dataset(name, X, y, [f"feature_{i}" for i in range(8)])
    elif name == "diabetes":
        path = Path(csv_path) if csv_path else Path(__file__).resolve().parents[1] / "diabetes.csv"
        with path.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.reader(stream)
            if next(reader, None) != FEATURES + ["Outcome"]:
                raise ValueError("CSV headers must match the documented Diabetes schema exactly")
            rows = list(reader)
        if not rows or any(len(row) != 9 for row in rows):
            raise ValueError("CSV must contain nonempty rows with exactly nine fields")
        frame = pd.DataFrame(rows, columns=FEATURES + ["Outcome"])
        try:
            values = frame.mask(frame.eq("")).astype(float).to_numpy()
        except (TypeError, ValueError) as error:
            raise ValueError("CSV contains a nonnumeric value") from error
        if np.isinf(values).any() or not np.isfinite(values[:, -1]).all():
            raise ValueError("CSV contains infinite values or a missing target")
        if set(np.unique(values[:, -1])) != {0, 1}:
            raise ValueError("Outcome must contain both integer classes 0 and 1")
        if np.nanmin(values[:, :-1]) < 0:
            raise ValueError("Diabetes features must be nonnegative")
        result = Dataset(name, values[:, :-1], values[:, -1].astype(int), FEATURES.copy(), True)
    else:
        raise ValueError("Unknown dataset")
    classes, counts = np.unique(result.y, return_counts=True)
    if counts.min() < 20:
        raise ValueError("Each class needs at least 20 rows for the split/CV protocol")
    if quick and len(result.y) > 400:
        indices, _ = train_test_split(np.arange(len(result.y)), train_size=400,
                                      stratify=result.y, random_state=seed)
        result = Dataset(result.name, result.X[indices], result.y[indices],
                         result.features, result.zero_missing)
        subset_classes, subset_counts = np.unique(result.y, return_counts=True)
        if len(subset_classes) != len(classes) or subset_counts.min() < 20:
            raise ValueError("Quick subset has too few rows per class; run without --quick")
    return result


def split_indices(y, seed=42):
    train, rest = train_test_split(np.arange(len(y)), test_size=.30,
                                   stratify=y, random_state=seed)
    validation, test = train_test_split(rest, test_size=.50,
                                        stratify=y[rest], random_state=seed)
    return train, validation, test


def make_pipeline(model, zero_missing=False, pca=None):
    steps = []
    if zero_missing:
        steps.append(("zero_missing", FunctionTransformer(clean_zeros)))
    steps.extend([("imputer", SimpleImputer(strategy="median", keep_empty_features=True)),
                  ("scaler", StandardScaler())])
    if pca is not None:
        steps.append(("pca", PCA(n_components=pca, svd_solver="full")))
    return Pipeline(steps + [("model", model)])


def model_configs(seed=42, quick=False):
    return {
        "KNN": (KNeighborsClassifier(), {"model__n_neighbors": [3] if quick else [3, 7, 11]}),
        "Random Forest": (RandomForestClassifier(n_estimators=40 if quick else 100,
                                                  random_state=seed, n_jobs=1),
                          {"model__max_depth": [8] if quick else [None, 8]}),
        "SVM": (SVC(), {"model__C": [1.] if quick else [.1, 1., 10.]}),
        "Logistic Regression": (LogisticRegression(max_iter=2000, random_state=seed),
                                {"model__C": [1.] if quick else [.1, 1., 10.]})
    }


def test_metrics(model, X, y):
    predicted = model.predict(X)
    labels = model.classes_
    return {"accuracy": float(accuracy_score(y, predicted)),
            "f1_weighted": float(f1_score(y, predicted, average="weighted", zero_division=0)),
            "labels": labels.tolist(),
            "confusion_matrix": confusion_matrix(y, predicted, labels=labels).tolist(),
            "classification_report": classification_report(
                y, predicted, labels=labels, output_dict=True, zero_division=0)}


def benchmark(data, seed=42, quick=False):
    train, validation, test = split_indices(data.y, seed)
    cv = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=seed)
    candidates, fitted = [], {}
    for name, (model, params) in model_configs(seed, quick).items():
        search = GridSearchCV(make_pipeline(model, data.zero_missing), params, cv=cv,
                              scoring="f1_weighted", n_jobs=1, error_score="raise")
        started = time.perf_counter()
        search.fit(data.X[train], data.y[train])
        elapsed = time.perf_counter() - started
        predicted = search.predict(data.X[validation])
        candidates.append({"model": name, "cv_f1_mean": float(search.best_score_),
                           "cv_f1_std": float(search.cv_results_["std_test_score"][search.best_index_]),
                           "validation_f1": float(f1_score(data.y[validation], predicted,
                                                          average="weighted", zero_division=0)),
                           "validation_accuracy": float(accuracy_score(data.y[validation], predicted)),
                           "search_seconds": elapsed, "params": search.best_params_})
        fitted[name] = search.best_estimator_
    # Stable ties follow the declared model order. Test data cannot select the winner.
    winner = max(candidates, key=lambda row: row["validation_f1"])["model"]
    development = np.concatenate([train, validation])
    final = clone(fitted[winner]).fit(data.X[development], data.y[development])
    dummy = make_pipeline(DummyClassifier(strategy="most_frequent"), data.zero_missing)
    dummy.fit(data.X[development], data.y[development])
    result = {"schema_version": 1, "dataset": data.name, "mode": "quick" if quick else "full",
              "seed": seed, "samples": len(data.y), "features": data.features,
              "split_sizes": {"train": len(train), "validation": len(validation), "test": len(test)},
              "cv_folds": cv.n_splits, "selection_metric": "validation_f1_weighted",
              "candidates": candidates, "selected_model": winner,
              "test": test_metrics(final, data.X[test], data.y[test]),
              "baseline_test": test_metrics(dummy, data.X[test], data.y[test]),
              "environment": {"python": platform.python_version(), "numpy": np.__version__,
                              "pandas": pd.__version__, "scikit_learn": sklearn.__version__}}
    return result
