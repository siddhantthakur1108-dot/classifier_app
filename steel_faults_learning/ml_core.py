"""Shared learning reference: model definitions, CSV contract, and evaluation.

Read the explanations in ML_Assignment_2_Phased_Guide.ipynb before adapting this
code. The assignment requires your own implementation and BITS Lab execution.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity", "Minimum_of_Luminosity",
    "Maximum_of_Luminosity", "Length_of_Conveyer", "TypeOfSteel_A300",
    "TypeOfSteel_A400", "Steel_Plate_Thickness", "Edges_Index", "Empty_Index",
    "Square_Index", "Outside_X_Index", "Edges_X_Index", "Edges_Y_Index",
    "Outside_Global_Index", "LogOfAreas", "Log_X_Index", "Log_Y_Index",
    "Orientation_Index", "Luminosity_Index", "SigmoidOfAreas",
]
SOURCE_TARGET_COLUMNS = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps", "Other_Faults"
]
# Preserve the original UCI spelling "K_Scatch" in the machine-readable labels.
CLASS_NAMES = sorted(SOURCE_TARGET_COLUMNS)
TARGET_COLUMN = "target"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}
METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def build_models(seed: int = 42) -> dict[str, Pipeline]:
    """Fixed, documented baselines; settings are not tuned on the test set.

    A Pipeline fits its scaler only when the training code calls fit(). Each CV
    fold therefore learns its own training-fold mean and standard deviation.
    Tree thresholds do not need standardized feature units.
    """
    def scaled(classifier):
        return Pipeline([("scale", StandardScaler()), ("classifier", classifier)])

    return {
        "Logistic Regression": scaled(LogisticRegression(C=1.0, max_iter=5000)),
        "Decision Tree": Pipeline([("classifier", DecisionTreeClassifier(
            max_depth=10, min_samples_leaf=3, random_state=seed))]),
        "KNN": scaled(KNeighborsClassifier(n_neighbors=7, weights="distance")),
        "Naive Bayes": scaled(GaussianNB(var_smoothing=1e-9)),
        "Support Vector Machine": scaled(SVC(
            C=10.0, kernel="rbf", gamma="scale", probability=True, random_state=seed)),
    }


def validate_test_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Check the documented CSV schema and return features in training order.

    This reference uses the complete numeric UCI data. Missing/nonfinite values
    are rejected instead of silently changing the evaluation population.
    """
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("The CSV must contain at least one data row.")
    if frame.columns.duplicated().any():
        raise ValueError("The CSV contains duplicate column names.")
    required = FEATURE_COLUMNS + [TARGET_COLUMN]
    missing = [c for c in required if c not in frame.columns]
    extra = [str(c) for c in frame.columns if c not in required]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    if extra:
        raise ValueError("Unexpected columns (remove row IDs and original class indicators): "
                         + ", ".join(extra))
    try:
        X = frame.loc[:, FEATURE_COLUMNS].apply(pd.to_numeric, errors="raise").astype(float)
    except (ValueError, TypeError) as exc:
        raise ValueError("Every input feature must contain numeric values.") from exc
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Input features must be complete and finite; blanks, NaN, and infinity are invalid.")
    if frame[TARGET_COLUMN].isna().any():
        raise ValueError("Every row must have a known target label.")
    y = frame[TARGET_COLUMN].astype(str).str.strip()
    unknown = sorted(set(y) - set(CLASS_NAMES))
    if unknown:
        raise ValueError("Unknown target labels: " + ", ".join(unknown)
                         + ". Use these labels: " + ", ".join(CLASS_NAMES))
    return X, y


def evaluate_model(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict:
    """Evaluate a fitted pipeline; never fit or adjust it on uploaded/test data."""
    classes = list(pipeline.classes_)
    predicted = pipeline.predict(X)
    probabilities = pipeline.predict_proba(X)
    # Fixed label universe: the same seven classes are averaged for every CSV.
    metrics = {
        "Accuracy": float(accuracy_score(y, predicted)),
        "AUC": None,
        "Precision": float(precision_score(y, predicted, labels=classes,
                                           average="macro", zero_division=0)),
        "Recall": float(recall_score(y, predicted, labels=classes,
                                     average="macro", zero_division=0)),
        "F1": float(f1_score(y, predicted, labels=classes,
                             average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y, predicted)),
    }
    absent = sorted(set(classes) - set(y))
    auc_note = ""
    if absent:
        auc_note = ("Macro one-vs-rest AUC across all seven classes is unavailable because "
                    "the CSV has no true examples of: " + ", ".join(absent)
                    + ". Upload the complete test_data.csv to reproduce the reported AUC.")
    else:
        # Probability columns follow pipeline.classes_; predicted labels are NOT AUC scores.
        metrics["AUC"] = float(roc_auc_score(
            y, probabilities, labels=classes, multi_class="ovr", average="macro"))
    return {
        "metrics": metrics,
        "predictions": predicted,
        "probabilities": probabilities,
        "classes": classes,
        "report": classification_report(y, predicted, labels=classes,
                                         output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y, predicted, labels=classes),
        "auc_note": auc_note,
    }
