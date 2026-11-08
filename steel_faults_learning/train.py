"""Reproducible training reference. Run your adapted implementation in BITS Lab.

All five models share an untouched 20% test split. Five-fold cross-validation
uses training data only and selects a preferred baseline by macro F1.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import time

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import accuracy_score, make_scorer, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from ml_core import CLASS_NAMES, FEATURE_COLUMNS, MODEL_FILES, METRIC_NAMES, build_models, evaluate_model
from prepare_data import prepare_dataset

ROOT = Path(__file__).resolve().parent


def run_experiment(root: Path = ROOT, seed: int = 42, cv_folds: int = 5,
                   execution_context: str = "Unspecified user environment") -> pd.DataFrame:
    root = Path(root)
    data = prepare_dataset(root)
    X, y = data[FEATURE_COLUMNS], data["target"]
    train_indices, test_indices = train_test_split(
        np.arange(len(data)), test_size=0.20, stratify=y, random_state=seed)
    X_train, X_test = X.iloc[train_indices], X.iloc[test_indices]
    y_train, y_test = y.iloc[train_indices], y.iloc[test_indices]
    if set(train_indices) & set(test_indices):
        raise AssertionError("Train/test row identities must be disjoint.")
    # Training data and uploaded test data use RAW units; preprocessing is saved in each pipeline.
    data.iloc[test_indices].to_csv(root / "test_data.csv", index=False)
    model_dir, result_dir = root / "model", root / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    splits = pd.DataFrame({"source_row": np.arange(1, len(data) + 1), "split": "train"})
    splits.loc[test_indices, "split"] = "test"
    splits.to_csv(result_dir / "split_membership.csv", index=False)
    pd.crosstab(y, splits["split"]).reindex(CLASS_NAMES).to_csv(result_dir / "class_distribution.csv")
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    scorer = make_scorer(f1_score, labels=CLASS_NAMES, average="macro", zero_division=0, pos_label=None)
    models, metric_rows, cv_rows, details = build_models(seed), [], [], {}
    print(f"Execution context: {execution_context}", flush=True)
    print(f"Dataset: {len(data)} rows; {len(FEATURE_COLUMNS)} features; {len(CLASS_NAMES)} classes", flush=True)
    print(f"Stratified split: {len(train_indices)} train / {len(test_indices)} test; seed={seed}", flush=True)
    for name, pipeline in models.items():
        started = time.perf_counter()
        print(f"Evaluating training folds and fitting {name}...", flush=True)
        folds = cross_val_score(pipeline, X_train, y_train, scoring=scorer, cv=cv,
                                n_jobs=1, error_score="raise")
        pipeline.fit(X_train, y_train)
        outcome = evaluate_model(pipeline, X_test, y_test)
        metric_rows.append({"ML Model Name": name, **outcome["metrics"]})
        cv_rows.append({"ML Model Name": name, "CV macro F1 mean": float(folds.mean()),
                        "CV macro F1 std": float(folds.std(ddof=1)),
                        "Training accuracy": float(accuracy_score(y_train, pipeline.predict(X_train))),
                        **{f"Fold {i+1}": float(s) for i, s in enumerate(folds)}})
        filename = MODEL_FILES[name]
        joblib.dump(pipeline, model_dir / filename, compress=3)
        slug = Path(filename).stem
        report_df = pd.DataFrame.from_dict(
            {label: stats for label, stats in outcome["report"].items()
             if isinstance(stats, dict)}, orient="index")
        report_df.index.name = "Class"
        report_df.to_csv(result_dir / f"{slug}_classification_report.csv")
        pd.DataFrame(outcome["confusion_matrix"], index=outcome["classes"],
                     columns=outcome["classes"]).to_csv(result_dir / f"{slug}_confusion_matrix.csv")
        predictions = pd.DataFrame({"source_row": test_indices + 1, "actual": y_test.to_numpy(),
                                    "predicted": outcome["predictions"]})
        for col, cls in enumerate(outcome["classes"]):
            predictions[f"probability_{cls}"] = outcome["probabilities"][:, col]
        predictions.to_csv(result_dir / f"{slug}_predictions.csv", index=False)
        details[name] = {
            "metrics": outcome["metrics"], "cv_macro_f1": folds.tolist(),
            "training_accuracy": cv_rows[-1]["Training accuracy"],
            "elapsed_seconds": round(time.perf_counter() - started, 3),
            "classification_report": outcome["report"],
            "confusion_matrix": outcome["confusion_matrix"].tolist(),
            "classes": outcome["classes"],
        }
    comparison = pd.DataFrame(metric_rows)[["ML Model Name"] + METRIC_NAMES]
    cv_results = pd.DataFrame(cv_rows)
    chosen_model = str(cv_results.loc[cv_results["CV macro F1 mean"].idxmax(), "ML Model Name"])
    comparison.to_csv(result_dir / "comparison.csv", index=False)
    cv_results.to_csv(result_dir / "cv_results.csv", index=False)
    (result_dir / "model_details.json").write_text(json.dumps(details, indent=2), encoding="utf-8")
    versions = {name: importlib.metadata.version(name) for name in [
        "numpy", "pandas", "scipy", "scikit-learn", "joblib", "matplotlib"]}
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "execution_context": execution_context,
        "note": "This field records a user-supplied description, not proof of BITS Lab execution. Capture genuine lab evidence separately.",
        "python_version": platform.python_version(), "sklearn_version": sklearn.__version__,
        "package_versions": versions, "seed": seed, "test_fraction": 0.20,
        "train_rows": len(train_indices), "test_rows": len(test_indices),
        "feature_columns": FEATURE_COLUMNS, "classes": CLASS_NAMES,
        "model_files": MODEL_FILES, "chosen_model": chosen_model,
        "selection_rule": f"Highest mean {cv_folds}-fold stratified training-only CV macro F1 among fixed baselines; no test tuning.",
        "metrics_convention": "P/R/F1 macro over all seven classes; AUC macro one-vs-rest from probabilities; multiclass MCC.",
        "test_data_sha256": hashlib.sha256((root / "test_data.csv").read_bytes()).hexdigest(),
        "data_manifest": json.loads((root / "data" / "dataset_manifest.json").read_text()),
        "model_parameters": {name: {k: v for k, v in pipe.get_params().items()
                                    if k.startswith("classifier__") and isinstance(v, (str, int, float, bool, type(None)))}
                             for name, pipe in models.items()},
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_figures(root, data, comparison, details)
    print("\nHeld-out test metrics (macro P/R/F1; macro OvR AUC):", flush=True)
    print(comparison.round(4).to_string(index=False), flush=True)
    print(f"\nPreferred baseline from training-only CV: {chosen_model}", flush=True)
    print(f"Saved all five pipelines, test_data.csv, and results under {root}", flush=True)
    return comparison


def save_figures(root: Path, data: pd.DataFrame, comparison: pd.DataFrame, details: dict) -> None:
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    counts = data.target.value_counts().reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(9, 4.4), layout="constrained")
    bars = ax.barh(counts.index, counts.values, color="#0d7d80")
    ax.bar_label(bars, padding=4)
    ax.set(xlabel="Number of examples", title="Steel Plates Faults: class distribution", xlim=(0, int(counts.max()) * 1.16))
    ax.grid(axis="x", alpha=0.18)
    fig.savefig(root / "results" / "class_distribution.png", dpi=170)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), layout="constrained")
    for ax, (name, result) in zip(axes.flat, details.items()):
        cm = np.array(result["confusion_matrix"])
        ax.imshow(cm, cmap="Blues")
        ax.set(xticks=range(7), yticks=range(7), xticklabels=CLASS_NAMES,
               yticklabels=CLASS_NAMES, title=name, xlabel="Predicted", ylabel="Actual")
        ax.tick_params(axis="x", labelrotation=55, labelsize=8)
        ax.tick_params(axis="y", labelsize=8)
        for r in range(7):
            for c in range(7):
                ax.text(c, r, str(cm[r, c]), ha="center", va="center", fontsize=8,
                        color="white" if cm[r, c] > cm.max() / 2 else "#23354a")
    axes.flat[-1].axis("off")
    test_rows = sum(map(sum, next(iter(details.values()))["confusion_matrix"]))
    axes.flat[-1].text(0.02, 0.7, f"Same {test_rows} held-out rows\nRows = actual class\nColumns = predicted class\n\nExecution context: model/metadata.json",
                       transform=axes.flat[-1].transAxes, va="top", fontsize=13, linespacing=1.6)
    fig.savefig(root / "results" / "confusion_matrices.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42, help="Fix before running experiments; do not choose using test results.")
    parser.add_argument("--cv-folds", type=int, default=5, choices=range(2, 11))
    parser.add_argument("--execution-context", default="Unspecified user environment",
                        help="Describe where YOU actually ran this; does not create lab evidence.")
    args = parser.parse_args()
    run_experiment(seed=args.seed, cv_folds=args.cv_folds, execution_context=args.execution_context)
