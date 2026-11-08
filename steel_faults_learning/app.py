"""Streamlit learning reference: evaluate saved classifiers on labelled test data.

The training workflow lives in train.py. This app intentionally calls only
predict/predict_proba through ml_core.evaluate_model; uploaded data never fits
an estimator, scaler, or imputer.
"""

from __future__ import annotations

import csv
import io
import json
import warnings
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import streamlit as st
from sklearn.exceptions import InconsistentVersionWarning

from ml_core import (
    CLASS_NAMES,
    FEATURE_COLUMNS,
    MODEL_FILES,
    evaluate_model,
    validate_test_frame,
)


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "model"
TEST_CSV = ROOT / "test_data.csv"
MAX_CSV_BYTES = 5 * 1024 * 1024
MAX_TEST_ROWS = 10_000
METRIC_NAMES = ("Accuracy", "AUC", "Precision", "Recall", "F1", "MCC")

st.set_page_config(
    page_title="Steel Faults | Model Bench",
    page_icon="🔎",
    layout="wide",
)


def parse_test_csv(raw: bytes) -> pd.DataFrame:
    """Bound the upload size and reject ambiguous CSV column headers."""
    if not raw:
        raise ValueError("The CSV is empty. Upload a file with a header and test rows.")
    if len(raw) > MAX_CSV_BYTES:
        raise ValueError("Use a test CSV smaller than 5 MB.")
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise ValueError("Save the CSV using UTF-8 encoding and try again.") from exc
    try:
        header = next(csv.reader(io.StringIO(text)))
    except (StopIteration, csv.Error) as exc:
        raise ValueError("The CSV needs a valid header row.") from exc
    if len(header) != len(set(header)):
        raise ValueError("The CSV contains duplicate column names. Each name must be unique.")
    try:
        frame = pd.read_csv(io.StringIO(text), nrows=MAX_TEST_ROWS + 1)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise ValueError("The file is not a valid comma-separated CSV with a header.") from exc
    if len(frame) > MAX_TEST_ROWS:
        raise ValueError("Upload test data only: the limit is 10,000 rows.")
    return frame


@st.cache_resource(show_spinner=False)
def load_saved_models(artifact_fingerprint: tuple):
    """Load each saved pipeline once; file changes invalidate this cache."""
    # The fingerprint is part of Streamlit's cache key, even though loading
    # below needs only the known local paths.
    del artifact_fingerprint
    metadata_file = MODEL_DIR / "metadata.json"
    metadata = (
        json.loads(metadata_file.read_text(encoding="utf-8"))
        if metadata_file.exists()
        else {}
    )
    trained_version = metadata.get("sklearn_version")
    if trained_version and trained_version != sklearn.__version__:
        raise ValueError(
            f"These saved models use scikit-learn {trained_version}, but this app "
            f"is running {sklearn.__version__}. Install the pinned requirements.txt "
            "or regenerate the models in the deployment environment."
        )
    with warnings.catch_warnings():
        warnings.simplefilter("error", InconsistentVersionWarning)
        models = {
            name: joblib.load(MODEL_DIR / filename)
            for name, filename in MODEL_FILES.items()
        }
    return models, metadata


def metric_text(value) -> str:
    return "N/A" if value is None or not np.isfinite(value) else f"{value:.4f}"


def draw_confusion_matrix(matrix, classes, normalize: bool):
    """Rows are actual labels, columns are predictions; zero-support rows stay zero."""
    counts = np.asarray(matrix)
    if normalize:
        denominators = counts.sum(axis=1, keepdims=True)
        shown = np.divide(
            counts,
            denominators,
            out=np.zeros_like(counts, dtype=float),
            where=denominators != 0,
        )
    else:
        shown = counts
    fig, axis = plt.subplots(figsize=(8.2, 6.2))
    image = axis.imshow(shown, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    axis.set(
        xticks=np.arange(len(classes)),
        yticks=np.arange(len(classes)),
        xticklabels=classes,
        yticklabels=classes,
        xlabel="Predicted fault",
        ylabel="Actual fault",
        title="Within-class proportions" if normalize else "Test examples per cell",
    )
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    plt.setp(axis.get_yticklabels(), fontsize=9)
    midpoint = shown.max() / 2 if shown.size else 0
    for row in range(len(classes)):
        for column in range(len(classes)):
            value = shown[row, column]
            label = f"{value:.2f}" if normalize else str(int(value))
            axis.text(
                column,
                row,
                label,
                ha="center",
                va="center",
                fontsize=9,
                color="white" if value > midpoint else "#163239",
            )
    fig.tight_layout()
    return fig


st.caption("INDUSTRIAL INSPECTION  /  SEVEN FAULT CLASSES")
st.title("Steel Faults · Model Bench")
st.write(
    "Compare five classifiers on the same labelled test examples. "
    "Select a model to inspect its errors and the recall of each fault class."
)

with st.sidebar:
    st.header("Test bench")
    source = st.radio(
        "Evaluation data",
        ("Bundled held-out test set", "Upload test CSV"),
    )
    uploaded_file = None
    if source == "Upload test CSV":
        uploaded_file = st.file_uploader(
            "Labelled test CSV", type=["csv"],
            help="Include the 27 original feature columns and the target column.",
        )
    selected_model = st.selectbox("Inspect a model", list(MODEL_FILES))
    if TEST_CSV.exists():
        st.download_button(
            "Download example test CSV",
            data=TEST_CSV.read_bytes(),
            file_name="test_data.csv",
            mime="text/csv",
            use_container_width=True,
        )
    st.caption("Upload test data only. Limit: 5 MB and 10,000 rows.")
    with st.expander("CSV requirements"):
        st.write("Use the original feature names and include `target` with the true fault label.")
        st.write("**Feature columns**")
        st.code(", ".join(FEATURE_COLUMNS), language=None)
        st.write("**Allowed target labels**")
        st.code(", ".join(CLASS_NAMES), language=None)

if source == "Upload test CSV":
    if uploaded_file is None:
        st.info("Upload a labelled test CSV to calculate metrics for all five models.")
        st.stop()
    raw_data = uploaded_file.getvalue()
    data_label = "Uploaded test CSV"
else:
    if not TEST_CSV.exists():
        st.error("The example test data is missing. Run `python train.py` to generate it.")
        st.stop()
    raw_data = TEST_CSV.read_bytes()
    data_label = "Bundled held-out test set"

try:
    test_frame = parse_test_csv(raw_data)
    X_test, y_test = validate_test_frame(test_frame)
except ValueError as exc:
    st.error(str(exc))
    st.stop()

artifact_paths = [MODEL_DIR / filename for filename in MODEL_FILES.values()]
missing_models = [path.name for path in artifact_paths if not path.is_file()]
if missing_models:
    st.error(
        "Saved models are missing. Run `python train.py`, then include all five "
        "files in the repository's `model/` directory."
    )
    st.stop()
metadata_path = MODEL_DIR / "metadata.json"
if metadata_path.exists():
    artifact_paths.append(metadata_path)
fingerprint = tuple(
    (path.name, path.stat().st_mtime_ns, path.stat().st_size)
    for path in artifact_paths
)
try:
    with st.spinner("Loading saved models and evaluating test examples…"):
        saved_models, metadata = load_saved_models(fingerprint)
        results = {
            name: evaluate_model(pipeline, X_test, y_test)
            for name, pipeline in saved_models.items()
        }
except Exception as exc:
    st.error(f"The saved models could not be evaluated: {exc}")
    st.info("Check that all model files and requirements.txt come from the same training run.")
    st.stop()

overview = st.columns(4)
overview[0].metric("Test examples", f"{len(X_test):,}")
overview[1].metric("Input features", len(FEATURE_COLUMNS))
overview[2].metric("Classes present", f"{y_test.nunique()} / {len(CLASS_NAMES)}")
overview[3].metric("Saved classifiers", len(saved_models))
st.caption(f"Data: {data_label}. Scores are recalculated on these exact rows.")

selected = results[selected_model]
st.subheader(selected_model)
metric_help = {
    "Accuracy": "Fraction of test rows with the correct predicted fault.",
    "AUC": "Macro one-versus-rest ROC-AUC from class probabilities; all seven classes must be present.",
    "Precision": "Unweighted mean of class precision over all seven fault labels.",
    "Recall": "Unweighted mean of class recall over all seven fault labels.",
    "F1": "Unweighted mean of per-class F1 over all seven fault labels.",
    "MCC": "Multiclass Matthews correlation coefficient, using all confusion-matrix cells.",
}
for offset in (0, 3):
    metric_columns = st.columns(3)
    for column, name in zip(metric_columns, METRIC_NAMES[offset:offset + 3]):
        column.metric(name, metric_text(selected["metrics"][name]), help=metric_help[name])
st.caption(
    "Precision, recall and F1 use macro averaging over the same seven labels. "
    "Undefined per-class precision/recall values are set to zero. "
    "AUC is macro one-versus-rest ROC-AUC; MCC uses its multiclass definition."
)
if selected.get("auc_note"):
    st.warning(selected["auc_note"])
if metadata.get("chosen_model"):
    st.caption(f"Model selected using training cross-validation: {metadata['chosen_model']}.")

comparison_tab, diagnostics_tab, predictions_tab, data_tab = st.tabs(
    ["All five models", "Confusion and class report", "Predictions", "Test data"]
)

with comparison_tab:
    st.write("Every row below uses the same test examples and metric definitions.")
    comparison = pd.DataFrame(
        [{"Model": name, **result["metrics"]} for name, result in results.items()]
    )
    comparison_display = comparison.copy()
    for name in METRIC_NAMES:
        comparison_display[name] = comparison_display[name].map(metric_text)
    st.dataframe(comparison_display, hide_index=True, use_container_width=True)
    st.download_button(
        "Download comparison CSV",
        data=comparison.to_csv(index=False).encode("utf-8"),
        file_name="test_model_comparison.csv",
        mime="text/csv",
    )
    st.caption(
        "A model can rank faults well (high AUC) while still confusing classes at "
        "its final prediction rule. Review macro F1 and per-class recall together."
    )

with diagnostics_tab:
    normalized = st.checkbox("Show within-class proportions", value=False)
    figure = draw_confusion_matrix(
        selected["confusion_matrix"], selected["classes"], normalized
    )
    st.pyplot(figure, use_container_width=True)
    plt.close(figure)
    st.caption(
        "Rows are actual faults; columns are predictions. The diagonal counts "
        "correct predictions. Normalized rows show each true class's error pattern."
    )
    report = pd.DataFrame(
        [
            {
                "Class": label,
                "Precision": statistics["precision"],
                "Recall": statistics["recall"],
                "F1": statistics["f1-score"],
                "Support": int(statistics["support"]),
            }
            for label, statistics in selected["report"].items()
            if isinstance(statistics, dict)
        ]
    )
    st.subheader("Classification report")
    st.dataframe(report.round(4), hide_index=True, use_container_width=True)
    st.caption("Support is the number of true examples of a class in this test CSV.")

with predictions_tab:
    predictions = np.asarray(selected["predictions"])
    probability = np.asarray(selected["probabilities"])
    actual = np.asarray(y_test)
    preview = pd.DataFrame(
        {
            "CSV row": np.arange(len(actual)) + 2,
            "Actual fault": actual,
            "Predicted fault": predictions,
            "Correct": actual == predictions,
        }
    )
    errors_only = st.checkbox("Show misclassified examples only", value=False)
    shown_preview = preview.loc[~preview["Correct"]] if errors_only else preview
    st.dataframe(shown_preview.head(250), hide_index=True, use_container_width=True)
    st.caption(f"Showing up to 250 of {len(shown_preview):,} matching test rows.")
    export = X_test.reset_index(drop=True).copy()
    export["target"] = actual
    export["predicted_target"] = predictions
    export["correct"] = actual == predictions
    for index, label in enumerate(selected["classes"]):
        export[f"probability_{label}"] = probability[:, index]
    st.download_button(
        "Download all predictions and probabilities",
        data=export.to_csv(index=False).encode("utf-8"),
        file_name=f"{Path(MODEL_FILES[selected_model]).stem}_predictions.csv",
        mime="text/csv",
    )

with data_tab:
    st.subheader("Fault distribution in the current test data")
    distribution = y_test.value_counts().reindex(CLASS_NAMES, fill_value=0)
    distribution.index.name = "Fault class"
    st.bar_chart(distribution.rename("Test examples"))
    st.write("First 20 test examples")
    st.dataframe(test_frame.head(20), hide_index=True, use_container_width=True)
    st.caption(
        "Class names retain UCI's original spelling, including K_Scatch. "
        "The source's seven target indicator columns are excluded from the 27 inputs."
    )

st.divider()
st.caption(
    "Learning reference prepared outside BITS Virtual Lab. Complete and document "
    "your own BITS Lab execution for the assignment."
)
