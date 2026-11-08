"""Convert the public UCI source into 27 inputs plus one categorical target.

The original archive is bundled for an offline BITS Lab run. If it is missing,
this script downloads the same archive directly from UCI, not another project.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import urllib.request
from zipfile import ZipFile

import pandas as pd

from ml_core import FEATURE_COLUMNS, SOURCE_TARGET_COLUMNS, validate_test_frame

ROOT = Path(__file__).resolve().parent
DATASET_URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"
DATASET_PAGE = "https://archive.ics.uci.edu/dataset/198/steel+plates+faults"


def prepare_dataset(root: Path = ROOT) -> pd.DataFrame:
    data_dir = Path(root) / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    archive = data_dir / "steel_plates_faults_source.zip"
    if not archive.exists():
        request = urllib.request.Request(DATASET_URL, headers={"User-Agent": "ML-learning-project/1.0"})
        with urllib.request.urlopen(request, timeout=30) as response:
            archive.write_bytes(response.read())
    with ZipFile(archive) as z:
        columns = z.read("Faults27x7_var").decode("utf-8").split()
        raw = pd.read_csv(io.BytesIO(z.read("Faults.NNA")), sep=r"\s+", header=None, names=columns)
    if columns != FEATURE_COLUMNS + SOURCE_TARGET_COLUMNS:
        raise ValueError("The UCI source schema differs from the documented 27 + 7 layout.")
    indicators = raw[SOURCE_TARGET_COLUMNS]
    if not indicators.isin([0, 1]).all().all() or not indicators.sum(axis=1).eq(1).all():
        raise ValueError("Each row must have exactly one active fault indicator.")
    data = raw[FEATURE_COLUMNS].copy()
    data["target"] = indicators.idxmax(axis=1)
    validate_test_frame(data)
    if len(data) < 500 or len(FEATURE_COLUMNS) < 12:
        raise ValueError("The assignment's dataset-size requirements are not met.")
    csv_path = data_dir / "steel_faults.csv"
    data.to_csv(csv_path, index=False)
    manifest = {
        "dataset": "Steel Plates Faults", "source_url": DATASET_PAGE,
        "download_url": DATASET_URL, "doi": "10.24432/C5J88N",
        "attribution": "Buscema, M., Terzi, S., & Tastle, W. (2010). Steel Plates Faults. UCI Machine Learning Repository.",
        "license": "CC BY 4.0", "rows": len(data), "input_features": len(FEATURE_COLUMNS),
        "classes": sorted(data.target.unique().tolist()),
        "class_counts": {str(k): int(v) for k, v in data.target.value_counts().sort_index().items()},
        "missing_cells": int(data.isna().sum().sum()),
        "duplicate_full_rows": int(data.duplicated().sum()),
        "duplicate_feature_rows": int(data[FEATURE_COLUMNS].duplicated().sum()),
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "csv_sha256": hashlib.sha256(csv_path.read_bytes()).hexdigest(),
        "transformation": "Convert seven mutually exclusive indicators to target using idxmax; remove all seven indicator columns from X.",
    }
    (data_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return data


if __name__ == "__main__":
    df = prepare_dataset()
    print(f"Prepared {len(df)} rows, {len(FEATURE_COLUMNS)} input features, {df.target.nunique()} classes.")
    print(df.target.value_counts().sort_index().to_string())
