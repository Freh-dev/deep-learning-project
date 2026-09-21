import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

TEXT_FIELDS = ("intended_use", "system_type", "input_data", "domain")
EXPECTED_LABELS = ("high-risk", "limited", "minimal", "prohibited")
REQUIRED_FIELDS = (
    "role",
    *TEXT_FIELDS,
    "related_articles",
    "obligations",
    "risk_level",
)
DEFAULT_DATA_PATH = Path(__file__).resolve().parents[2] / "data" / "raw" / "scenarios.json"


def normalize_text(value):
    text = unicodedata.normalize("NFKC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def build_feature_text(row):
    return " ".join(normalize_text(row[field]) for field in TEXT_FIELDS)


def validate_dataframe(dataframe):
    missing_fields = [field for field in REQUIRED_FIELDS if field not in dataframe.columns]
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")

    if dataframe.empty:
        raise ValueError("The scenario dataset is empty.")

    for field in (*TEXT_FIELDS, "risk_level"):
        if dataframe[field].isna().any():
            raise ValueError(f"Field contains missing values: {field}")

    labels = set(dataframe["risk_level"])
    unexpected_labels = labels.difference(EXPECTED_LABELS)
    if unexpected_labels:
        raise ValueError(f"Unexpected risk labels: {sorted(unexpected_labels)}")

    if dataframe["text"].duplicated().any():
        duplicate_count = int(dataframe["text"].duplicated().sum())
        raise ValueError(f"Found {duplicate_count} duplicate model inputs.")


def load_scenarios(path=DEFAULT_DATA_PATH):
    data_path = Path(path)
    with data_path.open("r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise ValueError("Expected a JSON object containing a 'data' list.")

    dataframe = pd.DataFrame(payload["data"]).copy()
    for field in TEXT_FIELDS:
        dataframe[field] = dataframe[field].map(normalize_text)
    dataframe["risk_level"] = dataframe["risk_level"].map(
        lambda value: normalize_text(value).lower()
    )
    dataframe["text"] = dataframe.apply(build_feature_text, axis=1)
    label_to_id = {label: index for index, label in enumerate(EXPECTED_LABELS)}
    dataframe["label"] = dataframe["risk_level"].map(label_to_id)
    validate_dataframe(dataframe)
    return dataframe


def split_scenarios(dataframe, test_size=0.15, validation_size=0.15, random_state=42):
    if not 0 < test_size < 1 or not 0 < validation_size < 1:
        raise ValueError("Split sizes must be between zero and one.")
    if test_size + validation_size >= 1:
        raise ValueError("Test and validation sizes must leave training data.")

    indices = dataframe.index.to_numpy()
    train_indices, remainder_indices = train_test_split(
        indices,
        test_size=test_size + validation_size,
        random_state=random_state,
        stratify=dataframe["label"],
    )
    remainder = dataframe.loc[remainder_indices]
    validation_fraction = validation_size / (test_size + validation_size)
    validation_indices, test_indices = train_test_split(
        remainder.index.to_numpy(),
        test_size=1 - validation_fraction,
        random_state=random_state,
        stratify=remainder["label"],
    )
    return {
        "train": dataframe.loc[train_indices].copy(),
        "validation": dataframe.loc[validation_indices].copy(),
        "test": dataframe.loc[test_indices].copy(),
    }


def get_class_weights(dataframe):
    """Return balanced weights in the fixed EXPECTED_LABELS order."""
    counts = dataframe["label"].value_counts().reindex(range(len(EXPECTED_LABELS)))
    return np.array(len(dataframe) / (len(EXPECTED_LABELS) * counts), dtype="float32")