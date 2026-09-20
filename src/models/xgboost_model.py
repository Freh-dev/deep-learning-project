import json
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score
from xgboost import XGBClassifier

from src.data.preprocess import EXPECTED_LABELS, load_scenarios, split_scenarios


def train_baseline(output_dir="results/xgboost"):
    splits = split_scenarios(load_scenarios())
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        stop_words="english",
        lowercase=True,
    )
    train_features = vectorizer.fit_transform(splits["train"]["text"])

    model = XGBClassifier(
        objective="multi:softprob",
        num_class=len(EXPECTED_LABELS),
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
    )
    model.fit(train_features, splits["train"]["label"])

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    metrics = {}
    for split_name in ("validation", "test"):
        split = splits[split_name]
        predictions = model.predict(vectorizer.transform(split["text"])).astype(int)
        metrics[split_name] = {
            "accuracy": accuracy_score(split["label"], predictions),
            "macro_f1": f1_score(
                split["label"], predictions, average="macro", labels=range(len(EXPECTED_LABELS))
            ),
            "classification_report": classification_report(
                split["label"],
                predictions,
                labels=range(len(EXPECTED_LABELS)),
                target_names=EXPECTED_LABELS,
                output_dict=True,
                zero_division=0,
            ),
        }

    with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


if __name__ == "__main__":
    train_baseline()
