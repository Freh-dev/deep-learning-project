import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from datasets import Dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)
from torch.nn import CrossEntropyLoss

from src.data.preprocess import EXPECTED_LABELS, get_class_weights, load_scenarios, split_scenarios

MODEL_NAME = "bert-base-uncased"
MAX_LENGTH = 64
SEED = 42


def set_reproducible_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    set_seed(seed)


def tokenize_splits(tokenizer, splits, max_length=MAX_LENGTH):
    tokenized = {}
    for name, dataframe in splits.items():
        dataset = Dataset.from_dict(
            {
                "text": dataframe["text"].tolist(),
                "labels": dataframe["label"].tolist(),
            }
        )

        def tokenize_batch(batch):
            return tokenizer(
                batch["text"],
                truncation=True,
                max_length=max_length,
            )

        tokenized[name] = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=["text"],
            desc=f"Tokenizing {name}",
        )
    return tokenized


def compute_metrics(eval_prediction):
    predictions = eval_prediction.predictions
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    predicted_labels = np.argmax(predictions, axis=1)
    labels = eval_prediction.label_ids
    return {
        "accuracy": accuracy_score(labels, predicted_labels),
        "macro_f1": f1_score(labels, predicted_labels, average="macro"),
        "weighted_f1": f1_score(labels, predicted_labels, average="weighted"),
    }


class WeightedTrainer(Trainer):
    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weights = self.class_weights.to(outputs.logits.device)
        loss = CrossEntropyLoss(weight=weights)(outputs.logits, labels)
        return (loss, outputs) if return_outputs else loss


def train_bert(output_dir="results/bert", model_name=MODEL_NAME,
               max_length=MAX_LENGTH, seed=SEED):
    # The test set is used only after training is complete.
    set_reproducible_seed(seed)
    splits = split_scenarios(load_scenarios(), random_state=seed)
    class_weights = get_class_weights(splits["train"])
    label_to_id = {label: index for index, label in enumerate(EXPECTED_LABELS)}
    id_to_label = {index: label for label, index in label_to_id.items()}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized = tokenize_splits(tokenizer, splits, max_length=max_length)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EXPECTED_LABELS),
        id2label=id_to_label,
        label2id=label_to_id,
        problem_type="single_label_classification",
    )

    output_path = Path(output_dir)
    training_args = TrainingArguments(
        output_dir=str(output_path),
        num_train_epochs=3,
        learning_rate=2e-5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_strategy="steps",
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=seed,
        data_seed=seed,
        dataloader_num_workers=0,
    )
    trainer = WeightedTrainer(
        class_weights=class_weights,
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    test_result = trainer.evaluate(tokenized["test"], metric_key_prefix="test")
    predictions = trainer.predict(tokenized["test"])
    predicted_labels = np.argmax(predictions.predictions, axis=1)
    test_labels = splits["test"]["label"].to_numpy()
    metrics = {
        "test": test_result,
        "classification_report": classification_report(
            test_labels,
            predicted_labels,
            labels=range(len(EXPECTED_LABELS)),
            target_names=EXPECTED_LABELS,
            output_dict=True,
            zero_division=0,
        ),
        "label_to_id": label_to_id,
        "model_name": model_name,
        "max_length": max_length,
        "seed": seed,
        "class_weights": dict(zip(EXPECTED_LABELS, class_weights.tolist())),
    }
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path)
    with (output_path / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/bert")
    parser.add_argument("--model-name", default=MODEL_NAME)
    parser.add_argument("--max-length", type=int, default=MAX_LENGTH)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--tokenize-only",
        action="store_true",
        help="Validate loading and tokenization without downloading model weights or training.",
    )
    args = parser.parse_args()
    if args.tokenize_only:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        splits = split_scenarios(load_scenarios(), random_state=args.seed)
        tokenized = tokenize_splits(tokenizer, splits, max_length=args.max_length)
        print({name: len(dataset) for name, dataset in tokenized.items()})
    else:
        train_bert(args.output_dir, args.model_name, args.max_length, args.seed)


if __name__ == "__main__":
    main()
