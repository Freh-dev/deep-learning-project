# Weekly Check-In 01

## Current Goal
Prepare the dataset and create comparable baseline and BERT experiments before starting CARRT.

## Completed

### 1. Dataset exploration
- Confirmed the repository uses `data/raw/scenarios.json`.
- Confirmed 339 labeled scenarios.
- Confirmed four labels: `high-risk`, `limited`, `minimal`, and `prohibited`.
- Checked required fields, missing values, duplicate model inputs, and text lengths.
- No missing required values or duplicate model inputs were found.

### 2. Label distribution

| Label | Count |
|---|---:|
| minimal | 99 |
| high-risk | 86 |
| limited | 84 |
| prohibited | 70 |

The classes are not perfectly balanced, so accuracy will not be the only metric. Macro-F1 and per-class recall will also be reported.

### 3. Text preprocessing
The model text uses only:

- `intended_use`
- `system_type`
- `input_data`
- `domain`

The preprocessing does the following:

- normalizes Unicode text;
- removes extra whitespace;
- normalizes label names;
- checks the expected schema and labels;
- does not use `obligations` or `related_articles` as input, avoiding target leakage.

No numeric normalization is needed because the input is text.

### 4. Data pipeline
The same stratified split is used for both models:

- Training: 237 scenarios
- Validation: 51 scenarios
- Test: 51 scenarios

The test set is kept separate until final evaluation.

### 5. Baseline progress
The TF-IDF + XGBoost baseline runs successfully using the shared pipeline.

Current held-out test result:

- Accuracy: 0.8431
- Macro-F1: 0.8320

### 6. BERT pipeline progress
The BERT pipeline is ready for training:

- `bert-base-uncased`
- maximum sequence length: 64 tokens
- batched tokenization
- dynamic padding
- fixed random seed
- validation macro-F1 for checkpoint selection
- test set evaluated only after training

The tokenization smoke test passed for all 339 examples.

## Not Yet Completed

- Run the corrected BERT fine-tuning experiment.
- Compare BERT and XGBoost using the same test split and metrics.
- Update the final results table with the new comparable scores.
- Design and implement CARRT after the BERT baseline is understood.

## Next Steps

1. Run BERT fine-tuning.
2. Save the validation and test metrics.
3. Compare accuracy, macro-F1, and per-class recall with XGBoost.
4. Discuss whether the BERT result supports moving to CARRT.

## Course Scope
This work stays within the course topics. XGBoost is the traditional baseline, BERT belongs to embeddings and attention/Transformers, and CARRT will be a focused rule-aware extension of BERT. No additional CNN, RNN, or generative model is needed.
