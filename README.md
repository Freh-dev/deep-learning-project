# AI Regulatory Risk Detection and Evidence Retrieval

**DSBA 6165: Deep Learning Project** | *Solo Project*

## Project Overview
A deep learning system that automatically classifies EU AI Act regulatory risk levels from natural-language AI system descriptions and retrieves supporting regulatory evidence. This project compares three models of increasing sophistication: a traditional machine learning baseline, a fine-tuned Transformer, and a novel rule-injected attention architecture.

## Research Question
**Can injecting regulatory rules into a Transformer's attention mechanism (CARRT) improve risk classification on EU AI Act scenarios compared to standard fine-tuned BERT?**

## Course Scope
This project stays within the course topics. The TF-IDF + XGBoost model is the baseline, BERT uses the course material on embeddings and attention/Transformers, and CARRT is a focused rule-aware extension of the BERT attention idea. No CNN, RNN, generative model, retrieval system, or unrelated architecture is required for the classification experiment.

## Dataset
- **AI Act Evaluation Benchmark** (Davvetas et al., 2026)
- **Contents:** 339 labeled scenarios (Prohibited: 70, High-Risk: 86, Limited: 84, Minimal: 99) + 137 QA pairs
- **Complemented with:** Full EU AI Act legal text (933 pre-processed chunks for evidence retrieval)
- **License:** CC-BY 4.0
- [View Paper](https://huggingface.co/papers/2603.09435) | [View Dataset](https://github.com/davidath/ai-act-evaluation-benchmark)

## Models

| Model | Description | Status |
|:------|:------------|:-------|
| **Model 1 (Baseline)** | TF-IDF + XGBoost | ✅ **Complete** (80.88% accuracy) |
| **Model 2 (Deep Learning)** | Fine-tuned BERT | 🔄 **In Progress** |
| **Model 3 (Improvement)** | CARRT (Constraint-Aware Regulatory Risk Transformer) | 📝 **Planned** |

## Repository Structure
├── data/

    ├── raw/ # Original datasets (scenarios.json, qa_pairs.json)
  
    ├── processed/ # Cleaned/preprocessed data

├── docs/ # Documentation and notes

├── models/ # Saved model checkpoints

├── notebooks/ # Jupyter notebooks (EDA, baseline, final)

├── presentations/ # In-class presentation slides

├── reports/ # Proposal and final report PDFs

├── results/ # Experiment outputs and figures

└── src/
    
    ├── data/ # Data loading scripts
  
    ├── models/ # Model definitions
  
    ├── training/ # Training and evaluation loops
  
    └── utils/ # Helper functions and utilities

## Requirements
- Python 3.11 recommended
- Create the environment: `py -3.11 -m venv .venv`
- Activate it in PowerShell: `.\.venv\Scripts\Activate.ps1`
- Install dependencies: `python -m pip install -r requirements.txt`

## Reproducible Workflow
Run commands from the repository root with the project interpreter:

```powershell
.\.venv\Scripts\python.exe -m src.data.load_benchmark
.\.venv\Scripts\python.exe -m src.models.xgboost_model
.\.venv\Scripts\python.exe -m src.models.bert_model --tokenize-only
.\.venv\Scripts\python.exe -m src.models.bert_model
```

The shared preprocessing code in `src/data/preprocess.py` validates the 339 labeled scenarios, applies conservative Unicode and whitespace normalization, creates the model text from `intended_use`, `system_type`, `input_data`, and `domain`, and produces deterministic stratified train/validation/test splits. The test split is reserved for final evaluation.

Start the exploratory analysis with `notebooks/01_eda_data_preparation.ipynb`. The TF-IDF/XGBoost baseline and BERT pipeline use the same normalized inputs and label mapping. BERT uses batched tokenization, dynamic padding, `max_length=64`, and validation macro-F1 for checkpoint selection.

## Project Timeline (Solo)
| Stage | Deliverable | Due Date | Status |
|:------|:------------|:---------|:-------|
| 1 | Team Registration | Aug 30 | ✅ Complete |
| 2 | Proposal | Sep 13 | ✅ Complete |
| 3 | EDA Notebook | Oct 4 | 📝 In Progress|
| 4 | Baseline Model | Nov 1 | ⬜ Pending |
| 5 | Final Model + Presentation | Nov 22 | ⬜ Pending |
| 6 | Written Report | Dec 6 | ⬜ Pending |

## Key Results (So Far)
| Model | Accuracy | High-Risk F1 |
|:------|:---------|:-------------|
| XGBoost (Baseline) | **80.88%** | **0.744** |

## License
MIT License

## References
- Davvetas, A., Papademas, M., Ziouvelou, X., & Karkaletsis, V. (2026). AI Act Evaluation Benchmark: An Open, Transparent, and Reproducible Evaluation Dataset for NLP and RAG Systems. *arXiv:2603.09435*.
- European Union. (2024). Regulation (EU) 2024/1689 (Artificial Intelligence Act).
- Meghdadi, N., & Inkpen, D. (2024). Transformer-based Classification for Legal NLP. In *NLLP Workshop 2024*.
- Ali, F. A. R., et al. (2025). Hybrid Rule-Based Systems for Regulatory Compliance. In *ICCR 2025*.

