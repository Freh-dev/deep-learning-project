[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
# AI Regulatory Risk Detection and Evidence Retrieval

## Project Overview
A deep learning system that automatically classifies EU AI Act regulatory risk levels from natural-language AI system descriptions and retrieves supporting regulatory evidence.

## Research Question
**Can a fine-tuned Transformer improve risk classification on EU AI Act scenarios without relying on a full Retrieval-Augmented Generation (RAG) pipeline?**

## Dataset
- **AI Act Evaluation Benchmark** (Davvetas et al., 2026)
- Expert-labeled risk classifications (Prohibited, High, Limited, Minimal)
- Includes supporting regulatory article references
- [View Dataset](https://huggingface.co/papers/2603.09435)

## Models

| Model | Description |
|:------|:------------|
| **Baseline** | TF-IDF + XGBoost |
| **Deep Learning** | Fine-tuned BERT/DistilBERT |
| **CARRT** | Constraint-Aware Regulatory Risk Transformer (novel architecture) |

## Repository Structure

├── data/ # Dataset (raw and processed)

├── notebooks/ # Jupyter notebooks (EDA, baseline, final)

├── src/ # Source code

├── models/ # Saved model checkpoints

├── reports/ # Proposal and final report

├── presentations/ # Final presentation

└── results/ # Experiment outputs and figures

## Requirements
Python 3.9+ with dependencies listed in `requirements.txt`

## Project Timeline
| Stage | Deliverable | Due Date |
|:------|:------------|:---------|
| 1 | Team Registration | Aug 30 |
| 2 | Proposal | Sep 13 |
| 3 | EDA Notebook | Oct 4 |
| 4 | Baseline Model | Nov 1 |
| 5 | Final Model + Presentation | Nov 22 |
| 6 | Written Report | Dec 6 |

## License
MIT License
