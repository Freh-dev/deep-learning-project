# AI Regulatory Risk Detection and Evidence Retrieval

**DSBA 6165: Deep Learning Project** | *Solo Project*

## Project Overview
A deep learning system that automatically classifies EU AI Act regulatory risk levels from natural-language AI system descriptions and retrieves supporting regulatory evidence. This project compares three models of increasing sophistication: a traditional machine learning baseline, a fine-tuned Transformer, and a novel rule-injected attention architecture.

## Research Question
**Can injecting regulatory rules into a Transformer's attention mechanism (CARRT) improve risk classification on EU AI Act scenarios compared to standard fine-tuned BERT?**

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
- Python 3.9+
- Install dependencies: `pip install -r requirements.txt`

## Project Timeline (Solo)
| Stage | Deliverable | Due Date | Status |
|:------|:------------|:---------|:-------|
| 1 | Team Registration | Aug 30 | ✅ Complete |
| 2 | Proposal | Sep 13 | 📝 In Progress |
| 3 | EDA Notebook | Oct 4 | ⬜ Pending |
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

