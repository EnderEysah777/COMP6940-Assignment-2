# COMP6940 Assignment 2 — Big Data and Data Visualization

Course project for **COMP6940** : two-part pipeline covering GPU-accelerated distribution fitting and a full credit-risk modelling workflow with interpretability and fairness.

## What this repository contains

- **Part 1 — GPU Weibull fitting** (`part1_gpu_weibull/`): NASA C-MAPSS turbofan run-to-failure data; time-to-failure extraction; Weibull MLE with **JAX** (gradient-based optimisation) compared with a CPU baseline; survival-style plots and maintenance-oriented discussion.
- **Part 2 — Classification and interpretability** (`part2_classification/`): **Home Credit Default Risk** data ingested and cleaned in `01_ingest_clean.ipynb`; `02_model_interpret.ipynb` trains models (including **XGBoost**), evaluates on a held-out test set, simulates **profit vs. approval threshold**, runs **SHAP** and **DiCE** counterfactuals on the best model, and reports **fairness** metrics plus a small mitigation experiment.

Raw data lives under `data/raw/`; curated outputs (e.g. parquet) under `data/curated/`. Use `requirements.txt` or `environment.yml` (if present) and run the notebooks in order for reproducibility.


Group Members:

- Dillon Carl
- Elena Panchoo
- Eysah Ali
- Zuhrah Mohammed 