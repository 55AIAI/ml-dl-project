# ML & Deep Learning Experiment Lab

[![Experiment checks](https://github.com/55AIAI/ml-dl-project/actions/workflows/checks.yml/badge.svg)](https://github.com/55AIAI/ml-dl-project/actions/workflows/checks.yml)

A course project covering classification, dimensionality reduction, MLPs, CNNs and
transfer learning. Six educational notebooks now share tested experiment helpers
and generate their own results. A lightweight CLI works without TensorFlow or Jupyter.

## Quick start

Python 3.11-3.13 for classical experiments:

```bash
git clone https://github.com/55AIAI/ml-dl-project.git
cd ml-dl-project
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# Linux/macOS: source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py --dataset digits --quick --output output/digits-quick
```

The command produces a JSON report, a development-results CSV, and a Markdown summary.
It compares KNN, Random Forest, SVM and Logistic Regression. A validation set selects
the model family; the final test report includes that model and a most-frequent baseline.
All values are computed during this run.

`--quick` uses at most 400 stratified samples, 3 CV folds and one configuration per
model. Omit it for the full dataset, 5 folds and the documented grids in
`ml_lab/experiments.py`. Quick results are not full-dataset benchmarks.

Other examples:

```bash
python run.py --dataset synthetic --quick --output output/synthetic
python run.py --dataset diabetes --output output/diabetes
python run.py --dataset diabetes --csv my-data.csv --seed 123 --output output/custom
```

Choose a new output directory for each run; existing directories are refused.

## What was corrected

- Median imputation and scaling in Task 2 were outside the CV loop. They now run
  inside each fold through a scikit-learn Pipeline.
- Task 3 previously fitted PCA on all samples before cross-validation. Classification
  now receives raw training rows and fits its own imputer, scaler and PCA per fold.
- Task 4 imputed before splitting. It now splits raw rows first and fits preprocessing
  only on training rows.
- Task 5 augmentation respects the normalized [0, 1] pixel range, output directories
  exist before checkpointing, and BatchNorm stays in inference mode during fine-tuning.
- Task 6 no longer uses manually entered accuracy or training-time arrays. It collects
  current JSON reports, with explicit dataset, mode, protocol, stage and metric labels.
- Stored notebook outputs and outdated hand-entered CSV summaries were removed from
  the current version. Old numbers are not evidence for the corrected pipeline.
- Colab-only download cells and shell ZIP commands were replaced with portable Python.

## Evaluation protocol

1. Make a deterministic, stratified 70/15/15 train/validation/test split.
2. Tune each classical model on **raw training rows**, with preprocessing inside CV.
3. Select the model family by **validation weighted F1**; ties use declared model order.
4. Refit that selected pipeline on train + validation.
5. Evaluate the selected model and baseline on the untouched test rows.

The best CV score is a tuning score, not an unbiased estimate after hyperparameter
selection. The held-out test set must not guide later changes. Random stratification
assumes independent rows; grouped patients, duplicate records or time-dependent data
need a different split protocol. This lab does not solve every possible source of leakage.

Task 3 uses training-partition-only CV accuracy for a fixed RF/PCA comparison.
Task 4 declares ReLU+Adam as its primary model before test evaluation. Task 5 compares
two predeclared architectures. Neural networks train on the training partition and
use validation for early stopping; their protocol differs from the classical refit.
The summary retains these distinctions and does not invent scores for missing runs.

## Notebooks

```bash
# Portable execution; starts a clean process per notebook:
python run_notebooks.py --tasks 1 2 3 6 --smoke

# Interactive notebooks:
python -m pip install -r requirements-notebooks.txt
jupyter lab
```

Open notebooks from the repository root so that local imports and dataset paths resolve.
The runner executes ordinary Python cells; it does not emulate IPython magic or produce
executed .ipynb copies. Figures and result files are written locally.

| Task | Topic | Dependencies |
|---|---|---|
| 1 | Classification theory and decision boundaries | Core |
| 2 | Tuned classifiers and held-out evaluation | Core |
| 3 | PCA, t-SNE, LDA, optional UMAP | Core; optional umap-learn |
| 4 | MLP, activation and optimizer comparisons | TensorFlow |
| 5 | Custom CNN and MobileNetV2 fine-tuning | TensorFlow |
| 6 | Survey and current generated result summary | Core; run earlier tasks first |

Task 6 reads available generated reports only. It can summarize a partial run.
Before switching between quick/smoke and full runs, move the old `results/` directory
aside if you want the summary to contain only one session's reports. Modes are labelled,
but files from separate sessions are not automatically isolated.

## Deep-learning checks and full training

TensorFlow is optional; its tested smoke environment is Python 3.11 on Linux CPU.

```bash
python -m pip install -r requirements-deep.txt
python run_notebooks.py --tasks 4 5 6 --smoke
# Full training (potentially long, and may download CIFAR-10 / ImageNet weights):
python run_notebooks.py --tasks 4 5 6
```

Deep smoke mode uses synthetic tabular data/images, one training epoch per stage and
random MobileNetV2 weights. It checks preprocessing, forward/backward passes, fine-tuning,
Keras model save/load and report generation. **Its scores do not measure Diabetes or
CIFAR-10 performance.** Full convergence and GPU performance are not verified by CI.

UMAP is optional: `python -m pip install umap-learn`. PCA/t-SNE/LDA run without it.
Seed 42 is recorded; deep-learning results can still vary by library/device.

## Dataset scope

- Digits is bundled with scikit-learn and requires no data download.
- `diabetes.csv` is the already included Pima Diabetes educational table. This project
  is not a diagnostic system; results should not be generalized to clinical populations.
- Synthetic examples are generated from fixed seeds.
- Full Task 5 downloads CIFAR-10 and optional ImageNet weights through Keras.

The optional Diabetes CSV must have exactly these columns in order:

```text
Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome
```

Features must be numeric/nonnegative or blank (missing); infinity is rejected.
Outcome must contain both integer classes 0 and 1, with at least 20 rows per class.
Zeros in Glucose, BloodPressure, SkinThickness, Insulin and BMI are treated as missing;
zero pregnancies is preserved. Schema validation does not establish data provenance,
clinical correctness, independence, or permission to use a dataset.

Generated reports/checkpoints are ignored by Git. Review outputs before publishing
results from any custom or private data. No new patient records are included.

## Verification and structure

```bash
python -m unittest discover -s tests -v
```

24 regression tests cover CSV validation, split isolation, per-fold imputer/scaler/PCA
fits, counterfactual test-label independence, final refit exclusion of test outliers,
report provenance, CLI export and notebook syntax. CI runs tests on Python
3.11/3.12/3.13 and executes all six notebooks in reduced mode. Full-model convergence
and recognition/clinical accuracy are not CI acceptance criteria.

- `run.py`: reproducible classical CLI
- `run_notebooks.py`: isolated notebook-cell execution
- `ml_lab/experiments.py`: raw data, split, Pipeline, tuning and metrics
- `ml_lab/reporting.py`: labelled current-run metric collection
- `ml_lab/deep.py`: bounded-thread TensorFlow datasets
- `tests/`: regression checks
- `output/`, `results/`, `models/`: generated local artifacts

Dependency files use bounded ranges, not a complete lock file. Record the resolved
environment when reproducing a run; CLI JSON records Python, NumPy, pandas and
scikit-learn versions. The deep CI artifact includes its installed versions.

Method references: [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html)
and [Keras transfer learning guide](https://keras.io/guides/transfer_learning/).
