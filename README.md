# ML & Deep Learning Course Project
**School of Artificial Intelligence and Data Science | 2024–2025**

---

## Project Structure

```
notebooks/
├── diabetes.csv                          ← Dataset A (required)
├── requirements.txt                      ← Python dependencies
├── README.md                             ← This file
│
├── Task1_Classification_Algorithms.ipynb     ← Theory + Deep Dive (10 pts)
├── Task2_Classification_Implementation.ipynb ← Full ML Pipeline (10 pts)
├── Task3_Dimensionality_Reduction.ipynb      ← PCA, t-SNE, UMAP, LDA (10 pts)
├── Task4_Deep_Learning_Fundamentals.ipynb    ← MLP, Optimizers, BN (10 pts)
├── Task5_CNNs.ipynb                          ← CNN + Transfer Learning (10 pts)
└── Task6_General_Overview_Integration.ipynb  ← Survey + Analysis (10 pts)
```

---

## Setup Instructions

### 1. Clone / download the project

```bash
unzip ProjectGroup_[ID].zip
cd notebooks/
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate          # Linux/macOS
venv\Scripts\activate             # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For GPU support (Tasks 4 & 5):
```bash
pip install tensorflow[and-cuda]   # TF with CUDA support
```

Alternatively, use **Google Colab** (free GPU):
- Upload notebooks and `diabetes.csv` to your Google Drive
- Open each notebook in Colab and run all cells

### 4. Launch Jupyter

```bash
jupyter notebook
```

Open `localhost:8888` in your browser and navigate to the `notebooks/` folder.

---

## Running Order

Run notebooks **in order** (Task 1 → 6). Some later tasks depend on preprocessing code from Task 2.

| Notebook | Expected Runtime | GPU Required |
|---|---|---|
| Task 1 | ~2 min | No |
| Task 2 | ~5 min | No |
| Task 3 | ~8 min | No (UMAP optional) |
| Task 4 | ~10 min | Recommended |
| Task 5 | ~30–60 min | **Yes** (CIFAR-10) |
| Task 6 | ~2 min | No |

---

## Datasets

| Name | Source | Location |
|---|---|---|
| Pima Indians Diabetes | UCI ML Repo | `diabetes.csv` (included) |
| Breast Cancer Wisconsin | `sklearn.datasets` | Auto-loaded |
| CIFAR-10 | `keras.datasets` | Auto-downloaded (~170 MB) |

---

## Notes

- All notebooks are self-contained and reproducible (seeds set to 42)
- UMAP requires: `pip install umap-learn` — Task 3 runs without it
- Task 5 trains for up to 50 epochs; reduce `EPOCHS` to run faster
- All figures are saved as `.png` in the same directory as the notebooks

---

## Dependencies Summary

- **scikit-learn**: Classical ML, preprocessing, evaluation
- **TensorFlow/Keras**: Deep learning (Tasks 4 & 5)
- **umap-learn**: UMAP dimensionality reduction (Task 3)
- **pandas / numpy**: Data manipulation
- **matplotlib / seaborn**: Visualization

---

*Good luck reviewing! All code runs end-to-end without errors on Python 3.10+.*
