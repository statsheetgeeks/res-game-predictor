# ⚾ MLB Win Predictor

Replication and extension of **Huang & Li (2021)** using modern (2023–2025) MLB data.

---

## Data Split Strategy

```
2023 (full) ──┐
2024 (full) ──┼──► TRAINING SET
2025 (first 50%, chronological) ─┘
                                        ┌──► 5-fold CV (model selection)
                                        └──► Final refit on full training set

2025 (last 50%, chronological) ────► HOLDOUT TEST SET (final evaluation)
```

The 2025 season is split **chronologically** — earlier games supplement training, later games form the held-out test set. This is a more realistic evaluation than random k-fold alone because it reflects real-world deployment (predicting future games from past data).

---

## Quickstart

```bash
pip install -r requirements.txt

# 1. Scrape 2023, 2024, 2025 seasons
python data_collection.py

# 2. Preprocess + build train/test splits + ReliefF selection
python preprocessing.py

# 3. Train all models (5-fold CV + holdout eval)
python train.py

# 4. Launch web app
streamlit run app.py
```

### Options

```bash
# Scrape specific years only
python data_collection.py --years 2024 2025

# Change the 2025 train/test split ratio (default 0.5)
python preprocessing.py --split-ratio 0.6

# Train a single model on Dataset 2 only
python train.py --dataset 2 --model svm

# Run GridSearchCV for SVM hyperparameters
python train.py --grid-search
```

---

## Models

| Model  | Architecture | Paper Best (2019) |
|--------|-------------|-------------------|
| 1DCNN  | 2 conv layers → maxpool → dropout → dense → sigmoid | 93.40% |
| ANN    | 1 hidden layer (sigmoid) → dropout → sigmoid output  | 94.18% |
| SVM    | RBF kernel, C=1000, γ=0.1                            | 94.16% |

---

## Datasets

**Dataset 1** — Starting Pitcher Only: `B1–B13 + SP1–SP16 + X1` (30 features)  
**Dataset 2** — All Pitchers: `B1–B13 + P1–P18 + X1` (32 features)

---

## Reference
> Huang, M.-L., & Li, Y.-Z. (2021). Use of Machine Learning and Deep Learning to Predict the Outcomes of Major League Baseball Matches. *Applied Sciences*, 11, 4499. https://doi.org/10.3390/app11104499
