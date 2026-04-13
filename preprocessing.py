"""
preprocessing.py
================
Builds train/test splits from multi-season data.

Split strategy
--------------
  Full training  : 2023, 2024, 2025  (complete seasons)
  Partial train  : first SPLIT_RATIO of 2026 games (chronological)
  Holdout test   : last  (1-SPLIT_RATIO) of 2026 games

As more 2026 games are played, re-run this script to refresh the split.

Usage
-----
  python preprocessing.py
  python preprocessing.py --split-ratio 0.6   # 60% of 2026 → train
"""

import os
import json
import argparse
import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

BATTING_VARS   = [f"B{i}"  for i in range(1, 14)]
SP_VARS        = [f"SP{i}" for i in range(1, 17)]
ALL_PITCH_VARS = [f"P{i}"  for i in range(1, 19)]
HOME_AWAY      = ["X1"]
TARGET         = "Y"

DATASET1_COLS = BATTING_VARS + SP_VARS        + HOME_AWAY
DATASET2_COLS = BATTING_VARS + ALL_PITCH_VARS + HOME_AWAY

FEATURE_LABELS = {
    "B1":"At Bats","B2":"Hits","B3":"Walks","B4":"Strikeouts",
    "B5":"Plate Appearances","B6":"Batting Avg","B7":"OBP","B8":"SLG","B9":"OPS",
    "B10":"Pitches per PA","B11":"Strikes","B12":"Putouts","B13":"Assists",
    "SP1":"IP (SP)","SP2":"H Allowed (SP)","SP3":"BB (SP)","SP4":"K (SP)",
    "SP5":"HR (SP)","SP6":"ERA (SP)","SP7":"BF (SP)","SP8":"Pitches (SP)",
    "SP9":"Strikes (SP)","SP10":"Contact K (SP)","SP11":"Swinging K (SP)",
    "SP12":"Looking K (SP)","SP13":"GB (SP)","SP14":"FB (SP)","SP15":"LD (SP)",
    "SP16":"Game Score (SP)",
    "P1":"IP (All)","P2":"H Allowed (All)","P3":"BB (All)","P4":"K (All)",
    "P5":"HR (All)","P6":"ERA (All)","P7":"BF (All)","P8":"Pitches (All)",
    "P9":"Strikes (All)","P10":"Contact K (All)","P11":"Swinging K (All)",
    "P12":"Looking K (All)","P13":"GB (All)","P14":"FB (All)","P15":"LD (All)",
    "P16":"Game Score (All)","P17":"Inherited Runners","P18":"Inherited Score",
    "X1":"Home Team",
}

TRAIN_SEASONS   = [2023, 2024, 2025]
PARTIAL_SEASON  = 2026
SPLIT_RATIO     = 0.5


# ---------------------------------------------------------------------------
# ReliefF
# ---------------------------------------------------------------------------
class ReliefFSelector:
    def __init__(self, n_neighbors=10, n_samples=300, threshold=0.0, random_state=42):
        self.k, self.m  = n_neighbors, n_samples
        self.threshold  = threshold
        self.rng        = np.random.default_rng(random_state)
        self.weights_   = None
        self.selected_mask_ = None

    def fit(self, X, y):
        n, f   = X.shape
        W      = np.zeros(f)
        ranges = X.max(0) - X.min(0)
        ranges[ranges == 0] = 1.0
        classes, counts = np.unique(y, return_counts=True)
        priors = dict(zip(classes, counts / n))
        idx    = self.rng.choice(n, size=min(self.m, n), replace=False)
        for i in idx:
            Ri, ci = X[i], y[i]
            same   = np.where(y == ci)[0]
            same   = same[same != i]
            hits   = X[same[np.argsort(np.sum(np.abs(X[same] - Ri), 1))[:self.k]]]
            if len(hits):
                W -= np.sum(np.abs(hits - Ri) / ranges, 0) / (self.m * self.k)
            for c in classes:
                if c == ci:
                    continue
                pool   = X[y == c]
                misses = pool[np.argsort(np.sum(np.abs(pool - Ri), 1))[:self.k]]
                if len(misses):
                    ratio = priors[c] / (1.0 - priors.get(ci, 0) + 1e-9)
                    W += ratio * np.sum(np.abs(misses - Ri) / ranges, 0) \
                         / (self.m * self.k)
        self.weights_        = W
        self.selected_mask_  = W > self.threshold
        return self

    def transform(self, X):
        return X[:, self.selected_mask_]

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["game_date"])
    df = df.dropna(subset=[TARGET])
    num = df.select_dtypes(include=[np.number]).columns
    df[num] = df[num].fillna(df[num].median())
    logger.info("Loaded %d rows from %s", len(df), path)
    return df


def build_xy(df, cols):
    for c in cols:
        if c not in df.columns:
            df[c] = 0.0
    return df[cols].values.astype(np.float32), df[TARGET].values.astype(np.int32)


def chrono_split(df_partial: pd.DataFrame, ratio: float):
    """Split games chronologically; keep both team rows together."""
    game_dates = (df_partial.groupby("game_id")["game_date"].min()
                  .sort_values().reset_index())
    n_tr  = int(len(game_dates) * ratio)
    tr_ids = set(game_dates.iloc[:n_tr]["game_id"])
    te_ids = set(game_dates.iloc[n_tr:]["game_id"])
    split_dt = game_dates.iloc[n_tr]["game_date"] if n_tr < len(game_dates) \
               else game_dates.iloc[-1]["game_date"]
    logger.info("Chrono split at %s: %d train games / %d test games",
                split_dt.strftime("%Y-%m-%d"), len(tr_ids), len(te_ids))
    return (df_partial[df_partial["game_id"].isin(tr_ids)].copy(),
            df_partial[df_partial["game_id"].isin(te_ids)].copy())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def preprocess(raw_path="data/raw_games_combined.csv", out_dir="data",
               train_seasons=None, partial_season=PARTIAL_SEASON,
               split_ratio=SPLIT_RATIO):

    if train_seasons is None:
        train_seasons = TRAIN_SEASONS
    os.makedirs(out_dir, exist_ok=True)
    df = load_raw(raw_path)

    df_train_base = df[df["season"].isin(train_seasons)].copy()
    df_partial    = df[df["season"] == partial_season].copy()

    if len(df_partial):
        df_2026_tr, df_test = chrono_split(df_partial, split_ratio)
    else:
        logger.warning("No %d data found – training on %s only.", partial_season, train_seasons)
        df_2026_tr = df_test = pd.DataFrame()

    df_train = pd.concat([df_train_base, df_2026_tr], ignore_index=True)
    logger.info(
        "Split: train=%d rows (%s + first %.0f%% of %d), test=%d rows (last %.0f%% of %d)",
        len(df_train), train_seasons, split_ratio*100, partial_season,
        len(df_test), (1-split_ratio)*100, partial_season,
    )

    summary = {
        "train_seasons": train_seasons,
        "partial_season": partial_season,
        "split_ratio": split_ratio,
        "n_train_rows": int(len(df_train)),
        "n_test_rows":  int(len(df_test)),
        "datasets": {},
    }

    for ds_num, feat_cols in [(1, DATASET1_COLS), (2, DATASET2_COLS)]:
        logger.info("── Dataset %d ────────────────────────────────", ds_num)
        X_tr, y_tr = build_xy(df_train, feat_cols)
        X_te, y_te = build_xy(df_test,  feat_cols) if len(df_test) \
                     else (np.empty((0, len(feat_cols))), np.array([], dtype=int))

        scaler   = MinMaxScaler()
        X_tr_n   = scaler.fit_transform(X_tr).astype(np.float32)
        X_te_n   = scaler.transform(X_te).astype(np.float32) if len(X_te) else X_te

        # Save scaler params for inference
        scaler_df = pd.DataFrame({"feature": feat_cols,
                                  "min": scaler.data_min_,
                                  "max": scaler.data_max_})
        scaler_df.to_csv(os.path.join(out_dir, f"scaler_ds{ds_num}.csv"), index=False)

        for split, X_n, y in [("train", X_tr_n, y_tr), ("test", X_te_n, y_te)]:
            out = pd.DataFrame(X_n, columns=feat_cols)
            out[TARGET] = y
            out.to_csv(os.path.join(out_dir, f"dataset{ds_num}_{split}.csv"), index=False)
            logger.info("  Saved ds%d %s → %d rows", ds_num, split, len(out))

        # ReliefF
        sel = ReliefFSelector()
        sel.fit(X_tr_n, y_tr)
        weights  = sel.weights_
        selected = [f for f, w in zip(feat_cols, weights) if w > 0]
        sel_idx  = [feat_cols.index(f) for f in selected]
        logger.info("  ReliefF: %d / %d features selected", len(selected), len(feat_cols))

        fi = pd.DataFrame({
            "feature":  feat_cols,
            "label":    [FEATURE_LABELS.get(c, c) for c in feat_cols],
            "weight":   weights,
            "selected": weights > 0,
        }).sort_values("weight", ascending=False)
        fi.to_csv(os.path.join(out_dir, f"feature_importance_ds{ds_num}.csv"), index=False)

        # Save selected-feature scaler params too
        sel_scaler_df = pd.DataFrame({"feature": selected,
                                      "min": scaler.data_min_[sel_idx],
                                      "max": scaler.data_max_[sel_idx]})
        sel_scaler_df.to_csv(
            os.path.join(out_dir, f"scaler_ds{ds_num}_selected.csv"), index=False)

        for split, X_n, y in [("train", X_tr_n[:, sel_idx], y_tr),
                               ("test",  X_te_n[:, sel_idx] if len(X_te_n) else X_te_n, y_te)]:
            out = pd.DataFrame(X_n, columns=selected)
            out[TARGET] = y
            out.to_csv(os.path.join(out_dir, f"dataset{ds_num}_{split}_selected.csv"), index=False)

        summary["datasets"][f"ds{ds_num}"] = {
            "n_features_orig":  len(feat_cols),
            "n_features_sel":   len(selected),
            "selected_features": selected,
        }

    with open(os.path.join(out_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info("✅  Done. Summary → data/split_summary.json")
    return summary


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--raw",            type=str,   default="data/raw_games_combined.csv")
    p.add_argument("--split-ratio",    type=float, default=SPLIT_RATIO)
    p.add_argument("--train-seasons",  type=int,   nargs="+", default=TRAIN_SEASONS)
    p.add_argument("--partial-season", type=int,   default=PARTIAL_SEASON)
    p.add_argument("--out-dir",        type=str,   default="data")
    a = p.parse_args()
    preprocess(raw_path=a.raw, out_dir=a.out_dir, train_seasons=a.train_seasons,
               partial_season=a.partial_season, split_ratio=a.split_ratio)
