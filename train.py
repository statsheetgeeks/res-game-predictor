"""
train.py
========
Training pipeline for MLB match prediction (Huang & Li 2021).

Split strategy
--------------
  • 2023 + 2024 + 2025 + first half of 2026  →  training set (with 5-fold CV inside)
  • Second half of 2026                       →  held-out test set (final evaluation)

Usage
-----
  python train.py                          # all datasets, all models
  python train.py --dataset 2              # Dataset 2 only
  python train.py --model svm             # SVM only
  python train.py --no-feature-selection  # skip FS experiments
  python train.py --grid-search           # GridSearchCV for SVM
"""

import os
import json
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

import tensorflow as tf
from tensorflow import keras

from models import (
    build_1dcnn, compile_1dcnn, prepare_1dcnn_input,
    build_ann,   compile_ann,
    build_svm,   binary_confusion,
    HPARAM_GRID_SVM,
)

os.makedirs("models", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("models/training.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)

tf.random.set_seed(42)
np.random.seed(42)

N_FOLDS    = 5
EPOCHS     = 150
BATCH_SIZE = 10
TARGET     = "Y"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
def load_split(ds_num: int, use_fs: bool, data_dir: str = "data"):
    suffix   = "_selected" if use_fs else ""
    tr_path  = os.path.join(data_dir, f"dataset{ds_num}_train{suffix}.csv")
    te_path  = os.path.join(data_dir, f"dataset{ds_num}_test{suffix}.csv")

    if not os.path.exists(tr_path):
        raise FileNotFoundError(f"Missing {tr_path} – run preprocessing.py first.")

    def _load(path):
        df = pd.read_csv(path)
        y  = df[TARGET].values.astype(np.int32)
        X  = df.drop(columns=[TARGET]).values.astype(np.float32)
        return X, y

    X_train, y_train = _load(tr_path)
    X_test,  y_test  = _load(te_path) if os.path.exists(te_path) else \
                       (np.empty((0, X_train.shape[1])), np.array([], dtype=np.int32))

    logger.info("Dataset %d%s | train=%d  test=%d  features=%d",
                ds_num, " [FS]" if use_fs else "",
                len(y_train), len(y_test), X_train.shape[1])
    return X_train, y_train, X_test, y_test


# ---------------------------------------------------------------------------
# Cross-validation helpers
# ---------------------------------------------------------------------------
def _print_cv_header(model_name: str):
    logger.info("┌─ %s %s", model_name, "─" * (50 - len(model_name)))
    logger.info("│  Fold  │  Accuracy  │  TP   FN   FP   TN")
    logger.info("│─────────────────────────────────────────────")


def _print_cv_row(fold, m):
    logger.info("│  CV=%d  │  %.4f    │  %4d %4d %4d %4d",
                fold, m["accuracy"], m["TP"], m["FN"], m["FP"], m["TN"])


def cv_svm(X, y, use_grid_search=False):
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    _print_cv_header("SVM")

    if use_grid_search:
        tr_i, _ = next(skf.split(X, y))
        gs = GridSearchCV(SVC(probability=True, random_state=42),
                          HPARAM_GRID_SVM, cv=3, scoring="accuracy", n_jobs=-1)
        gs.fit(X[tr_i], y[tr_i])
        best_params = gs.best_params_
        logger.info("│  Best params: %s", best_params)
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    else:
        best_params = {"kernel": "rbf", "C": 1000, "gamma": 0.1}

    results, best_acc, best_model = [], 0.0, None
    for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
        sc   = MinMaxScaler()
        X_tr = sc.fit_transform(X[tr_i])
        X_va = sc.transform(X[va_i])
        svm  = SVC(probability=True, random_state=42, **best_params)
        svm.fit(X_tr, y[tr_i])
        m = binary_confusion(y[va_i], svm.predict(X_va))
        results.append({"fold": fold, **m})
        _print_cv_row(fold, m)
        if m["accuracy"] > best_acc:
            best_acc, best_model = m["accuracy"], (svm, sc)

    avg = np.mean([r["accuracy"] for r in results])
    logger.info("└─ SVM  CV Average Accuracy = %.4f", avg)
    return results, best_model, best_params


def cv_ann(X, y, optimizer="adam"):
    n_feat = X.shape[1]
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    _print_cv_header("ANN")

    results, best_acc, best_model = [], 0.0, None
    for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
        sc    = MinMaxScaler()
        X_tr  = sc.fit_transform(X[tr_i]).astype(np.float32)
        X_va  = sc.transform(X[va_i]).astype(np.float32)
        model = compile_ann(build_ann(n_feat), optimizer=optimizer)
        cb    = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                              restore_best_weights=True)
        model.fit(X_tr, y[tr_i], epochs=EPOCHS, batch_size=BATCH_SIZE,
                  validation_data=(X_va, y[va_i]), callbacks=[cb], verbose=0)
        y_pred = (model.predict(X_va, verbose=0).squeeze() >= 0.5).astype(int)
        m = binary_confusion(y[va_i], y_pred)
        results.append({"fold": fold, **m})
        _print_cv_row(fold, m)
        if m["accuracy"] > best_acc:
            best_acc, best_model = m["accuracy"], model

    avg = np.mean([r["accuracy"] for r in results])
    logger.info("└─ ANN  CV Average Accuracy = %.4f", avg)
    return results, best_model


def cv_1dcnn(X, y, optimizer="rmsprop"):
    n_feat = X.shape[1]
    skf    = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    _print_cv_header("1DCNN")

    results, best_acc, best_model = [], 0.0, None
    for fold, (tr_i, va_i) in enumerate(skf.split(X, y), 1):
        sc    = MinMaxScaler()
        X_tr  = prepare_1dcnn_input(sc.fit_transform(X[tr_i]).astype(np.float32))
        X_va  = prepare_1dcnn_input(sc.transform(X[va_i]).astype(np.float32))
        model = compile_1dcnn(build_1dcnn(n_feat), optimizer=optimizer)
        cb    = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,
                                              restore_best_weights=True)
        model.fit(X_tr, y[tr_i], epochs=EPOCHS, batch_size=30,
                  validation_data=(X_va, y[va_i]), callbacks=[cb], verbose=0)
        y_pred = (model.predict(X_va, verbose=0).squeeze() >= 0.5).astype(int)
        m = binary_confusion(y[va_i], y_pred)
        results.append({"fold": fold, **m})
        _print_cv_row(fold, m)
        if m["accuracy"] > best_acc:
            best_acc, best_model = m["accuracy"], model

    avg = np.mean([r["accuracy"] for r in results])
    logger.info("└─ 1DCNN  CV Average Accuracy = %.4f", avg)
    return results, best_model


# ---------------------------------------------------------------------------
# Final refit + test evaluation
# ---------------------------------------------------------------------------
def final_eval_svm(X_train, y_train, X_test, y_test, best_params, tag):
    sc   = MinMaxScaler()
    X_tr = sc.fit_transform(X_train)
    svm  = SVC(probability=True, random_state=42, **best_params)
    svm.fit(X_tr, y_train)
    joblib.dump((svm, sc), f"models/svm_{tag}.pkl")

    if len(X_test) == 0:
        return {}
    X_te = sc.transform(X_test)
    m    = binary_confusion(y_test, svm.predict(X_te))
    logger.info("SVM  [%s] HOLDOUT Accuracy = %.4f  (TP=%d FN=%d FP=%d TN=%d)",
                tag, m["accuracy"], m["TP"], m["FN"], m["FP"], m["TN"])
    return m


def final_eval_ann(X_train, y_train, X_test, y_test, tag, optimizer="adam"):
    n_feat = X_train.shape[1]
    sc     = MinMaxScaler()
    X_tr   = sc.fit_transform(X_train).astype(np.float32)
    model  = compile_ann(build_ann(n_feat), optimizer=optimizer)
    model.fit(X_tr, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=0)
    model.save(f"models/ann_{tag}.keras")

    if len(X_test) == 0:
        return {}
    X_te   = sc.transform(X_test).astype(np.float32)
    y_pred = (model.predict(X_te, verbose=0).squeeze() >= 0.5).astype(int)
    m      = binary_confusion(y_test, y_pred)
    logger.info("ANN  [%s] HOLDOUT Accuracy = %.4f  (TP=%d FN=%d FP=%d TN=%d)",
                tag, m["accuracy"], m["TP"], m["FN"], m["FP"], m["TN"])
    return m


def final_eval_1dcnn(X_train, y_train, X_test, y_test, tag, optimizer="rmsprop"):
    n_feat = X_train.shape[1]
    sc     = MinMaxScaler()
    X_tr   = prepare_1dcnn_input(sc.fit_transform(X_train).astype(np.float32))
    model  = compile_1dcnn(build_1dcnn(n_feat), optimizer=optimizer)
    model.fit(X_tr, y_train, epochs=EPOCHS, batch_size=30, verbose=0)
    model.save(f"models/1dcnn_{tag}.keras")

    if len(X_test) == 0:
        return {}
    X_te   = prepare_1dcnn_input(sc.transform(X_test).astype(np.float32))
    y_pred = (model.predict(X_te, verbose=0).squeeze() >= 0.5).astype(int)
    m      = binary_confusion(y_test, y_pred)
    logger.info("1DCNN [%s] HOLDOUT Accuracy = %.4f  (TP=%d FN=%d FP=%d TN=%d)",
                tag, m["accuracy"], m["TP"], m["FN"], m["FP"], m["TN"])
    return m


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train_all(
    datasets: list      = [1, 2],
    models_to_run: list = ["1dcnn", "ann", "svm"],
    feature_selection: bool = True,
    grid_search: bool   = False,
    data_dir: str       = "data",
):
    all_results = {}

    for ds_num in datasets:
        all_results[f"ds{ds_num}"] = {}

        for use_fs in ([False, True] if feature_selection else [False]):
            fs_label = "fs" if use_fs else "no_fs"
            try:
                X_train, y_train, X_test, y_test = load_split(ds_num, use_fs, data_dir)
            except FileNotFoundError as e:
                logger.error(str(e))
                continue

            for model_name in models_to_run:
                # 1DCNN not used with feature selection (matches paper)
                if use_fs and model_name == "1dcnn":
                    continue

                tag = f"ds{ds_num}_{fs_label}_{model_name}"
                logger.info("\n═══ %s ═══════════════════════════════", tag.upper())

                # Cross-validation
                if model_name == "svm":
                    cv_res, best, best_params = cv_svm(X_train, y_train, grid_search)
                elif model_name == "ann":
                    cv_res, best = cv_ann(X_train, y_train)
                    best_params  = {}
                elif model_name == "1dcnn":
                    cv_res, best = cv_1dcnn(X_train, y_train)
                    best_params  = {}
                else:
                    logger.warning("Unknown model: %s", model_name)
                    continue

                cv_avg = np.mean([r["accuracy"] for r in cv_res])

                # Final refit + holdout evaluation
                if model_name == "svm":
                    test_m = final_eval_svm(X_train, y_train, X_test, y_test,
                                            best_params, tag)
                elif model_name == "ann":
                    test_m = final_eval_ann(X_train, y_train, X_test, y_test, tag)
                elif model_name == "1dcnn":
                    test_m = final_eval_1dcnn(X_train, y_train, X_test, y_test, tag)

                all_results[f"ds{ds_num}"][f"{model_name}_{fs_label}"] = {
                    "cv_avg_accuracy": cv_avg,
                    "test_accuracy":   test_m.get("accuracy"),
                    "test_metrics":    test_m,
                    "cv_folds":        cv_res,
                }

    # Summary table
    logger.info("\n%s", "═" * 75)
    logger.info("  FINAL RESULTS SUMMARY")
    logger.info("═" * 75)
    logger.info("  %-12s  %-8s  %-18s  %-14s  %s",
                "Dataset", "Model", "Setting", "CV Accuracy", "Holdout Accuracy")
    logger.info("─" * 75)
    for ds_key, ds_vals in all_results.items():
        for exp_key, exp_val in ds_vals.items():
            model, _, fs = exp_key.rpartition("_")
            setting = "w/ feature sel." if fs == "fs" else "no feature sel."
            holdout = f'{exp_val["test_accuracy"]:.4f}' \
                      if exp_val["test_accuracy"] is not None else "  N/A  "
            logger.info("  %-12s  %-8s  %-18s  %-14.4f  %s",
                        ds_key, model.upper(), setting,
                        exp_val["cv_avg_accuracy"], holdout)
    logger.info("═" * 75)

    with open("models/training_summary.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    logger.info("Summary saved → models/training_summary.json")
    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   type=int, default=None)
    parser.add_argument("--model",     type=str, default=None,
                        help="1dcnn | ann | svm")
    parser.add_argument("--no-feature-selection", dest="fs", action="store_false")
    parser.add_argument("--grid-search", action="store_true")
    parser.add_argument("--data-dir",  type=str, default="data")
    args = parser.parse_args()

    train_all(
        datasets        = [args.dataset] if args.dataset else [1, 2],
        models_to_run   = [args.model]   if args.model   else ["1dcnn", "ann", "svm"],
        feature_selection = args.fs,
        grid_search     = args.grid_search,
        data_dir        = args.data_dir,
    )
