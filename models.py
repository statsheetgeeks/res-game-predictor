"""
models.py
=========
Defines the three prediction models from Huang & Li (2021):

  1. 1D Convolutional Neural Network (1DCNN)  – deep learning
  2. Artificial Neural Network (ANN)           – traditional ML / shallow NN
  3. Support Vector Machine (SVM)             – classical ML

All models produce a binary win/loss prediction (1 = win, 0 = loss).
"""

import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers


# ---------------------------------------------------------------------------
# 1. 1DCNN
# ---------------------------------------------------------------------------
def build_1dcnn(n_features: int,
                filters1: int = 16,
                filters2: int = 32,
                kernel_size: int = 3,
                pool_size: int = 2,
                dense_units: int = 50,
                dropout_rate: float = 0.2) -> keras.Model:
    """
    Build the 1DCNN architecture from Table 3 / Figure 5 of the paper.

    Input shape per sample: (n_features, 1)
    The 32 (or 30) game statistics are treated as a 1-D sequence of length n_features.

    Architecture
    ------------
    conv1d_1      → 16 filters, kernel=3, same padding, ReLU
    maxpooling1d  → pool=2, stride=1
    conv1d_2      → 32 filters, kernel=3, same padding, ReLU
    maxpooling1d  → pool=2, stride=1
    dropout       → 0.2
    flatten
    dense(ReLU)   → 50 units
    dropout       → 0.2
    sigmoid       → 1 (binary output)
    """
    inp = keras.Input(shape=(n_features, 1), name="input")

    x = layers.Conv1D(filters1, kernel_size, padding="same",
                      activation="relu", name="conv1d_1")(inp)
    x = layers.MaxPooling1D(pool_size=pool_size, strides=1,
                            padding="same", name="maxpooling1d_1")(x)
    x = layers.Conv1D(filters2, kernel_size, padding="same",
                      activation="relu", name="conv1d_2")(x)
    x = layers.MaxPooling1D(pool_size=pool_size, strides=1,
                            padding="same", name="maxpooling1d_2")(x)
    x = layers.Dropout(dropout_rate, name="dropout_1")(x)
    x = layers.Flatten(name="flatten")(x)
    x = layers.Dense(dense_units, activation="relu", name="dense_relu")(x)
    x = layers.Dropout(dropout_rate, name="dropout_2")(x)
    out = layers.Dense(1, activation="sigmoid", name="sigmoid")(x)

    model = keras.Model(inputs=inp, outputs=out, name="1DCNN")
    return model


def compile_1dcnn(model: keras.Model, optimizer: str = "rmsprop",
                  learning_rate: float = 1e-3) -> keras.Model:
    """Compile with binary cross-entropy loss (paper setting)."""
    if optimizer.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


def prepare_1dcnn_input(X: np.ndarray) -> np.ndarray:
    """Reshape (n_samples, n_features) → (n_samples, n_features, 1) for 1DCNN."""
    return X.reshape(X.shape[0], X.shape[1], 1)


# ---------------------------------------------------------------------------
# 2. ANN
# ---------------------------------------------------------------------------
def build_ann(n_features: int,
              hidden_units: int | None = None,
              dropout_rate: float = 0.1,
              kernel_initializer: str = "glorot_uniform") -> keras.Model:
    """
    Build the ANN from Section 3.4.2 of the paper.

    Architecture
    ------------
    dense(sigmoid)  → hidden_units (default: 16 for DS1, 17 for DS2)
    dropout         → 0.1
    dense(sigmoid)  → 1 (binary output)

    The paper uses the sigmoid function as the excitation function
    in the hidden layer for a nonlinear binary classification problem.
    """
    # Paper sets hidden units to 16 (DS1) or 17 (DS2); auto-derive if not given
    if hidden_units is None:
        hidden_units = 16 if n_features <= 30 else 17

    inp = keras.Input(shape=(n_features,), name="input")
    x   = layers.Dense(hidden_units, activation="sigmoid",
                        kernel_initializer=kernel_initializer,
                        name="hidden")(inp)
    x   = layers.Dropout(dropout_rate, name="dropout")(x)
    out = layers.Dense(1, activation="sigmoid",
                       kernel_initializer=kernel_initializer,
                       name="output")(x)

    model = keras.Model(inputs=inp, outputs=out, name="ANN")
    return model


def compile_ann(model: keras.Model, optimizer: str = "adam",
                learning_rate: float = 1e-3) -> keras.Model:
    if optimizer.lower() == "adam":
        opt = keras.optimizers.Adam(learning_rate=learning_rate)
    else:
        opt = keras.optimizers.RMSprop(learning_rate=learning_rate)
    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


# ---------------------------------------------------------------------------
# 3. SVM
# ---------------------------------------------------------------------------
def build_svm(kernel: str = "rbf", C: float = 1000.0, gamma: float = 0.1) -> SVC:
    """
    Build the RBF-SVM from Section 3.4.3 of the paper.
    Best params from GridSearchCV (Table 4): kernel=RBF, C=1000, gamma=0.1.
    """
    return SVC(kernel=kernel, C=C, gamma=gamma,
               probability=True,   # enables predict_proba for the web app
               random_state=42)


# ---------------------------------------------------------------------------
# Hyper-parameter grids for GridSearchCV (mirroring paper's sweep)
# ---------------------------------------------------------------------------
HPARAM_GRID_SVM = [
    {
        "kernel": ["linear"],
        "C":      [1, 10, 100, 1000],
    },
    {
        "kernel": ["rbf"],
        "C":      [1, 10, 100, 1000],
        "gamma":  [0.0001, 0.001, 0.01, 0.1, 1, 10, 100],
    },
]

HPARAM_GRID_KERAS = {
    "optimizer":   ["adam", "rmsprop"],
    "batch_size":  [10, 20, 30],
    "epochs":      [50, 100, 150],
}


# ---------------------------------------------------------------------------
# Confusion-matrix utilities
# ---------------------------------------------------------------------------
def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Return TP, FP, FN, TN, accuracy, precision, recall, F1."""
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    total     = TP + FP + FN + TN
    accuracy  = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) else 0.0)
    return dict(TP=TP, FP=FP, FN=FN, TN=TN,
                accuracy=accuracy, precision=precision,
                recall=recall, f1=f1)
