"""
Autoencoder anomaly detector.

Trains a small Keras autoencoder on clean-only GNSS feature data, then
flags epochs as anomalous when their reconstruction error (MSE) exceeds
a threshold calibrated on a held-out slice of clean data.

This is the "deep" counterpart to isolation_forest.py, per the project's
architecture doc, which calls for training BOTH a classical model and an
autoencoder and comparing them -- not because one is a placeholder for
the other, but because the comparison itself is a legitimate result.

Architecture (per 01-architecture.md):
    6 -> [2 or 3] -> 6   (NOT 6 -> 1 -> 6; a single-scalar bottleneck
    tends to just learn a crude average and gives noisy, unstable
    reconstruction error)

All modeling comes from tf.keras (Dense layers, Adam optimizer, MSE
loss) and numpy/pandas/sklearn for data handling and evaluation -- no
custom neural network code, only the standard Keras Sequential API.

Threshold calibration: the reconstruction-error threshold is set to the
Nth percentile (default 95th, per the architecture doc's suggestion of
95th or 99th) of reconstruction error on a held-out clean validation
split, NOT on the training data itself -- this avoids overfitting the
threshold to the exact data the model was fit on.

Feature choice and evaluation: identical to isolation_forest.py, so the
two detectors are directly comparable. See that module's docstring for
why hdop/pdop/fix_quality are excluded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Same feature set as isolation_forest.py -- kept identical so the two
# detectors are directly comparable on the same inputs.
FEATURE_COLUMNS = [
    "mean_snr",
    "max_snr",
    "var_snr",
    "tracked_satellites",
    "delta_mean_snr",
    "rolling_var_snr",
]


@dataclass
class TrainedAutoencoder:
    scaler: StandardScaler
    model: keras.Model
    threshold: float
    feature_columns: list[str]


@dataclass
class ScenarioResult:
    name: str
    n_epochs: int
    n_true_spoofed: int
    precision: float
    recall: float
    f1: float
    confusion: np.ndarray  # [[TN, FP], [FN, TP]]
    detection_latency_epochs: int | None


def load_feature_csv(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


def build_autoencoder(n_features: int, bottleneck_dim: int = 3) -> keras.Model:
    """
    Build the 6 -> [2 or 3] -> 6 autoencoder specified in the architecture
    doc. bottleneck_dim defaults to 3; pass 2 to try the smaller option
    the doc also allows.
    """
    if bottleneck_dim not in (2, 3):
        raise ValueError(
            "Architecture doc specifies a bottleneck of 2 or 3 -- "
            "not 1 (crude-average problem) and not larger (defeats the "
            "point of a compressed representation)."
        )

    inputs = keras.Input(shape=(n_features,))
    x = layers.Dense(4, activation="relu")(inputs)
    bottleneck = layers.Dense(bottleneck_dim, activation="relu", name="bottleneck")(x)
    x = layers.Dense(4, activation="relu")(bottleneck)
    outputs = layers.Dense(n_features, activation="linear")(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="mse")
    return model


def train_autoencoder(
    clean_dfs: list[pd.DataFrame],
    bottleneck_dim: int = 3,
    threshold_percentile: float = 95.0,
    epochs: int = 50,
    validation_fraction: float = 0.2,
    random_state: int = 42,
    verbose: int = 0,
) -> TrainedAutoencoder:
    """
    Fit a StandardScaler + autoencoder on concatenated clean-only data,
    holding out `validation_fraction` of it to calibrate the anomaly
    threshold (rather than calibrating on the exact data the model
    trained on, which would underestimate normal reconstruction error).
    """
    combined = pd.concat(clean_dfs, ignore_index=True)
    X = combined[FEATURE_COLUMNS].to_numpy()

    X_train, X_val = train_test_split(
        X, test_size=validation_fraction, random_state=random_state
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    tf.random.set_seed(random_state)
    model = build_autoencoder(n_features=len(FEATURE_COLUMNS), bottleneck_dim=bottleneck_dim)

    model.fit(
        X_train_scaled, X_train_scaled,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        verbose=verbose,
        callbacks=[keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)],
    )

    # Calibrate threshold on the held-out clean validation split.
    val_recon = model.predict(X_val_scaled, verbose=0)
    val_mse = np.mean(np.square(X_val_scaled - val_recon), axis=1)
    threshold = float(np.percentile(val_mse, threshold_percentile))

    return TrainedAutoencoder(
        scaler=scaler, model=model, threshold=threshold, feature_columns=FEATURE_COLUMNS
    )


def score_dataframe(detector: TrainedAutoencoder, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the trained autoencoder on a DataFrame. Returns (mse, y_pred)
    where y_pred is 1 if mse > threshold (flagged anomalous), else 0.
    """
    X = df[detector.feature_columns].to_numpy()
    X_scaled = detector.scaler.transform(X)
    recon = detector.model.predict(X_scaled, verbose=0)
    mse = np.mean(np.square(X_scaled - recon), axis=1)
    y_pred = (mse > detector.threshold).astype(int)
    return mse, y_pred


def evaluate_scenario(
    detector: TrainedAutoencoder,
    df: pd.DataFrame,
    name: str,
    label_column: str = "known_spoof_window",
) -> ScenarioResult:
    if label_column not in df.columns:
        raise ValueError(
            f"'{label_column}' not found in {name} -- re-run texbat_loader.py "
            f"with the current version to get ground-truth labels."
        )

    y_true = df[label_column].to_numpy()
    if (y_true == -1).any():
        raise ValueError(
            f"{name} has unknown (-1) ground-truth labels -- scenario name "
            f"wasn't recognized when the CSV was generated."
        )

    _, y_pred = score_dataframe(detector, df)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    latency = None
    spoofed_indices = np.where(y_true == 1)[0]
    if spoofed_indices.size > 0:
        attack_start_idx = spoofed_indices[0]
        for i in range(attack_start_idx, len(y_true)):
            if y_pred[i] == 1:
                latency = i - attack_start_idx
                break

    return ScenarioResult(
        name=name,
        n_epochs=len(df),
        n_true_spoofed=int(y_true.sum()),
        precision=precision,
        recall=recall,
        f1=f1,
        confusion=cm,
        detection_latency_epochs=latency,
    )


def print_scenario_result(result: ScenarioResult, epoch_period_s: float | None = None) -> None:
    print(f"\n=== {result.name} ===")
    print(f"Epochs: {result.n_epochs}  (true spoofed: {result.n_true_spoofed})")
    print(f"Precision: {result.precision:.3f}  Recall: {result.recall:.3f}  F1: {result.f1:.3f}")
    tn, fp, fn, tp = result.confusion.ravel()
    print(f"Confusion matrix: TN={tn} FP={fp} FN={fn} TP={tp}")
    if result.detection_latency_epochs is None:
        print("Detection latency: NEVER detected during the spoofed window")
    else:
        lat = result.detection_latency_epochs
        if epoch_period_s:
            print(f"Detection latency: {lat} epochs (~{lat * epoch_period_s:.1f}s after attack start)")
        else:
            print(f"Detection latency: {lat} epochs after attack start")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an autoencoder on clean data and evaluate against labeled scenarios."
    )
    parser.add_argument("--train", nargs="+", required=True)
    parser.add_argument("--eval", nargs="+", required=True)
    parser.add_argument("--bottleneck", type=int, default=3, choices=[2, 3])
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    print(f"Training on {len(args.train)} clean file(s):")
    clean_dfs = []
    for path in args.train:
        df = load_feature_csv(path)
        print(f"  {path}: {len(df)} epochs")
        clean_dfs.append(df)

    detector = train_autoencoder(
        clean_dfs,
        bottleneck_dim=args.bottleneck,
        threshold_percentile=args.threshold_percentile,
        epochs=args.epochs,
    )
    print(f"\nAutoencoder trained. Bottleneck={args.bottleneck}, threshold={detector.threshold:.5f}")

    for path in args.eval:
        df = load_feature_csv(path)
        name = Path(path).stem
        result = evaluate_scenario(detector, df, name=name)
        print_scenario_result(result, epoch_period_s=0.2)
