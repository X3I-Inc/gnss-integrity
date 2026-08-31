"""
Isolation Forest anomaly detector.

Trains scikit-learn's IsolationForest on clean-only GNSS feature data
(from nmea_loader.py and/or texbat_loader.py), then evaluates it against
labeled scenarios to see how well it separates clean epochs from
spoofed/jammed ones.

This module is deliberately thin: all actual modeling comes from
scikit-learn (IsolationForest for detection, StandardScaler for feature
normalization, precision/recall/confusion-matrix for evaluation). No
custom detection math is implemented here -- only the glue code needed
to load this project's specific CSV schema and report results broken
down by scenario, per the build plan.

Feature choice:
    Only columns that are semantically comparable across BOTH the NMEA
    and TEXBAT loaders are used for training/scoring:
        mean_snr, max_snr, var_snr, tracked_satellites,
        delta_mean_snr, rolling_var_snr
    hdop/pdop/fix_quality are deliberately excluded: the NMEA loader
    reports real HDOP/PDOP from the receiver, while the TEXBAT loader
    can only report a coarse discrete proxy (see texbat_loader.py) --
    mixing those would confound the model with a spurious signal that
    has nothing to do with actual GNSS integrity.

Ground truth:
    Evaluation expects a `known_spoof_window` column (0/1) as produced
    by the current texbat_loader.py. Clean-only sources (your own NMEA
    logs, TEXBAT cleanStatic) are assumed to have no spoofed epochs at
    all when used purely for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

# Feature columns shared by both nmea_loader.py and texbat_loader.py output.
FEATURE_COLUMNS = [
    "mean_snr",
    "max_snr",
    "var_snr",
    "tracked_satellites",
    "delta_mean_snr",
    "rolling_var_snr",
]


@dataclass
class TrainedDetector:
    scaler: StandardScaler
    model: IsolationForest
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
    detection_latency_epochs: int | None  # epochs from true attack start to
                                            # first correct detection, or
                                            # None if never detected


def load_feature_csv(path: str | Path) -> pd.DataFrame:
    """Load a feature CSV produced by nmea_loader.py or texbat_loader.py."""
    return pd.read_csv(path)


def train_detector(
    clean_dfs: list[pd.DataFrame],
    contamination: float = 0.01,
    random_state: int = 42,
) -> TrainedDetector:
    """
    Fit a StandardScaler + IsolationForest on concatenated clean-only data.

    contamination=0.01 (sklearn's IsolationForest default region) assumes
    training data is "mostly" clean; since we're training on data we
    believe to be entirely clean, this just controls the internal
    decision threshold rather than expressing a known outlier fraction.
    """
    combined = pd.concat(clean_dfs, ignore_index=True)
    X = combined[FEATURE_COLUMNS].to_numpy()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=random_state,
    )
    model.fit(X_scaled)

    return TrainedDetector(scaler=scaler, model=model, feature_columns=FEATURE_COLUMNS)


def score_dataframe(detector: TrainedDetector, df: pd.DataFrame) -> np.ndarray:
    """
    Run the trained detector on a DataFrame, returning a 0/1 array where
    1 means "flagged as anomalous" (matches the `known_spoof_window`
    convention: 1 = spoofed/anomalous, 0 = clean).
    """
    X = df[detector.feature_columns].to_numpy()
    X_scaled = detector.scaler.transform(X)
    # sklearn's IsolationForest.predict returns -1 for outliers, 1 for
    # inliers -- flip that to our 1=anomalous, 0=clean convention.
    raw = detector.model.predict(X_scaled)
    return (raw == -1).astype(int)


def evaluate_scenario(
    detector: TrainedDetector,
    df: pd.DataFrame,
    name: str,
    label_column: str = "known_spoof_window",
) -> ScenarioResult:
    """
    Score a labeled scenario and compute precision/recall/F1/confusion
    matrix against its ground-truth spoofing label, plus detection
    latency: how many epochs after the true attack start the detector
    first correctly flags an anomaly.
    """
    if label_column not in df.columns:
        raise ValueError(
            f"'{label_column}' not found in {name} -- re-run texbat_loader.py "
            f"with the current version to get ground-truth labels."
        )

    y_true = df[label_column].to_numpy()
    if (y_true == -1).any():
        raise ValueError(
            f"{name} has unknown ({-1}) ground-truth labels -- scenario name "
            f"wasn't recognized when the CSV was generated. Re-run the loader "
            f"from a properly named folder (ds1..ds8, cleanStatic, etc.)."
        )

    y_pred = score_dataframe(detector, df)

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Detection latency: find the first true-spoofed epoch, then count
    # forward until the first epoch the detector ALSO flags as spoofed.
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
        description="Train an Isolation Forest on clean data and evaluate against labeled scenarios."
    )
    parser.add_argument(
        "--train", nargs="+", required=True,
        help="One or more feature CSVs of CLEAN-ONLY data (e.g. your NMEA log, TEXBAT cleanStatic)",
    )
    parser.add_argument(
        "--eval", nargs="+", required=True,
        help="One or more feature CSVs with a known_spoof_window ground-truth column (e.g. ds2, ds3, ds7)",
    )
    parser.add_argument("--contamination", type=float, default=0.01)
    args = parser.parse_args()

    print(f"Training on {len(args.train)} clean file(s):")
    clean_dfs = []
    for path in args.train:
        df = load_feature_csv(path)
        print(f"  {path}: {len(df)} epochs")
        clean_dfs.append(df)

    detector = train_detector(clean_dfs, contamination=args.contamination)
    print("\nDetector trained.")

    for path in args.eval:
        df = load_feature_csv(path)
        name = Path(path).stem
        result = evaluate_scenario(detector, df, name=name)
        print_scenario_result(result, epoch_period_s=0.2)
