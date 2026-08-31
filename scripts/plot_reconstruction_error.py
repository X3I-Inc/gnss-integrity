"""
Plot autoencoder reconstruction error (MSE) over time for one scenario.

This is a diagnostic script, not part of the detector itself: it reuses
your existing autoencoder.py training/scoring code and just adds a
matplotlib visualization on top, so you can *see* why ds7's fast
latency (0.2s) coexists with its low recall (76.9%) -- the headline
number alone doesn't show whether the model catches the transition and
then loses track, or catches it late and never fully recovers.

Usage (same pattern as autoencoder.py's own CLI):

    python.exe scripts\\plot_reconstruction_error.py ^
        --train data\\texbat\\cleanStatic\\features.csv ^
        --eval data\\texbat\\ds7\\features.csv ^
        -o ds7_reconstruction_error.png

Requires matplotlib (already in your installed libs from the earlier
pip command).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Make sure the repo's src/ layout is importable regardless of where this
# script is invoked from (it lives in scripts/, not src/, so it isn't on
# sys.path by default).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from gnss_integrity.detect.autoencoder import (
    FEATURE_COLUMNS,
    load_feature_csv,
    score_dataframe,
    train_autoencoder,
)


def plot_reconstruction_error(
    train_paths: list[str],
    eval_path: str,
    out_path: str,
    bottleneck_dim: int = 3,
    threshold_percentile: float = 95.0,
    epochs: int = 50,
    label_column: str = "known_spoof_window",
    time_column: str = "fix_time",
) -> None:
    print(f"Training on {len(train_paths)} clean file(s):")
    clean_dfs = []
    for path in train_paths:
        df = load_feature_csv(path)
        print(f"  {path}: {len(df)} epochs")
        clean_dfs.append(df)

    detector = train_autoencoder(
        clean_dfs,
        bottleneck_dim=bottleneck_dim,
        threshold_percentile=threshold_percentile,
        epochs=epochs,
    )
    print(f"Autoencoder trained. Threshold={detector.threshold:.5f}")

    eval_df = load_feature_csv(eval_path)
    mse, y_pred = score_dataframe(detector, eval_df)

    if label_column not in eval_df.columns:
        raise ValueError(
            f"'{label_column}' not found in {eval_path} -- can't shade the "
            f"true attack window without it."
        )
    y_true = eval_df[label_column].to_numpy()

    # x-axis: prefer the real fix_time column if present, else fall back
    # to a plain epoch index (0, 1, 2, ...).
    if time_column in eval_df.columns:
        t = eval_df[time_column].to_numpy()
        t = t - t[0]  # relative seconds from scenario start
        x_label = "Time since scenario start (s)"
    else:
        t = np.arange(len(eval_df))
        x_label = "Epoch index"

    attack_indices = np.where(y_true == 1)[0]
    flagged_indices = np.where(y_pred == 1)[0]

    fig, ax = plt.subplots(figsize=(11, 5))

    # Shade the true spoofed window.
    if attack_indices.size > 0:
        ax.axvspan(
            t[attack_indices[0]], t[attack_indices[-1]],
            color="red", alpha=0.08, label="True spoofed window",
        )
        ax.axvline(t[attack_indices[0]], color="red", linestyle="--", linewidth=1,
                    label="Attack start")

    # Reconstruction error line.
    ax.plot(t, mse, color="#1f77b4", linewidth=1.2, label="Reconstruction MSE")

    # Threshold.
    ax.axhline(detector.threshold, color="black", linestyle=":", linewidth=1,
                label=f"Threshold ({detector.threshold:.3f})")

    # Mark epochs the model actually flagged as anomalous.
    if flagged_indices.size > 0:
        ax.scatter(
            t[flagged_indices], mse[flagged_indices],
            color="darkorange", s=10, zorder=5, label="Flagged by model",
        )

    ax.set_xlabel(x_label)
    ax.set_ylabel("Reconstruction MSE")
    ax.set_title("Autoencoder reconstruction error over time -- ds7 (subtle attack)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")

    # Quick numeric summary to complement the plot.
    n_attack = attack_indices.size
    n_flagged_during_attack = np.intersect1d(attack_indices, flagged_indices).size
    if n_attack > 0:
        print(f"Flagged {n_flagged_during_attack}/{n_attack} "
              f"({100 * n_flagged_during_attack / n_attack:.1f}%) of true-spoofed epochs.")
        # Where in the attack window does it stop catching things?
        missed_after_first_catch = [
            i for i in attack_indices if i > flagged_indices[0]
        ] if flagged_indices.size > 0 else []
        missed_and_unflagged = [i for i in missed_after_first_catch if y_pred[i] == 0]
        if missed_and_unflagged:
            print(f"After first detection, {len(missed_and_unflagged)} later "
                  f"spoofed epochs were NOT flagged -- i.e. the model catches "
                  f"the transition, then loses track for stretches.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot autoencoder reconstruction error over time for one scenario."
    )
    parser.add_argument("--train", nargs="+", required=True,
                         help="Clean-only feature CSV(s), e.g. TEXBAT cleanStatic")
    parser.add_argument("--eval", required=True,
                         help="Single feature CSV to plot, e.g. ds7 features.csv")
    parser.add_argument("-o", "--output", default="reconstruction_error.png")
    parser.add_argument("--bottleneck", type=int, default=3, choices=[2, 3])
    parser.add_argument("--threshold-percentile", type=float, default=95.0)
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    plot_reconstruction_error(
        train_paths=args.train,
        eval_path=args.eval,
        out_path=args.output,
        bottleneck_dim=args.bottleneck,
        threshold_percentile=args.threshold_percentile,
        epochs=args.epochs,
    )