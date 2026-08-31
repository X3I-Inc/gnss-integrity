"""
End-to-end validation of the Phase 3 EKF on synthetic data.

This is the Phase 3 deliverable: it answers, with numbers and a plot,
"does the fused GPS+IMU estimate track ground truth better than raw GPS
alone -- especially through a GPS dropout?"

What it does
------------
1. Generates a synthetic trajectory + IMU + GPS stream (trajectory_sim),
   with an injected GPS dropout window (fixes withheld) and a degraded
   window (fixes present but noisy/biased).
2. Runs the EKF over the whole stream: predict() at the IMU rate,
   update_gps() at the GPS rate whenever a fix is available. Inside the
   dropout window there is simply no fix to apply. Inside the degraded
   window the fix is applied at a *fixed* reduced trust weight -- this is
   NOT the detector (Phase 4); it just shows the trust hook works and
   keeps a known-bad fix from dragging the estimate off.
3. Prints RMSE of fused-vs-truth and raw-GPS-vs-truth, overall and
   restricted to the dropout window (where "raw GPS" means "hold the
   last fix", the only honest baseline when GPS is gone).
4. Saves a plot: ground-truth path, raw GPS fixes, fused estimate, with
   the dropout/degraded windows marked, plus a position-error-vs-time
   panel.

Usage (same -o/--output convention as scripts/plot_reconstruction_error.py):

    python -m gnss_integrity.fuse.validate -o fuse_validation.png
    python src/gnss_integrity/fuse/validate.py --kind turns -o fuse_validation.png
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow running this file directly (python src/gnss_integrity/fuse/validate.py)
# as well as via -m; mirrors the pattern in scripts/plot_reconstruction_error.py.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from gnss_integrity.fuse.ekf import ExtendedKalmanFilter, ImuReading
from gnss_integrity.fuse.trajectory_sim import (
    SimConfig,
    SimulationResult,
    simulate,
)

# Fixed trust weight applied to fixes inside the degraded window. Constant,
# not detector-driven -- Phase 4 replaces this constant with the detector's
# per-epoch confidence.
_DEGRADED_TRUST_WEIGHT = 0.05


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    """RMS Euclidean distance between two (N, 2) position arrays."""
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def run_ekf(result: SimulationResult) -> tuple[np.ndarray, np.ndarray]:
    """
    Run the filter over the full stream.

    Returns:
        fused    : (N, 2) fused position estimate at every IMU step.
        held_gps : (N, 2) "last known GPS fix, held constant" baseline at
                   every IMU step (raw GPS with no inertial help).
    """
    gt = result.ground_truth
    imu = result.imu
    gps = result.gps
    n = len(gt.t)

    ekf = ExtendedKalmanFilter(
        initial_state=[gt.x[0], gt.y[0], gt.vx[0], gt.vy[0], gt.heading[0]],
        gps_position_noise_std=result.config.gps_noise_std,
    )

    fused = np.empty((n, 2))
    held_gps = np.empty((n, 2))

    dt = gt.dt
    gps_ptr = 0
    last_fix = np.array([gps.x[0], gps.y[0]], dtype=float)

    for i in range(n):
        if i > 0:
            ekf.predict(
                dt,
                ImuReading(
                    accel_x=float(imu.accel_body[i, 0]),
                    accel_y=float(imu.accel_body[i, 1]),
                    yaw_rate=float(imu.yaw_rate[i]),
                ),
            )

        t = imu.t[i]
        # Apply any GPS fix whose timestamp lines up with this IMU step.
        while gps_ptr < len(gps.t) and gps.t[gps_ptr] <= t + dt / 2:
            if gps.available[gps_ptr]:
                fix = np.array([gps.x[gps_ptr], gps.y[gps_ptr]], dtype=float)
                last_fix = fix
                if gps.degraded[gps_ptr]:
                    ekf.update_gps(fix, trust_weight=_DEGRADED_TRUST_WEIGHT)
                else:
                    ekf.update_gps(fix)  # nominal trust_weight = 1.0
            gps_ptr += 1

        fused[i] = ekf.position
        held_gps[i] = last_fix

    return fused, held_gps


def _window_mask(t: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros_like(t, dtype=bool)
    for start, end in windows:
        mask |= (t >= start) & (t <= end)
    return mask


def report(result: SimulationResult, fused: np.ndarray, held_gps: np.ndarray) -> dict:
    gt = result.ground_truth
    gps = result.gps
    truth = gt.position()

    # Raw GPS vs truth, at the fix times that actually delivered a fix.
    got_fix = gps.available
    gps_xy = np.column_stack([gps.x[got_fix], gps.y[got_fix]])
    truth_at_fix = np.column_stack([
        np.interp(gps.t[got_fix], gt.t, gt.x),
        np.interp(gps.t[got_fix], gt.t, gt.y),
    ])
    # Same, excluding the degraded window -- "how good is GPS when nothing
    # is wrong with it", the fairest like-for-like against the filter's
    # clean-tracking performance.
    nominal_fix = gps.available & ~gps.degraded
    gps_xy_nom = np.column_stack([gps.x[nominal_fix], gps.y[nominal_fix]])
    truth_at_nom = np.column_stack([
        np.interp(gps.t[nominal_fix], gt.t, gt.x),
        np.interp(gps.t[nominal_fix], gt.t, gt.y),
    ])

    drop_mask = _window_mask(gt.t, result.dropout_windows)

    metrics = {
        "fused_rmse_overall": _rmse(fused, truth),
        "raw_gps_rmse_at_fixes": _rmse(gps_xy, truth_at_fix),
        "raw_gps_rmse_nominal_only": _rmse(gps_xy_nom, truth_at_nom),
        "fused_rmse_in_dropout": _rmse(fused[drop_mask], truth[drop_mask])
        if drop_mask.any() else float("nan"),
        "held_gps_rmse_in_dropout": _rmse(held_gps[drop_mask], truth[drop_mask])
        if drop_mask.any() else float("nan"),
        "fused_max_err_in_dropout": float(
            np.max(np.linalg.norm(fused[drop_mask] - truth[drop_mask], axis=1))
        ) if drop_mask.any() else float("nan"),
        "held_gps_max_err_in_dropout": float(
            np.max(np.linalg.norm(held_gps[drop_mask] - truth[drop_mask], axis=1))
        ) if drop_mask.any() else float("nan"),
    }

    print("\n=== Phase 3 EKF validation ===")
    print(f"trajectory: {len(gt.t)} IMU samples @ {1/gt.dt:.0f} Hz, "
          f"{len(gps.t)} GPS fixes @ {result.config.gps_rate_hz:.0f} Hz")
    print(f"dropout window(s):  {result.dropout_windows}")
    print(f"degraded window(s): {result.degraded_windows} "
          f"(applied at fixed trust_weight={_DEGRADED_TRUST_WEIGHT})")
    print()
    print(f"RMSE raw GPS vs truth (all delivered fixes) : "
          f"{metrics['raw_gps_rmse_at_fixes']:.2f} m")
    print(f"RMSE raw GPS vs truth (nominal fixes only)  : "
          f"{metrics['raw_gps_rmse_nominal_only']:.2f} m")
    print(f"RMSE fused vs truth (overall)               : "
          f"{metrics['fused_rmse_overall']:.2f} m")
    print()
    print("through the GPS dropout window:")
    print(f"  RMSE fused vs truth        : {metrics['fused_rmse_in_dropout']:.2f} m "
          f"(max {metrics['fused_max_err_in_dropout']:.2f} m)")
    print(f"  RMSE held-last GPS vs truth : {metrics['held_gps_rmse_in_dropout']:.2f} m "
          f"(max {metrics['held_gps_max_err_in_dropout']:.2f} m)")

    if drop_mask.any():
        better = metrics["fused_rmse_in_dropout"] < metrics["held_gps_rmse_in_dropout"]
        ratio = metrics["held_gps_rmse_in_dropout"] / max(
            metrics["fused_rmse_in_dropout"], 1e-9
        )
        verdict = "PASS" if better else "FAIL"
        print(f"\n  [{verdict}] fused is {ratio:.1f}x closer to truth than raw GPS "
              f"through the dropout")
    return metrics


def make_plot(
    result: SimulationResult,
    fused: np.ndarray,
    held_gps: np.ndarray,
    out_path: str,
) -> None:
    gt = result.ground_truth
    gps = result.gps
    truth = gt.position()

    drop_mask_t = _window_mask(gt.t, result.dropout_windows)
    degr_mask_t = _window_mask(gt.t, result.degraded_windows)

    fig, (ax_path, ax_err) = plt.subplots(
        2, 1, figsize=(11, 10), gridspec_kw={"height_ratios": [3, 2]}
    )

    # --- spatial path panel ---
    ax_path.plot(truth[:, 0], truth[:, 1], color="#999999", linewidth=1.2,
                 label="Ground truth", zorder=1)
    # Highlight the stretch of true path traversed while GPS was out.
    if drop_mask_t.any():
        ax_path.plot(truth[drop_mask_t, 0], truth[drop_mask_t, 1],
                     color="red", linewidth=3.0, alpha=0.5,
                     label="Truth during GPS dropout", zorder=2)
    if degr_mask_t.any():
        ax_path.plot(truth[degr_mask_t, 0], truth[degr_mask_t, 1],
                     color="orange", linewidth=3.0, alpha=0.5,
                     label="Truth during degraded GPS", zorder=2)

    nominal = gps.available & ~gps.degraded
    ax_path.scatter(gps.x[nominal], gps.y[nominal], s=14, color="#1f77b4",
                    alpha=0.6, label="Raw GPS fix", zorder=3)
    if gps.degraded.any():
        ax_path.scatter(gps.x[gps.degraded], gps.y[gps.degraded], s=26,
                        color="orange", edgecolor="k", linewidth=0.4,
                        label="Raw GPS fix (degraded)", zorder=4)

    ax_path.plot(fused[:, 0], fused[:, 1], color="#2ca02c", linewidth=1.6,
                 label="Fused EKF estimate", zorder=5)

    ax_path.set_xlabel("east (m)")
    ax_path.set_ylabel("north (m)")
    ax_path.set_title("GPS/IMU fusion vs raw GPS -- synthetic trajectory")
    ax_path.legend(loc="best", fontsize=8)
    ax_path.set_aspect("equal", adjustable="datalim")

    # --- error-vs-time panel ---
    t = gt.t
    fused_err = np.linalg.norm(fused - truth, axis=1)
    held_err = np.linalg.norm(held_gps - truth, axis=1)

    for start, end in result.dropout_windows:
        ax_err.axvspan(start, end, color="red", alpha=0.10,
                       label="GPS dropout")
    for start, end in result.degraded_windows:
        ax_err.axvspan(start, end, color="orange", alpha=0.12,
                       label="GPS degraded")
    ax_err.plot(t, held_err, color="#1f77b4", linewidth=1.0,
                label="raw GPS (held last fix) error")
    ax_err.plot(t, fused_err, color="#2ca02c", linewidth=1.4,
                label="fused EKF error")

    ax_err.set_xlabel("time since start (s)")
    ax_err.set_ylabel("position error (m)")
    ax_err.set_title("Position error over time")
    # De-duplicate axvspan labels.
    handles, labels = ax_err.get_legend_handles_labels()
    seen: dict[str, object] = {}
    for h, lab in zip(handles, labels):
        seen.setdefault(lab, h)
    ax_err.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


def run_validation(
    kind: str,
    out_path: str,
    duration_s: float,
    dropout: tuple[float, float],
    degraded: tuple[float, float],
    seed: int,
) -> dict:
    cfg = replace(SimConfig(), duration_s=duration_s, seed=seed)
    result = simulate(
        kind=kind,
        config=cfg,
        dropout_windows=[dropout],
        degraded_windows=[degraded],
    )
    fused, held_gps = run_ekf(result)
    metrics = report(result, fused, held_gps)
    make_plot(result, fused, held_gps, out_path)
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Validate the Phase 3 GPS/IMU EKF on a synthetic trajectory."
    )
    parser.add_argument("--kind", choices=["line", "turns"], default="turns")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument("--dropout", nargs=2, type=float, metavar=("START", "END"),
                        default=[40.0, 55.0],
                        help="GPS dropout window, seconds (fixes withheld)")
    parser.add_argument("--degraded", nargs=2, type=float, metavar=("START", "END"),
                        default=[85.0, 95.0],
                        help="GPS degraded window, seconds (noisy/biased fixes)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default="fuse_validation.png")
    args = parser.parse_args()

    run_validation(
        kind=args.kind,
        out_path=args.output,
        duration_s=args.duration,
        dropout=tuple(args.dropout),
        degraded=tuple(args.degraded),
        seed=args.seed,
    )
