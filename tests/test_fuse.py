"""
Tests for the Phase 3 fusion module (trajectory_sim + ekf).

All synthetic, no file I/O, fast enough for CI. The thresholds below are
deliberately loose -- these are behavioural sanity checks (does the
filter converge? does dead reckoning drift? does the trust hook do
anything?), not tight numerical regression bounds.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from gnss_integrity.fuse.ekf import ExtendedKalmanFilter, ImuReading
from gnss_integrity.fuse.trajectory_sim import (
    SimConfig,
    make_ground_truth,
    make_gps,
    make_imu,
    simulate,
)


# --- trajectory_sim -------------------------------------------------------


@pytest.mark.parametrize("kind", ["line", "turns"])
def test_ground_truth_shape(kind: str) -> None:
    cfg = replace(SimConfig(), duration_s=20.0, imu_rate_hz=50.0)
    gt = make_ground_truth(kind, cfg)

    expected = int(round(cfg.duration_s * cfg.imu_rate_hz)) + 1
    assert gt.t.shape == (expected,)
    for arr in (gt.x, gt.y, gt.vx, gt.vy, gt.heading):
        assert arr.shape == (expected,)
    assert gt.dt == pytest.approx(1.0 / cfg.imu_rate_hz)
    # The path should actually go somewhere.
    assert np.hypot(np.ptp(gt.x), np.ptp(gt.y)) > 50.0


def test_imu_noise_is_actually_added() -> None:
    cfg = replace(SimConfig(), duration_s=15.0)
    gt = make_ground_truth("turns", cfg)

    noiseless = replace(
        cfg,
        accel_noise_std=0.0,
        accel_bias_init_std=0.0,
        accel_bias_walk_std=0.0,
        gyro_noise_std=0.0,
        gyro_bias_init_std=0.0,
        gyro_bias_walk_std=0.0,
    )
    imu_clean = make_imu(gt, noiseless)
    imu_noisy = make_imu(gt, cfg)

    # With every noise/bias term zeroed the injected bias array is exactly 0.
    assert np.allclose(imu_clean.accel_bias_true, 0.0)
    assert np.allclose(imu_clean.yaw_bias_true, 0.0)
    # Turning the terms back on must perturb the measured stream.
    assert not np.allclose(imu_clean.accel_body, imu_noisy.accel_body)
    assert not np.allclose(imu_clean.yaw_rate, imu_noisy.yaw_rate)
    # Bias should drift, i.e. not be a single constant offset over time.
    assert imu_noisy.accel_bias_true[:, 0].std() > 0.0


def test_gps_noise_is_actually_added() -> None:
    cfg = replace(SimConfig(), duration_s=30.0)
    gt = make_ground_truth("line", cfg)

    gps_clean = make_gps(gt, replace(cfg, gps_noise_std=0.0))
    gps_noisy = make_gps(gt, cfg)

    true_x = np.interp(gps_clean.t, gt.t, gt.x)
    assert np.allclose(gps_clean.x, true_x, atol=1e-9)
    assert not np.allclose(gps_noisy.x, true_x)


def test_dropout_window_withholds_fixes() -> None:
    cfg = replace(SimConfig(), duration_s=60.0)
    gt = make_ground_truth("turns", cfg)
    window = (20.0, 35.0)
    gps = make_gps(gt, cfg, dropout_windows=[window])

    inside = (gps.t >= window[0]) & (gps.t <= window[1])
    assert not gps.available[inside].any()
    assert gps.available[~inside].all()
    # Withheld fixes must be unusable, not silently zero.
    assert np.isnan(gps.x[inside]).all()
    assert np.isfinite(gps.x[~inside]).all()


def test_degraded_window_marks_and_worsens_fixes() -> None:
    cfg = replace(SimConfig(), duration_s=60.0)
    gt = make_ground_truth("line", cfg)
    window = (20.0, 35.0)
    gps = make_gps(gt, cfg, degraded_windows=[window])

    inside = (gps.t >= window[0]) & (gps.t <= window[1])
    assert gps.degraded[inside].all()
    assert not gps.degraded[~inside].any()
    assert gps.available.all()  # degraded != withheld

    true_x = np.interp(gps.t, gt.t, gt.x)
    true_y = np.interp(gps.t, gt.t, gt.y)
    err_inside = np.hypot(gps.x[inside] - true_x[inside], gps.y[inside] - true_y[inside])
    err_outside = np.hypot(
        gps.x[~inside] - true_x[~inside], gps.y[~inside] - true_y[~inside]
    )
    # Degraded fixes carry a big bias + extra noise -> much larger error.
    assert err_inside.mean() > 5.0 * err_outside.mean()


# --- ekf ----------------------------------------------------------------


def _imu_reading(imu, i: int) -> ImuReading:
    return ImuReading(
        accel_x=float(imu.accel_body[i, 0]),
        accel_y=float(imu.accel_body[i, 1]),
        yaw_rate=float(imu.yaw_rate[i]),
    )


def test_predict_only_dead_reckoning_drifts() -> None:
    """No GPS at all -> integration drift should grow over time."""
    cfg = replace(SimConfig(), duration_s=60.0)
    result = simulate("turns", cfg)
    gt, imu = result.ground_truth, result.imu

    ekf = ExtendedKalmanFilter(
        initial_state=[gt.x[0], gt.y[0], gt.vx[0], gt.vy[0], gt.heading[0]]
    )
    errors = []
    for i in range(1, len(gt.t)):
        ekf.predict(gt.dt, _imu_reading(imu, i))
        errors.append(np.linalg.norm(ekf.position - [gt.x[i], gt.y[i]]))
    errors = np.array(errors)

    early = errors[: len(errors) // 5].mean()
    late = errors[-len(errors) // 5 :].mean()
    # Consistent with unbounded dead-reckoning drift: later error is
    # clearly larger, and by the end we are well off truth.
    assert late > early
    assert late > 1.0
    # Covariance must also grow (filter knows it is losing confidence).
    assert ekf.position_covariance.trace() > ExtendedKalmanFilter(
        initial_state=np.zeros(5)
    ).position_covariance.trace()


def test_regular_clean_gps_keeps_filter_close() -> None:
    """Predict at IMU rate + clean GPS at 1 Hz -> small tracking error."""
    cfg = replace(SimConfig(), duration_s=90.0)
    result = simulate("turns", cfg)
    gt, imu, gps = result.ground_truth, result.imu, result.gps

    ekf = ExtendedKalmanFilter(
        initial_state=[gt.x[0], gt.y[0], gt.vx[0], gt.vy[0], gt.heading[0]],
        gps_position_noise_std=cfg.gps_noise_std,
    )
    gps_ptr = 0
    errors = []
    for i in range(1, len(gt.t)):
        ekf.predict(gt.dt, _imu_reading(imu, i))
        t = gt.t[i]
        while gps_ptr < len(gps.t) and gps.t[gps_ptr] <= t + gt.dt / 2:
            if gps.available[gps_ptr]:
                ekf.update_gps([gps.x[gps_ptr], gps.y[gps_ptr]])
            gps_ptr += 1
        errors.append(np.linalg.norm(ekf.position - [gt.x[i], gt.y[i]]))

    mean_err = float(np.mean(errors))

    # Raw GPS mean Euclidean error for comparison (per-axis noise 3 m ->
    # ~3.8 m mean radial). A working filter must beat that and stay
    # comfortably bounded.
    got = gps.available
    raw_err = float(np.mean(np.hypot(
        gps.x[got] - np.interp(gps.t[got], gt.t, gt.x),
        gps.y[got] - np.interp(gps.t[got], gt.t, gt.y),
    )))
    assert mean_err < raw_err
    assert mean_err < 4.0


def test_trust_weight_scales_measurement_pull() -> None:
    """Low trust weight -> update barely moves the state; high -> snaps to it."""
    measurement = [100.0, 100.0]

    low = ExtendedKalmanFilter(initial_state=np.zeros(5))
    low.update_gps(measurement, trust_weight=1e-3)
    moved_low = np.linalg.norm(low.position - [0.0, 0.0])

    high = ExtendedKalmanFilter(initial_state=np.zeros(5))
    high.update_gps(measurement, trust_weight=1e3)
    dist_high = np.linalg.norm(high.position - measurement)

    assert moved_low < 1.0          # near-ignored
    assert dist_high < 1.0          # near-snapped to the fix
    # And a nominal-weight update should land between the two extremes.
    nominal = ExtendedKalmanFilter(initial_state=np.zeros(5))
    nominal.update_gps(measurement, trust_weight=1.0)
    moved_nominal = np.linalg.norm(nominal.position - [0.0, 0.0])
    assert moved_low < moved_nominal < np.linalg.norm(np.array(measurement))


def test_trust_weight_zero_is_safe() -> None:
    """trust_weight=0 must mean 'ignore', not blow up."""
    ekf = ExtendedKalmanFilter(initial_state=np.zeros(5))
    ekf.update_gps([50.0, -30.0], trust_weight=0.0)
    assert np.all(np.isfinite(ekf.state))
    assert np.linalg.norm(ekf.position) < 1.0


def test_validation_scenario_beats_raw_gps_through_dropout() -> None:
    """The headline Phase 3 acceptance check, run directly."""
    from gnss_integrity.fuse.validate import run_ekf, report

    cfg = replace(SimConfig(), duration_s=120.0, seed=42)
    result = simulate(
        "turns", cfg,
        dropout_windows=[(40.0, 55.0)],
        degraded_windows=[(85.0, 95.0)],
    )
    fused, held_gps = run_ekf(result)
    metrics = report(result, fused, held_gps)

    assert metrics["fused_rmse_in_dropout"] < metrics["held_gps_rmse_in_dropout"]
    assert metrics["fused_rmse_overall"] < metrics["raw_gps_rmse_at_fixes"]
