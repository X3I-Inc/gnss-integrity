"""
Synthetic trajectory + sensor simulator for EKF development.

Phase 3 (GPS/IMU fusion) needs paired GPS+IMU data with a known
ground truth, and no such real dataset exists in this project: TEXBAT
and the other GNSS sources are GPS-only, with no inertial channel. The
project's architecture doc calls for exactly this approach -- simulate a
ground-truth path, derive synthetic IMU and GPS streams from it (with
realistic noise, bias and drift), then validate the filter's fused
output against the known truth.

What this module produces
-------------------------
* GroundTruth : the true 2D trajectory (position, velocity, heading)
  sampled at the IMU rate.
* ImuStream   : body-frame linear acceleration + yaw rate at the IMU
  rate, with white noise and a slowly drifting (random-walk) bias.
  Bias drift -- not white noise -- is the dominant real-world error that
  dead reckoning has to fight, so it is modelled explicitly.
* GpsStream   : noisy position fixes at the (much lower) GPS rate, on a
  fixed time grid, with two kinds of injectable fault window:
      - withheld : no fix at all (signal-loss dropout)
      - degraded : a fix is present but with large extra noise plus a
                   bias offset (jamming / spoofing-style bad fix)
  Both are driven by plain ``list[tuple[float, float]]`` start/end time
  windows, so Phase 4 can inject an attack with a single argument.

What this module deliberately does NOT do
-----------------------------------------
* No 3D / altitude / Earth-curvature modelling. This is a flat local
  ENU plane in metres, which is all the 2D EKF needs.
* No attempt to match a specific IMU/GPS part number. Noise magnitudes
  are labelled, order-of-magnitude consumer-MEMS / civilian-SPS values.
* No filtering or fusion -- that is ekf.py's job.

Frame convention
----------------
World frame is a fixed 2D plane, x = east, y = north, metres. Heading
(theta) is measured counter-clockwise from the +x axis, radians, and is
kept unwrapped (continuous) so differentiation is well behaved. IMU
acceleration is reported in the body frame (x = forward / along-heading,
y = left / lateral), which is what a strapdown IMU actually measures.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

# --- default sensor characteristics -----------------------------------------
# All values are deliberately generic, order-of-magnitude figures for a
# low-cost consumer MEMS IMU and an unaided civilian single-frequency GPS.
# They are NOT calibrated to a specific device -- they only need to be the
# right size for the filter to be exercised realistically.


@dataclass
class SimConfig:
    duration_s: float = 120.0
    imu_rate_hz: float = 50.0          # strapdown IMUs vastly outpace GPS
    gps_rate_hz: float = 1.0           # typical consumer GPS fix rate

    # Accelerometer: white noise + a turn-on bias that then random-walks.
    accel_noise_std: float = 0.05         # m/s^2, per-axis white noise
    accel_bias_init_std: float = 0.02     # m/s^2, per-axis turn-on bias
    accel_bias_walk_std: float = 2.0e-3   # m/s^2 / sqrt(s), bias drift rate

    # Gyro (yaw only, since the trajectory is planar).
    gyro_noise_std: float = 2.0e-3        # rad/s, white noise (~0.11 deg/s)
    gyro_bias_init_std: float = 1.0e-3    # rad/s, turn-on bias
    gyro_bias_walk_std: float = 5.0e-5    # rad/s / sqrt(s), bias drift rate

    # GPS position fix.
    gps_noise_std: float = 3.0            # m, per-axis nominal fix noise
    gps_degraded_noise_std: float = 25.0  # m, extra noise inside a degraded window
    gps_degraded_bias: tuple[float, float] = (30.0, -20.0)  # m, offset injected
                                                            # inside a degraded window

    seed: int | None = 42


@dataclass
class GroundTruth:
    t: np.ndarray        # (N,) seconds from start
    x: np.ndarray        # (N,) east position, m
    y: np.ndarray        # (N,) north position, m
    vx: np.ndarray       # (N,) east velocity, m/s
    vy: np.ndarray       # (N,) north velocity, m/s
    heading: np.ndarray  # (N,) unwrapped heading, rad
    dt: float            # IMU sample period, s

    def position(self) -> np.ndarray:
        """(N, 2) stacked [x, y] -- convenient for error metrics/plots."""
        return np.column_stack([self.x, self.y])


@dataclass
class ImuStream:
    t: np.ndarray              # (N,) seconds, same grid as GroundTruth
    accel_body: np.ndarray     # (N, 2) measured [forward, lateral] accel, m/s^2
    yaw_rate: np.ndarray       # (N,) measured yaw rate, rad/s
    accel_bias_true: np.ndarray  # (N, 2) bias actually injected (diagnostics only)
    yaw_bias_true: np.ndarray    # (N,) bias actually injected (diagnostics only)
    dt: float


@dataclass
class GpsStream:
    t: np.ndarray          # (M,) fix times on a fixed grid
    x: np.ndarray          # (M,) measured east position, m (NaN where withheld)
    y: np.ndarray          # (M,) measured north position, m (NaN where withheld)
    available: np.ndarray  # (M,) bool -- False inside a withheld/dropout window
    degraded: np.ndarray   # (M,) bool -- True inside a degraded window


@dataclass
class SimulationResult:
    ground_truth: GroundTruth
    imu: ImuStream
    gps: GpsStream
    config: SimConfig
    dropout_windows: list[tuple[float, float]] = field(default_factory=list)
    degraded_windows: list[tuple[float, float]] = field(default_factory=list)


# --- ground-truth trajectory generation -----------------------------------


def _integrate_ctrv(
    speed: np.ndarray,
    yaw_rate: np.ndarray,
    dt: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Forward-Euler integrate a constant-turn-rate/velocity style profile.

    ``speed`` and ``yaw_rate`` are per-sample commanded values. Euler
    integration at the IMU rate (50 Hz here) is more than accurate enough
    for a synthetic reference path, and keeps the generator trivial to
    read -- there is no need for RK4 on a path we defined ourselves.
    """
    # Heading is the running integral of yaw rate; keep it unwrapped.
    heading = np.concatenate([[0.0], np.cumsum(yaw_rate[:-1]) * dt])
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    x = np.concatenate([[0.0], np.cumsum(vx[:-1]) * dt])
    y = np.concatenate([[0.0], np.cumsum(vy[:-1]) * dt])
    return x, y, vx, vy, heading


def _time_grid(config: SimConfig) -> np.ndarray:
    dt = 1.0 / config.imu_rate_hz
    n = int(round(config.duration_s * config.imu_rate_hz)) + 1
    return np.arange(n) * dt


def _profile_straightish(t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Sanity-check case: near-constant speed with a very gentle weaving
    curve. Small enough that dead reckoning should track it easily; its
    job is to catch gross filter bugs, not to stress anything.
    """
    speed = np.full_like(t, 12.0)                     # m/s, roughly 43 km/h
    yaw_rate = 0.02 * np.sin(2.0 * np.pi * t / 50.0)  # rad/s, ~1 deg/s peak
    return speed, yaw_rate


def _profile_turns(t: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    More realistic vehicle-like case: piecewise constant-turn-rate /
    constant-velocity segments with a few speed changes. Built from a
    short segment list ``(duration_s, speed_mps, yaw_rate_rps)`` and
    tiled onto the sample grid; the last segment is held to fill any
    remaining time.
    """
    segments = [
        (15.0, 10.0, 0.00),   # straight
        (8.0, 10.0, 0.20),    # sweeping left turn (~92 deg)
        (12.0, 14.0, 0.00),   # straight, accelerated
        (8.0, 14.0, -0.20),   # sweeping right turn
        (10.0, 8.0, 0.00),    # slower straight
        (7.0, 8.0, -0.28),    # sharper right
        (20.0, 11.0, 0.00),   # straight to the end
    ]
    n = len(t)
    speed = np.empty(n)
    yaw_rate = np.empty(n)
    idx = 0
    for dur, spd, yr in segments:
        k = int(round(dur / dt))
        end = min(idx + k, n)
        speed[idx:end] = spd
        yaw_rate[idx:end] = yr
        idx = end
        if idx >= n:
            break
    if idx < n:  # pad with a final straight cruise
        speed[idx:] = segments[-1][1]
        yaw_rate[idx:] = 0.0
    return speed, yaw_rate


def make_ground_truth(kind: str = "turns", config: SimConfig | None = None) -> GroundTruth:
    """
    Generate a ground-truth trajectory.

    kind:
        "line"  -- near-constant-speed gentle curve (sanity check).
        "turns" -- piecewise turns + speed changes (vehicle-like).
    """
    config = config or SimConfig()
    dt = 1.0 / config.imu_rate_hz
    t = _time_grid(config)

    if kind == "line":
        speed, yaw_rate = _profile_straightish(t)
    elif kind == "turns":
        speed, yaw_rate = _profile_turns(t, dt)
    else:
        raise ValueError(f"unknown trajectory kind {kind!r} (use 'line' or 'turns')")

    x, y, vx, vy, heading = _integrate_ctrv(speed, yaw_rate, dt)
    return GroundTruth(t=t, x=x, y=y, vx=vx, vy=vy, heading=heading, dt=dt)


# --- synthetic IMU -------------------------------------------------------


def _random_walk(rng: np.random.Generator, walk_std: float, dt: float, shape) -> np.ndarray:
    """Discrete random walk: cumulative sum of N(0, walk_std * sqrt(dt))."""
    steps = rng.normal(0.0, walk_std * np.sqrt(dt), size=shape)
    return np.cumsum(steps, axis=0)


def make_imu(gt: GroundTruth, config: SimConfig | None = None) -> ImuStream:
    """
    Derive a body-frame IMU stream from the ground truth.

    Truth accelerations come from differentiating the ground-truth
    velocity (world frame), then rotating into the body frame by the
    ground-truth heading -- a plain 2x2 rotation. (scipy's Rotation is
    3D-oriented; for a single planar angle the explicit cos/sin form is
    the standard, clearer choice.) On top of the truth we add:
      * a turn-on bias (constant offset, drawn once),
      * a random-walk bias drift (the term dead reckoning fights),
      * white measurement noise.
    """
    config = config or SimConfig()
    dt = gt.dt
    rng = np.random.default_rng(config.seed)

    # World-frame acceleration from the (already consistent) velocity arrays.
    ax_w = np.gradient(gt.vx, dt)
    ay_w = np.gradient(gt.vy, dt)

    # Rotate world -> body by -heading:  a_body = R(-theta) a_world.
    c = np.cos(gt.heading)
    s = np.sin(gt.heading)
    ax_b = c * ax_w + s * ay_w
    ay_b = -s * ax_w + c * ay_w
    accel_true = np.column_stack([ax_b, ay_b])

    # True yaw rate is the derivative of the (unwrapped) heading.
    yaw_true = np.gradient(gt.heading, dt)

    n = len(gt.t)
    accel_bias = rng.normal(0.0, config.accel_bias_init_std, size=(1, 2))
    accel_bias = accel_bias + _random_walk(rng, config.accel_bias_walk_std, dt, (n, 2))
    yaw_bias = rng.normal(0.0, config.gyro_bias_init_std)
    yaw_bias = yaw_bias + _random_walk(rng, config.gyro_bias_walk_std, dt, (n,))

    accel_meas = (
        accel_true
        + accel_bias
        + rng.normal(0.0, config.accel_noise_std, size=(n, 2))
    )
    yaw_meas = yaw_true + yaw_bias + rng.normal(0.0, config.gyro_noise_std, size=n)

    return ImuStream(
        t=gt.t.copy(),
        accel_body=accel_meas,
        yaw_rate=yaw_meas,
        accel_bias_true=accel_bias,
        yaw_bias_true=yaw_bias,
        dt=dt,
    )


# --- synthetic GPS -----------------------------------------------------


def _in_any_window(times: np.ndarray, windows: list[tuple[float, float]]) -> np.ndarray:
    mask = np.zeros_like(times, dtype=bool)
    for start, end in windows:
        mask |= (times >= start) & (times <= end)
    return mask


def make_gps(
    gt: GroundTruth,
    config: SimConfig | None = None,
    dropout_windows: list[tuple[float, float]] | None = None,
    degraded_windows: list[tuple[float, float]] | None = None,
) -> GpsStream:
    """
    Derive a noisy GPS fix stream from the ground truth.

    dropout_windows : list of (start_s, end_s). Fixes whose timestamp
        falls in one of these are *withheld entirely* -- ``available`` is
        False and x/y are NaN (simulating loss of signal).
    degraded_windows : list of (start_s, end_s). Fixes in these windows
        are still delivered, but with a large extra noise term and a
        fixed bias offset added (simulating a jammed / spoofed fix).
        ``degraded`` is True for those fixes.

    The two are independent; a Phase 4 test can use either or both. This
    is the key interface for later work -- injecting a bad window must be
    a one-argument change, not a code edit.
    """
    config = config or SimConfig()
    dropout_windows = dropout_windows or []
    degraded_windows = degraded_windows or []
    rng = np.random.default_rng(
        None if config.seed is None else config.seed + 1  # decouple from IMU noise
    )

    m = int(round(config.duration_s * config.gps_rate_hz)) + 1
    t_gps = np.arange(m) / config.gps_rate_hz

    # Interpolate the truth onto the GPS time grid.
    true_x = np.interp(t_gps, gt.t, gt.x)
    true_y = np.interp(t_gps, gt.t, gt.y)

    meas_x = true_x + rng.normal(0.0, config.gps_noise_std, size=m)
    meas_y = true_y + rng.normal(0.0, config.gps_noise_std, size=m)

    degraded = _in_any_window(t_gps, degraded_windows)
    if degraded.any():
        k = int(degraded.sum())
        meas_x[degraded] += config.gps_degraded_bias[0] + rng.normal(
            0.0, config.gps_degraded_noise_std, size=k
        )
        meas_y[degraded] += config.gps_degraded_bias[1] + rng.normal(
            0.0, config.gps_degraded_noise_std, size=k
        )

    available = ~_in_any_window(t_gps, dropout_windows)
    meas_x = np.where(available, meas_x, np.nan)
    meas_y = np.where(available, meas_y, np.nan)

    return GpsStream(t=t_gps, x=meas_x, y=meas_y, available=available, degraded=degraded)


# --- one-call bundle -----------------------------------------------------


def simulate(
    kind: str = "turns",
    config: SimConfig | None = None,
    dropout_windows: list[tuple[float, float]] | None = None,
    degraded_windows: list[tuple[float, float]] | None = None,
) -> SimulationResult:
    """Generate ground truth + IMU + GPS in one call, returned as one bundle."""
    config = config or SimConfig()
    gt = make_ground_truth(kind, config)
    imu = make_imu(gt, config)
    gps = make_gps(gt, config, dropout_windows, degraded_windows)
    return SimulationResult(
        ground_truth=gt,
        imu=imu,
        gps=gps,
        config=config,
        dropout_windows=list(dropout_windows or []),
        degraded_windows=list(degraded_windows or []),
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a synthetic GPS/IMU trajectory and print a summary."
    )
    parser.add_argument("--kind", choices=["line", "turns"], default="turns")
    parser.add_argument("--duration", type=float, default=120.0)
    parser.add_argument(
        "--dropout", nargs=2, type=float, metavar=("START", "END"), default=[40.0, 55.0]
    )
    parser.add_argument(
        "--degraded", nargs=2, type=float, metavar=("START", "END"), default=[80.0, 92.0]
    )
    args = parser.parse_args()

    cfg = replace(SimConfig(), duration_s=args.duration)
    result = simulate(
        kind=args.kind,
        config=cfg,
        dropout_windows=[tuple(args.dropout)],
        degraded_windows=[tuple(args.degraded)],
    )
    gt, imu, gps = result.ground_truth, result.imu, result.gps
    print(f"kind={args.kind}  duration={args.duration}s")
    print(f"ground truth: {len(gt.t)} samples @ {1/gt.dt:.0f} Hz")
    print(f"IMU: {len(imu.t)} samples, accel |mean bias| "
          f"{np.abs(imu.accel_bias_true).mean():.4f} m/s^2")
    print(f"GPS: {len(gps.t)} fixes, {int((~gps.available).sum())} withheld, "
          f"{int(gps.degraded.sum())} degraded")
    span = np.hypot(np.ptp(gt.x), np.ptp(gt.y))
    print(f"path bounding span: {span:.0f} m")
