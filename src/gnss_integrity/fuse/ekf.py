"""
Extended Kalman Filter fusing GPS position with IMU dead reckoning.

Between GPS fixes the filter propagates the state forward using the IMU
(body-frame acceleration + yaw rate) -- this is the dead-reckoning
"predict" step. When a GPS position fix arrives it corrects the state
with a standard EKF "update". The point of fusion is that the IMU keeps
the estimate moving sensibly through GPS gaps/dropouts, while GPS keeps
the IMU's integration drift bounded the rest of the time.

State vector (5)
----------------
    [ x, y, vx, vy, theta ]
        x, y     : world-frame position, m  (x = east, y = north)
        vx, vy   : world-frame velocity, m/s
        theta    : heading, rad, unwrapped (CCW from +x)

Position + velocity are the quantities downstream code and the plots
care about. Heading is carried as a state (rather than treated as a
pure input) because the IMU acceleration is measured in the *body*
frame: rotating it into the world frame needs theta, so an error in
theta feeds directly into the position estimate and the filter should
track its own heading uncertainty. GPS here measures position only, so
theta is observed only indirectly (through the motion it implies).

Motion model (nonlinear -> EKF, not plain KF)
---------------------------------------------
Given a step dt and an ImuReading (a_fwd, a_lat, yaw_rate):

    a_wx = a_fwd*cos(theta) - a_lat*sin(theta)
    a_wy = a_fwd*sin(theta) + a_lat*cos(theta)

    x'     = x  + vx*dt + 0.5*a_wx*dt^2
    y'     = y  + vy*dt + 0.5*a_wy*dt^2
    vx'    = vx + a_wx*dt
    vy'    = vy + a_wy*dt
    theta' = theta + yaw_rate*dt

The world acceleration depends on theta through cos/sin, so the state
transition is nonlinear and we linearize it (Jacobian F) for the
covariance propagation -- a textbook EKF, no exotic tricks.

Measurement model
-----------------
GPS gives (x, y) directly, so H is constant and linear:

    z = H x + v,   H = [[1,0,0,0,0],
                        [0,1,0,0,0]],   v ~ N(0, R)

Trust weighting (Phase 4 hook -- not wired to anything yet)
----------------------------------------------------------
``update_gps`` accepts ``trust_weight`` (default 1.0). The effective
measurement covariance is ``R_effective = R / trust_weight``: a weight
below 1 inflates R and makes the filter trust that fix less; a weight
above 1 does the opposite. Nothing calls this with a non-default value
today. Phase 4 will feed the anomaly detector's per-epoch confidence in
here to down-weight GPS during suspected spoofing. The "naive" baseline
filter for later comparison is simply this same class run with
``trust_weight`` left at 1.0 for every fix -- no separate class needed.

What this module deliberately does NOT do
-----------------------------------------
* No detector integration (Phase 4).
* No bias states for the IMU. Estimating accel/gyro bias online is a
  reasonable extension, but it is out of scope here and would make the
  filter harder to read; process noise absorbs the bias for now.
* No smoothing / no fixed-lag -- forward filter only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_STATE_DIM = 5


@dataclass
class ImuReading:
    """One IMU sample, body frame (x = forward/along-heading, y = left)."""
    accel_x: float   # m/s^2, longitudinal
    accel_y: float   # m/s^2, lateral
    yaw_rate: float  # rad/s


class ExtendedKalmanFilter:
    """5-state EKF: see module docstring for the model."""

    STATE_DIM = _STATE_DIM
    # Guard so a caller passing trust_weight=0 gets "ignore this fix"
    # instead of a divide-by-zero / singular covariance.
    _MIN_TRUST = 1.0e-6

    def __init__(
        self,
        initial_state: np.ndarray | list[float],
        initial_covariance: np.ndarray | None = None,
        accel_noise_std: float = 0.2,
        yaw_rate_noise_std: float = 2.0e-2,
        gps_position_noise_std: float = 3.0,
    ) -> None:
        """
        accel_noise_std / yaw_rate_noise_std drive the process noise Q.
        They are set well above the simulator's true sensor noise on
        purpose: Q also has to absorb un-modelled effects (IMU bias
        drift, the constant-acceleration approximation over a step, the
        model mismatch during turns), and an over-tight Q makes an EKF
        lag the truth and eventually diverge. These defaults were tuned
        on the synthetic scenarios in validate.py -- loose enough to
        track the turns, tight enough that the IMU still smooths GPS
        noise and carries the estimate through a dropout.

        gps_position_noise_std sets the default measurement noise R
        (per axis, isotropic). Callers may override R per update.
        """
        self.x = np.asarray(initial_state, dtype=float).reshape(_STATE_DIM).copy()

        if initial_covariance is None:
            # Modest starting uncertainty: we usually initialize near a
            # known point, but leave enough slack that the first GPS
            # fixes can pull the state in.
            self.P = np.diag([4.0, 4.0, 1.0, 1.0, 0.05])
        else:
            self.P = np.asarray(initial_covariance, dtype=float).reshape(
                _STATE_DIM, _STATE_DIM
            ).copy()

        self.accel_noise_std = float(accel_noise_std)
        self.yaw_rate_noise_std = float(yaw_rate_noise_std)
        self.R_default = np.eye(2) * float(gps_position_noise_std) ** 2

        self._H = np.zeros((2, _STATE_DIM))
        self._H[0, 0] = 1.0
        self._H[1, 1] = 1.0

    # --- prediction -----------------------------------------------------

    def _process_noise(self, dt: float) -> np.ndarray:
        """
        Discrete white-noise-acceleration process noise.

        Per world axis, an unknown acceleration of variance sigma_a^2
        acting over dt contributes
            [[dt^4/4, dt^3/2],
             [dt^3/2, dt^2  ]] * sigma_a^2
        to the (position, velocity) block. Heading gets a simple
        (sigma_omega * dt)^2 term from yaw-rate noise.
        """
        sa2 = self.accel_noise_std ** 2
        q = np.array(
            [[0.25 * dt ** 4, 0.5 * dt ** 3],
             [0.5 * dt ** 3, dt ** 2]]
        ) * sa2

        Q = np.zeros((_STATE_DIM, _STATE_DIM))
        Q[np.ix_([0, 2], [0, 2])] = q          # x  / vx
        Q[np.ix_([1, 3], [1, 3])] = q          # y  / vy
        Q[4, 4] = (self.yaw_rate_noise_std * dt) ** 2
        return Q

    def predict(self, dt: float, imu: ImuReading) -> None:
        """Propagate the state and covariance forward by dt using the IMU."""
        x, y, vx, vy, theta = self.x
        c, s = math.cos(theta), math.sin(theta)

        # Body-frame acceleration rotated into the world frame.
        a_wx = imu.accel_x * c - imu.accel_y * s
        a_wy = imu.accel_x * s + imu.accel_y * c

        # Nonlinear state transition (constant accel / yaw rate over the step).
        self.x = np.array([
            x + vx * dt + 0.5 * a_wx * dt * dt,
            y + vy * dt + 0.5 * a_wy * dt * dt,
            vx + a_wx * dt,
            vy + a_wy * dt,
            theta + imu.yaw_rate * dt,
        ])

        # Jacobian F = d f / d state. The only nonlinearity is theta's
        # effect on the rotated acceleration:
        #   d a_wx / d theta = -(a_x*sin + a_y*cos) = -a_wy
        #   d a_wy / d theta =  (a_x*cos - a_y*sin) =  a_wx
        d_ax = -a_wy
        d_ay = a_wx
        F = np.array([
            [1.0, 0.0, dt, 0.0, 0.5 * dt * dt * d_ax],
            [0.0, 1.0, 0.0, dt, 0.5 * dt * dt * d_ay],
            [0.0, 0.0, 1.0, 0.0, dt * d_ax],
            [0.0, 0.0, 0.0, 1.0, dt * d_ay],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ])

        self.P = F @ self.P @ F.T + self._process_noise(dt)
        # Keep P numerically symmetric.
        self.P = 0.5 * (self.P + self.P.T)

    # --- update -------------------------------------------------------

    def update_gps(
        self,
        position: np.ndarray | list[float],
        trust_weight: float = 1.0,
        R: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Correct the state with a GPS position fix [x, y].

        trust_weight : scales the measurement covariance as
            R_effective = R / trust_weight.
            1.0 -> nominal (the only value used before Phase 4).
            <1  -> trust this fix less (e.g. detector flags spoofing).
            >1  -> trust this fix more.
        R : optional explicit 2x2 measurement covariance; defaults to the
            filter's isotropic R_default.

        Returns the innovation (measurement - predicted measurement),
        which is useful for logging / later gating.
        """
        z = np.asarray(position, dtype=float).reshape(2)
        base_R = self.R_default if R is None else np.asarray(R, dtype=float).reshape(2, 2)
        tw = max(float(trust_weight), self._MIN_TRUST)
        R_eff = base_R / tw

        H = self._H
        innovation = z - H @ self.x
        S = H @ self.P @ H.T + R_eff
        K = self.P @ H.T @ np.linalg.inv(S)

        self.x = self.x + K @ innovation
        self.P = (np.eye(_STATE_DIM) - K @ H) @ self.P
        self.P = 0.5 * (self.P + self.P.T)
        return innovation

    # --- read-out -----------------------------------------------------

    @property
    def state(self) -> np.ndarray:
        return self.x.copy()

    @property
    def position(self) -> np.ndarray:
        return self.x[0:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    @property
    def heading(self) -> float:
        return float(self.x[4])

    @property
    def covariance(self) -> np.ndarray:
        return self.P.copy()

    @property
    def position_covariance(self) -> np.ndarray:
        return self.P[0:2, 0:2].copy()
