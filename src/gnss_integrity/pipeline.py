"""Integration point wiring `detect` output into the `fuse` filter.

This module will eventually own the end-to-end flow: raw GPS/IMU
observations go through `detect` to produce an anomaly/spoofing signal,
which is then fed into the `fuse` EKF as a measurement-trust weight,
producing a position estimate plus an integrity flag. No logic yet.
"""
