# gnss-integrity

GPS receivers report a position even when that position is wrong — multipath, jamming, and spoofing all degrade signal quality silently, with no built-in warning in the receiver's output. `gnss_integrity` watches raw signal-quality features in real time and uses ML, trained only on clean/nominal conditions, to flag when the current signal doesn't match "normal." That flag feeds into a GPS+IMU sensor fusion filter (EKF), so the filter automatically trusts GPS less exactly when it shouldn't. The result isn't just a position — it's a position plus an integrity flag saying how much to trust it right now.

## Architecture

```
                (placeholder — fill in)

  NMEA / TEXBAT / Yunnan  ->  [ data ]
                                  |
                                  v
                            [ detect ]  --(anomaly / integrity signal)--+
                                  |                                     |
                                  v                                     v
                            [  fuse  ]  <----------------------- [ pipeline ]
                                  |
                                  v
                    position estimate + integrity flag
```

When the detector flags a suspect signal, the fusion filter down-weights GPS and leans on IMU dead-reckoning until the signal is trusted again.

## Installation

```bash
pip install -e .
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

## Status

Under active development. Phase 0: repo scaffold only — no data loading, detection, or fusion logic implemented yet.
