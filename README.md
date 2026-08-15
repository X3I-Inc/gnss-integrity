# gnss-integrity

`gnss_integrity` is an open-source Python toolkit that fuses GPS and IMU data into a clean position estimate via an Extended Kalman Filter, and layers an anomaly/spoofing detector on top — so the output isn't just a position, it's a position plus a confidence/integrity flag telling you how much to trust it right now.

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
