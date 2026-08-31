# gnss-integrity

GPS receivers report a position even when that position is wrong — multipath, jamming, and spoofing all degrade signal quality silently, with no warning built in. `gnss_integrity` watches raw signal-quality features in real time and uses ML, trained only on clean/nominal conditions, to flag when the current signal doesn't match "normal." That flag feeds into a GPS+IMU sensor fusion filter (EKF), so the filter automatically trusts GPS less exactly when it shouldn't. The result isn't just a position — it's a position plus an integrity flag saying how much to trust it right now.

## Architecture

```
  recorded NMEA logs          TEXBAT channel.mat / navsol.mat
         |                                   |
         v                                   v
  data/nmea_loader.py                data/texbat_loader.py
         |                                   |
         +----------------+------------------+
                          v
             per-epoch signal-quality features
        (mean/max/var SNR, tracked sats, dSNR, rolling var)
                          |
                          v
                  detect/  (Phase 2, done)
        isolation_forest.py   |   autoencoder.py
                          |
                          v
              anomaly flag  (0 = clean, 1 = anomalous)
                          |
                          x   <-- Phase 4: flag -> EKF trust_weight
                          |       NOT WIRED YET (pipeline.py is a stub)
                          v
   GPS fix  ------>  fuse/ekf.py  (Phase 3, done)  <------  IMU (accel + gyro)
                          |         5-state EKF; GPS update takes a
                          |         trust_weight that scales R
                          v
              fused position  ( + integrity flag, once Phase 4 lands )
```

`fuse/ekf.py` already exposes the `trust_weight` hook on `update_gps()`, and
`fuse/validate.py` exercises it with a **hand-set constant** during an
injected "degraded GPS" window. Phase 4 is the missing link: replacing that
constant with the detector's real per-epoch output so the filter
down-weights GPS automatically when the signal looks wrong. Until then,
`detect/` and `fuse/` are validated independently and do not talk to each
other.

## Installation

```bash
pip install -e .
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

## Status

Under active development.

- **Phase 0 — scaffold.** Done.
- **Phase 1 — data loaders.** Done. `data/nmea_loader.py` and
  `data/texbat_loader.py` parse recorded NMEA and TEXBAT
  (`channel.mat` / `navsol.mat`) into a common per-epoch feature schema.
- **Phase 2 — detectors.** Done. `detect/isolation_forest.py` and
  `detect/autoencoder.py`, both trained on clean-only features and
  evaluated against labelled TEXBAT spoofing scenarios (ds2, ds3, ds7).
  Each emits a 0/1 anomaly flag per epoch.
- **Phase 3 — GPS/IMU fusion.** Done. `fuse/ekf.py` is a 5-state EKF that
  dead-reckons on the IMU between GPS fixes; `fuse/trajectory_sim.py`
  generates synthetic ground-truth + IMU + GPS with injectable GPS
  dropout / degraded windows; `fuse/validate.py` shows the fused estimate
  tracking through a GPS outage roughly an order of magnitude closer to
  truth than a raw-GPS baseline. Validated on synthetic trajectories
  only — there is no real paired GPS+IMU dataset in this project.
- **Phase 4 — detector → EKF integration.** *In progress / next.* This is
  the project's headline contribution and it is **not done yet**. The EKF
  currently has no knowledge of the detector; `fuse/validate.py`
  down-weights bad GPS with a hardcoded constant, not the detector's
  output. `pipeline.py` is still a stub. Until Phase 4 lands, "integrity-
  aware fusion" is not yet demonstrated end-to-end.
- **Phase 5 — CI.** Not started.
