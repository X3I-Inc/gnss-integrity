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
                          v   pipeline.py  (Phase 4, wired -- result mixed)
                  flag -> GPS trust_weight
                          |
                          v
   GPS fix  ------>  fuse/ekf.py  (Phase 3, done)  <------  IMU (accel + gyro)
                          |         5-state EKF; GPS update takes a
                          |         trust_weight that scales R
                          v
              fused position  (+ integrity flag)
```

`pipeline.py` connects the two: the detector's per-epoch flag is mapped to a
`trust_weight` and fed to `ekf.update_gps()`, replacing the hand-set constant
`fuse/validate.py` used. With a *perfect* (oracle) detector this beats naive
"always trust GPS" fusion through a spoofing window; with the current Phase 2
IsolationForest it does **not** yet — detection latency and false positives
make it scenario-dependent and, on the default ds2 case, worse than naive.
See the Phase 4 status note below.

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
- **Phase 4 — detector → EKF integration.** *Wired; result is mixed, with
  a now-characterised caveat.* `pipeline.py` maps the detector's per-epoch
  flag to an EKF GPS `trust_weight` (hybrid setup: real detector + real
  TEXBAT features, synthetic trajectory with a co-timed synthetic GPS
  fault, exact ground truth). It runs naive vs integrity-aware fusion and
  produces the comparison plot (`python -m gnss_integrity.pipeline -o out.png`).
  Findings:
  - With an *oracle* (perfect) detector, integrity-aware fusion beats
    naive through the attack (~1.3–1.7×) — the integration is sound.
  - With the **current Phase 2 IsolationForest**, on the default
    *sustained* attack (spoofed to end of run, matching TEXBAT), it is
    **~23 % worse than naive** on ds2: ~5 s detection latency lets biased
    fixes corrupt the filter's velocity before rejection starts, then
    dead reckoning drifts past the spoof bias; ~33 % pre-attack false
    positives add noise.
  - On a **bounded attack with a GPS-recovery period**
    (`--attack-duration 25 --recovery 45`), that mostly reverses: the EKF
    re-anchors once trust is restored (recovery-window RMSE beats naive by
    ~1.5×) and overall RMSE returns to parity (ds2) or better (ds7). So
    the real-world caveat is specific: **sustained attacks with no GPS
    recovery period are a known hard case**, not a general failure of the
    approach. The residual in-attack shortfall (~10 %, from latency +
    false positives) is smaller and is a detector-quality problem, not a
    wiring one.
- **Phase 5 — CI.** Not started. Test suite (`pytest tests/`, 23 tests)
  is CI-ready — `pip install -e ".[dev]"` is the only setup.
