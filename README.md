# gnss-integrity

**A GPS integrity toolkit: detect spoofing/jamming from raw signal-quality
features, then fuse GPS with IMU so the filter trusts GPS less exactly when
it shouldn't.**

GPS receivers report a position even when that position is wrong — multipath,
jamming, and spoofing all degrade signal quality silently, with no warning
built in. `gnss_integrity` watches raw signal-quality features in real time
and uses ML, trained only on clean/nominal conditions, to flag when the
current signal doesn't match "normal." That flag feeds into a GPS+IMU sensor
fusion filter (EKF), so the filter automatically down-weights GPS during
suspect epochs. The result isn't just a position — it's a position plus an
integrity flag saying how much to trust it right now.

> **Project status: research prototype, under active development.** Phases 0–3
> (data loaders, detectors, EKF fusion) are complete and tested. Phase 4 (the
> detector → EKF integration) is wired and characterised but the end-to-end
> win is not yet demonstrated with a *real* detector — see
> [Status](#status) for the honest breakdown.

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
                          v   pipeline.py  (Phase 4, wired)
                  flag -> GPS trust_weight
                          |
                          v
   GPS fix  ------>  fuse/ekf.py  (Phase 3, done)  <------  IMU (accel + gyro)
                          |         5-state EKF; GPS update takes a
                          |         trust_weight that scales R
                          v
              fused position  (+ integrity flag)
```

`pipeline.py` connects the two halves: the detector's per-epoch flag is
mapped to a `trust_weight` and fed to `ekf.update_gps()`, replacing the
hand-set constant `fuse/validate.py` used. With a *perfect* (oracle) detector
this beats naive "always trust GPS" fusion through a spoofing window. With
the current Phase 2 IsolationForest the result is scenario-dependent: worse
than naive on a sustained attack with no GPS recovery, back to parity or
better once a recovery period exists. See [Status](#status).

## Installation

Python ≥ 3.10.

```bash
pip install -e .
```

For development (tests, linting):

```bash
pip install -e ".[dev]"
```

To record your own NMEA logs from a serial GPS receiver (see
[Usage](#usage)), install the `hardware` extra (adds `pyserial`):

```bash
pip install -e ".[hardware]"
```

Dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `matplotlib`,
`pynmea2`. The autoencoder detector additionally needs `tensorflow`
(not a declared dependency — install it separately if you want that path;
the Isolation Forest detector works without it).

## Usage

**Parse a TEXBAT scenario into per-epoch features:**

```bash
python -m gnss_integrity.data.texbat_loader \
    data/texbat/ds2/channel.mat data/texbat/ds2/navsol.mat \
    -o data/texbat/ds2/features.csv
```

**Train + evaluate a detector on TEXBAT spoofing scenarios:**

```bash
python -m gnss_integrity.detect.isolation_forest \
    --train data/texbat/cleanStatic/features.csv \
    --eval  data/texbat/ds2/features.csv data/texbat/ds7/features.csv
```

**Phase 3 — validate the EKF on a synthetic trajectory with a GPS dropout:**

```bash
python -m gnss_integrity.fuse.validate -o fuse_validation.png
```

Prints RMSE for fused-vs-raw-GPS and saves a ground-truth / raw-GPS / fused
plot with the dropout window shaded.

**Phase 4 — detector-driven integrity-aware fusion (naive vs integrity-aware
vs oracle):**

```bash
# sustained attack (spoofed to end of run, matches real TEXBAT)
python -m gnss_integrity.pipeline --scenario ds2 -o pipeline.png

# bounded attack + a clean-GPS recovery period
python -m gnss_integrity.pipeline --scenario ds2 \
    --attack-duration 25 --recovery 45 -o pipeline_bounded.png
```

**Record your own NMEA logs from a serial GPS receiver** (u-blox NEO-6M
via ESP32-S3 passthrough, or any receiver that streams NMEA over serial):

```bash
python scripts/nmea_logger.py --port COM6 --baud 115200 --label clean
```

Writes `logs/nmea_<label>_<timestamp>.log`; turn it into the same
per-epoch feature CSV as the TEXBAT path with:

```bash
python -m gnss_integrity.data.nmea_loader logs/nmea_clean_<timestamp>.log -o my_features.csv
```

Requires the `hardware` extra (`pyserial`).

## Repository layout

```
src/gnss_integrity/
  data/      nmea_loader.py, texbat_loader.py   -> per-epoch feature CSVs
  detect/    isolation_forest.py, autoencoder.py -> 0/1 anomaly flag per epoch
  fuse/      ekf.py            5-state GPS/IMU Extended Kalman Filter
             trajectory_sim.py synthetic ground-truth + IMU + GPS generator
             validate.py       Phase 3 CLI (EKF vs raw GPS through a dropout)
  pipeline.py                  Phase 4 CLI (detector flag -> EKF trust weight)
data/texbat/<scenario>/        derived features.csv (raw .mat not committed)
notebooks/                     01_clean_baseline_exploration.ipynb (clean-data EDA)
scripts/
  nmea_logger.py               live NMEA capture over serial (ESP32-S3 / NEO-6M)
  plot_reconstruction_error.py autoencoder reconstruction-error diagnostic plot
tests/                         pytest suite (test_fuse.py, test_pipeline.py)
```

## Data

Detector training/evaluation uses **TEXBAT** (Texas Spoofing Test Battery),
the public GPS spoofing dataset from the UT Austin Radionavigation Laboratory
— scenarios `cleanStatic`, `ds2`, `ds3`, `ds7`. Only the small derived
per-epoch `features.csv` files are committed; the multi-MB raw `channel.mat` /
`navsol.mat` recordings are not — download them from UT Austin and run
`texbat_loader.py` to regenerate.

There is no real paired GPS+IMU dataset in this project (TEXBAT is GPS-only),
so the EKF and the Phase 4 integration are validated on **synthetic
trajectories** with synthetic IMU and synthetic GPS faults, against exact
ground truth. See `fuse/trajectory_sim.py`.

To capture your own clean/nominal NMEA data from real hardware, see
`scripts/nmea_logger.py` (documented under [Usage](#usage)).

## Testing

```bash
pytest tests/ -v
```

23 tests, all synthetic (no network, no large data), typically ~8 s.

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

## License

MIT — see [LICENSE](LICENSE).
