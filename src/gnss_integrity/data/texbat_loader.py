"""
TEXBAT loader.

Parses the UT GRID/pprx receiver's processed .mat outputs (channel.mat,
navsol.mat) -- available under texbat/processed/<scenario>/ -- into the
same per-epoch feature schema used by nmea_loader.py, so TEXBAT scenarios
can feed the same downstream pipeline as your own NMEA logs.

Format reference (UT Radionavigation Lab, satNavCourse/logFileDocumentation):

channel.mat -> channel.log columns (after transpose, per their README):
    1  RRT week number
    2  RRT seconds of week
    3  ORT week number (9999 = not yet valid)
    4  ORT whole seconds of week
    5  ORT fractional second
    6  Apparent Doppler frequency (Hz)
    7  Beat carrier phase (cycles)
    8  Pseudorange (m)
    9  Carrier-to-noise ratio, C/N0 (dB-Hz)  <- SNR equivalent
    10 Validity flag (1 = pseudorange/carrier phase valid)
    11 Error indicator (bit 0: phase anomaly, bit 1: spoofing detected,
       bit 2: possible half-cycle offset)
    12 Channel status (0=NULL .. 6=DATA_LOCK)
    13 Signal type (encoded)
    14 Transmitter ID (TXID / PRN)

navsol.mat -> navsol.log columns (after transpose):
    1  ORT week number
    2  ORT whole seconds of week
    3  ORT fractional second
    4-6   X,Y,Z ECEF position (m)
    7  Receiver clock error (m, equivalent)
    8-10  Xdot,Ydot,Zdot ECEF velocity (m/s)
    11 Receiver clock error rate (m/s)
    12 Fix flag (0=NO_FIX,1=PRELIMINARY,2=STANDARD,3=PRECISE)
    13 NISratio (may be absent in some exports)

Key differences from the NMEA schema this loader must bridge:
  - No HDOP/PDOP is reported directly. We approximate a DOP-like quality
    proxy from the navsol fix flag (higher flag = better geometry/fix),
    since true DOP would require the full satellite geometry matrix,
    which isn't in these files.

Ground truth for spoofing labels:
  channel.mat's error-indicator bit 1 ("spoofing detected") is exposed
  as `spoof_flag_raw`, but empirically it reads 0 across every TEXBAT
  scenario we've tested -- including ds2, a scenario with a well-
  documented, easily-visible attack. This strongly suggests the flag
  simply isn't populated in this batch of processed files, so it is
  NOT a reliable ground-truth signal on its own.

  Instead, this loader also emits `known_spoof_window`, computed from
  the *documented* attack start times published by UT Austin / the
  TEXBAT literature (see SCENARIO_ATTACK_START_S below), which is
  time-since-recording-start in seconds. Any epoch at or after that
  time is labeled 1. This is a coarse but trustworthy label: it won't
  capture the exact takeover transition, but it correctly separates
  "before attack" from "during/after attack" for training and
  evaluation purposes.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import scipy.io as sio

# GRID channel status codes worth trusting for SNR stats: freq lock or better
MIN_TRUSTED_STATUS = 4  # STATUS_FREQ_LOCK

# navsol fix-flag -> rough DOP-quality proxy (lower "proxy_dop" = better,
# mirroring how real DOP works, so it plugs into the same downstream code
# that expects "lower is better" for hdop/pdop-like columns)
FIX_FLAG_TO_DOP_PROXY = {
    0: float("nan"),  # SOL_NO_FIX -- no usable solution
    1: 5.0,            # SOL_PRELIMINARY_FIX -- rough analogue of "fair"
    2: 1.5,             # SOL_STANDARD_FIX -- rough analogue of "good"
    3: 0.8,             # SOL_PRECISE_FIX -- rough analogue of "excellent"
}

# Documented attack start times (seconds from recording start), sourced
# from TEXBAT literature / UT Austin scenario descriptions. Epochs at or
# after this time are labeled as "known spoofed" via `known_spoof_window`.
# Scenario names are matched case-insensitively against a substring of the
# input file path (see _infer_scenario_name below).
SCENARIO_ATTACK_START_S: dict[str, float | None] = {
    "ds1": 100.0,
    "ds2": 100.0,
    "ds3": 100.0,
    "ds4": 110.0,
    "ds5": 100.0,
    "ds6": 105.0,
    "ds7": 110.0,
    "ds8": 110.0,
    "cleanstatic": None,  # never spoofed; every epoch labeled 0
    "cleandynamic": None,
}


def _infer_scenario_name(channel_path: str | Path) -> str | None:
    """
    Guess the TEXBAT scenario name from the channel.mat file's path, by
    matching known scenario folder names (ds1..ds8, cleanStatic,
    cleanDynamic) as a case-insensitive substring. Returns None if no
    match is found, in which case the caller should treat the attack
    timing as unknown rather than guessing.
    """
    path_str = str(channel_path).lower()
    # Check longer names first so "cleanstatic" doesn't get missed by a
    # shorter accidental match, and ds1 doesn't shadow ds10-style names
    # (not that TEXBAT has any, but this keeps the matching robust).
    for name in sorted(SCENARIO_ATTACK_START_S, key=len, reverse=True):
        if name in path_str:
            return name
    return None

@dataclass
class FeatureVector:
    """Same schema as nmea_loader.FeatureVector, plus TEXBAT-specific extras."""
    fix_time: float  # RRT seconds of week, used as the epoch key here
    mean_snr: float
    max_snr: float
    var_snr: float
    hdop: float       # proxy, see FIX_FLAG_TO_DOP_PROXY
    pdop: float       # same proxy value; no independent PDOP available
    tracked_satellites: int
    fix_quality: int  # navsol fix flag (0-3), NOT the NMEA 0/1 convention
    delta_mean_snr: float
    rolling_var_snr: float
    spoof_flag_raw: int       # GRID's own error-bit flag -- UNRELIABLE,
                               # observed to be 0 across every scenario
                               # tested so far; kept for transparency only
    known_spoof_window: int   # 1 if epoch is at/after the scenario's
                               # documented attack start time, else 0.
                               # -1 if the scenario couldn't be identified
                               # from the file path (unknown ground truth).


def _load_channel(path: str | Path) -> np.ndarray:
    """Load and transpose channel.mat per the UT Radionavigation Lab README."""
    mat = sio.loadmat(path)
    return mat["channel"].T  # (n_measurements, 14)


def _load_navsol(path: str | Path) -> np.ndarray:
    """Load and transpose navsol.mat per the UT Radionavigation Lab README."""
    mat = sio.loadmat(path)
    return mat["navsol"].T  # (n_epochs, 12 or 13)


def _group_channel_by_epoch(channel: np.ndarray, epoch_period_s: float = 0.2) -> dict[float, np.ndarray]:
    """
    Group channel.mat rows by RRT seconds-of-week, bucketed to the
    receiver's actual measurement period (GRID logs at 5 Hz / 0.2s
    intervals in these TEXBAT exports -- confirmed empirically, not
    assumed). Naively rounding to whole seconds would lump 5 real
    epochs together and wildly overstate tracked-satellite count.

    RRT is used instead of ORT because ORT can be marked invalid
    (week=9999) early in a session.
    """
    bucket = np.round(channel[:, 1] / epoch_period_s) * epoch_period_s
    bucket = np.round(bucket, 3)  # kill float dust so identical buckets match
    epochs: dict[float, list] = {}
    for row, t in zip(channel, bucket):
        epochs.setdefault(t, []).append(row)
    return {t: np.array(rows) for t, rows in epochs.items()}


def _nearest_navsol_flag(navsol: np.ndarray, ort_week: float, ort_sec: float) -> int:
    """
    Find the navsol row closest in time to a given ORT week/whole-second and
    return its fix flag. Falls back to SOL_NO_FIX (0) if navsol has no rows
    for that week (e.g. ORT wasn't valid yet for this channel epoch).
    """
    if navsol.shape[0] == 0:
        return 0
    same_week = navsol[navsol[:, 0] == ort_week]
    if same_week.shape[0] == 0:
        return 0
    idx = np.argmin(np.abs(same_week[:, 1] - ort_sec))
    return int(same_week[idx, 11])


def load_texbat_scenario(
    channel_path: str | Path,
    navsol_path: str | Path,
    rolling_window: int = 10,
) -> list[FeatureVector]:
    """
    Parse a TEXBAT scenario's channel.mat + navsol.mat into per-epoch
    feature vectors matching the project's standard schema.
    """
    channel = _load_channel(channel_path)
    navsol = _load_navsol(navsol_path)

    epochs = _group_channel_by_epoch(channel)

    scenario = _infer_scenario_name(channel_path)
    attack_start = SCENARIO_ATTACK_START_S.get(scenario) if scenario else "unknown"

    # RRT starts near-zero at recording start (confirmed: first timestamps
    # in cleanStatic/ds2/ds3/ds7 all begin around t=0.2-27s depending on
    # when tracking first locked), so fix_time is directly comparable to
    # the documented attack-start offsets without further adjustment.
    first_epoch_t = min(epochs.keys()) if epochs else 0.0

    features: list[FeatureVector] = []
    prev_mean_snr: float | None = None
    recent_mean_snrs: list[float] = []

    for t in sorted(epochs.keys()):
        rows = epochs[t]

        # Trust only entries that are valid and at least frequency-locked;
        # this filters out acquisition-stage noise from the SNR stats.
        valid_mask = (rows[:, 9] == 1) & (rows[:, 11] >= MIN_TRUSTED_STATUS)
        trusted = rows[valid_mask]

        if trusted.shape[0] == 0:
            continue

        snrs = trusted[:, 8]
        m = float(np.mean(snrs))
        mx = float(np.max(snrs))
        v = float(np.var(snrs))

        tracked = int(trusted.shape[0])

        # GRID's own spoofing bit, checked across ALL rows this epoch.
        # Kept for transparency but NOT trusted as ground truth -- see
        # module docstring for why.
        err = rows[:, 10].astype(int)
        spoof_flag_raw = int(np.any((err >> 1) & 1))

        # Time-based ground truth from documented attack-start times.
        if attack_start == "unknown":
            known_spoof = -1
        elif attack_start is None:
            known_spoof = 0  # cleanStatic/cleanDynamic -- never spoofed
        else:
            known_spoof = int(t >= attack_start)

        # Look up nearest navsol fix flag using this epoch's ORT, if valid.
        ort_week = rows[0, 2]
        ort_sec = rows[0, 3]
        if ort_week != 9999:
            fix_flag = _nearest_navsol_flag(navsol, ort_week, ort_sec)
        else:
            fix_flag = 0

        dop_proxy = FIX_FLAG_TO_DOP_PROXY.get(fix_flag, float("nan"))

        delta = 0.0 if prev_mean_snr is None else m - prev_mean_snr
        recent_mean_snrs.append(m)
        if len(recent_mean_snrs) > rolling_window:
            recent_mean_snrs.pop(0)
        rolling_var = (
            float(np.var(recent_mean_snrs)) if len(recent_mean_snrs) > 1 else 0.0
        )

        features.append(
            FeatureVector(
                fix_time=float(t),
                mean_snr=round(m, 3),
                max_snr=round(mx, 3),
                var_snr=round(v, 3),
                hdop=dop_proxy,
                pdop=dop_proxy,
                tracked_satellites=tracked,
                fix_quality=fix_flag,
                delta_mean_snr=round(delta, 3),
                rolling_var_snr=round(rolling_var, 3),
                spoof_flag_raw=spoof_flag_raw,
                known_spoof_window=known_spoof,
            )
        )
        prev_mean_snr = m

    return features


def write_csv(features: list[FeatureVector], out_path: str | Path) -> None:
    if not features:
        raise ValueError("No feature vectors to write.")
    fieldnames = list(asdict(features[0]).keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fv in features:
            writer.writerow(asdict(fv))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse a TEXBAT scenario (channel.mat + navsol.mat) into feature vectors."
    )
    parser.add_argument("channel_mat", help="Path to channel.mat")
    parser.add_argument("navsol_mat", help="Path to navsol.mat")
    parser.add_argument("-o", "--output", default="texbat_features.csv")
    parser.add_argument("--rolling-window", type=int, default=10)
    args = parser.parse_args()

    feats = load_texbat_scenario(
        args.channel_mat, args.navsol_mat, rolling_window=args.rolling_window
    )
    write_csv(feats, args.output)

    print(f"Parsed {len(feats)} epochs")
    print(f"Wrote features to {args.output}")

    scenario = _infer_scenario_name(args.channel_mat)
    n_raw_flagged = sum(f.spoof_flag_raw for f in feats)
    print(f"GRID's own spoofing bit set (unreliable, see docstring): {n_raw_flagged} / {len(feats)}")

    if scenario is None:
        print("Scenario name not recognized from path -- known_spoof_window is -1 (unknown) for all epochs.")
        print("Rename your folder to include ds1..ds8 / cleanStatic / cleanDynamic for automatic labeling.")
    else:
        attack_start = SCENARIO_ATTACK_START_S.get(scenario)
        n_known_spoofed = sum(1 for f in feats if f.known_spoof_window == 1)
        if attack_start is None:
            print(f"Scenario detected: {scenario} (no attack -- clean baseline)")
        else:
            print(f"Scenario detected: {scenario} (documented attack start: {attack_start}s)")
            print(f"Epochs labeled known_spoof_window=1: {n_known_spoofed} / {len(feats)}")

    sats = [f.tracked_satellites for f in feats]
    print(f"Tracked satellites avg: {sum(sats)/len(sats):.1f}")