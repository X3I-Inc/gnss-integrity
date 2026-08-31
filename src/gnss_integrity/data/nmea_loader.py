"""
NMEA log loader and feature extractor.

Parses raw, timestamped NMEA logs (as produced by nmea_logger.py) into
per-epoch feature vectors matching the architecture spec in
01-architecture.md:

    X = [ mean(SNR), max(SNR), var(SNR), HDOP, PDOP, tracked_satellites ]

Extended with temporal features:

    X_extended = X + [ delta_mean_SNR_from_prev_epoch,
                        rolling_var_SNR_over_N_epochs ]

Feature source:
    $GPGSV -> per-satellite SNR (C/N0, dB-Hz)
    $GPGSA -> HDOP, PDOP, tracked satellite count (from PRN list)
    $GPGGA -> tracked satellite count (redundant cross-check), fix quality

Epochs are grouped by the GGA/RMC timestamp (HHMMSS.ss field), since a
full round of GSV+GSA+GGA sentences shares that fix time even though
each sentence arrives on its own log line with its own wall-clock
timestamp.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, asdict
from pathlib import Path
from statistics import mean, pstdev
from typing import Iterator


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Epoch:
    """Raw per-epoch data accumulated from one round of NMEA sentences."""
    fix_time: str  # HHMMSS.ss as reported by the receiver
    snr_values: list[int]
    hdop: float | None = None
    pdop: float | None = None
    vdop: float | None = None
    tracked_satellites: int | None = None  # from GSA PRN list
    fix_quality: int | None = None  # from GGA (0 = no fix)


@dataclass
class FeatureVector:
    """One row of the output feature table."""
    fix_time: str
    mean_snr: float
    max_snr: float
    var_snr: float
    hdop: float
    pdop: float
    tracked_satellites: int
    fix_quality: int
    delta_mean_snr: float
    rolling_var_snr: float


# ---------------------------------------------------------------------------
# NMEA sentence parsing helpers
# ---------------------------------------------------------------------------

def _strip_checksum(sentence: str) -> str:
    """Drop the trailing *CS checksum, if present."""
    return sentence.split("*", 1)[0]


def _parse_gsv(fields: list[str]) -> list[int]:
    """
    Extract SNR values from a single $--GSV sentence.

    Format: $--GSV,total_msgs,msg_num,total_sats,
            [prn,elev,az,snr] x up to 4,checksum

    SNR field is blank when a satellite is visible but not tracked
    strongly enough to report C/N0 -- those are skipped, matching the
    architecture note that blank SNR means "visible but not usable".
    """
    snrs = []
    # fields[0] = talker+GSV, [1]=total_msgs, [2]=msg_num, [3]=total_sats
    sat_fields = fields[4:]
    for i in range(0, len(sat_fields) - 3, 4):
        snr_str = sat_fields[i + 3]
        if snr_str:
            try:
                snrs.append(int(snr_str))
            except ValueError:
                pass
    return snrs


def _parse_gsa(fields: list[str]) -> tuple[int, float, float, float]:
    """
    Extract tracked satellite count and DOP values from $--GSA.

    Format: $--GSA,mode1,mode2,prn1..prn12,pdop,hdop,vdop,checksum
    """
    # fields[0]=talker+GSA, [1]=mode1(M/A), [2]=mode2(1/2/3)
    prn_fields = fields[3:15]
    tracked = sum(1 for p in prn_fields if p)
    pdop = float(fields[15]) if len(fields) > 15 and fields[15] else float("nan")
    hdop = float(fields[16]) if len(fields) > 16 and fields[16] else float("nan")
    vdop = float(fields[17]) if len(fields) > 17 and fields[17] else float("nan")
    return tracked, hdop, pdop, vdop


def _parse_gga(fields: list[str]) -> tuple[str, int, int]:
    """
    Extract fix time, fix quality, and satellite count from $--GGA.

    Format: $--GGA,time,lat,NS,lon,EW,fixq,numsat,hdop,alt,...
    """
    fix_time = fields[1]
    fix_quality = int(fields[6]) if fields[6] else 0
    numsat = int(fields[7]) if fields[7] else 0
    return fix_time, fix_quality, numsat


# ---------------------------------------------------------------------------
# Log-level parsing
# ---------------------------------------------------------------------------

def parse_log(path: str | Path) -> Iterator[Epoch]:
    """
    Stream Epoch objects from a raw NMEA log file.

    Log lines look like:
        2026-08-15T17:31:04.229677 $GPRMC,153104.00,A,...
        2026-08-15T17:31:04.342895 $GPGGA,153104.00,...

    Epochs are keyed by the GGA fix-time field. GSV sentences accumulate
    SNR values into the *current* epoch (the most recently seen GGA time,
    or the log's running epoch if GGA hasn't appeared yet); GSA sentences
    set HDOP/PDOP/tracked count for the current epoch. A new epoch starts
    each time a fresh $--GGA sentence appears.
    """
    current: Epoch | None = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue

            # Strip the leading ISO timestamp our logger prepends, if present.
            # Lines look like "<timestamp> $SENTENCE" -- find the '$' start.
            dollar_idx = line.find("$")
            if dollar_idx == -1:
                continue
            sentence = _strip_checksum(line[dollar_idx:])
            fields = sentence.split(",")
            if not fields or len(fields[0]) < 3:
                continue

            msg_type = fields[0][-3:]  # e.g. GSV, GSA, GGA (talker-agnostic)

            if msg_type == "GGA":
                fix_time, fix_quality, numsat_gga = _parse_gga(fields)

                # A new GGA time means a new epoch. Flush the previous one.
                if current is not None and current.fix_time != fix_time:
                    yield current
                    current = None

                if current is None:
                    current = Epoch(fix_time=fix_time, snr_values=[])

                current.fix_quality = fix_quality
                # Keep GSA's tracked-satellite count as primary; fall back
                # to GGA's numsat if GSA hasn't been seen yet this epoch.
                if current.tracked_satellites is None:
                    current.tracked_satellites = numsat_gga

            elif msg_type == "GSA":
                if current is None:
                    # GSA arrived before the first GGA of a session; start
                    # a placeholder epoch keyed by "pending" so we don't
                    # drop data. It will be reconciled once GGA arrives.
                    current = Epoch(fix_time="pending", snr_values=[])
                tracked, hdop, pdop, vdop = _parse_gsa(fields)
                current.tracked_satellites = tracked
                current.hdop = hdop
                current.pdop = pdop
                current.vdop = vdop

            elif msg_type == "GSV":
                if current is None:
                    current = Epoch(fix_time="pending", snr_values=[])
                current.snr_values.extend(_parse_gsv(fields))

            # Other sentence types (RMC, VTG, GLL) are ignored for the
            # feature vector -- position/velocity live in the fusion
            # module's concern, not the integrity detector's.

    if current is not None:
        yield current


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def epochs_to_features(
    epochs: Iterator[Epoch],
    rolling_window: int = 10,
) -> list[FeatureVector]:
    """
    Convert a stream of Epoch objects into feature vectors, computing the
    temporal features (delta mean SNR, rolling variance of mean SNR) as
    we go.

    Epochs with no SNR readings at all (e.g. malformed/partial data) are
    skipped rather than emitting NaN-poisoned rows.
    """
    features: list[FeatureVector] = []
    prev_mean_snr: float | None = None
    recent_mean_snrs: list[float] = []

    for ep in epochs:
        if not ep.snr_values:
            continue

        m = mean(ep.snr_values)
        mx = max(ep.snr_values)
        v = pstdev(ep.snr_values) ** 2 if len(ep.snr_values) > 1 else 0.0

        delta = 0.0 if prev_mean_snr is None else m - prev_mean_snr

        recent_mean_snrs.append(m)
        if len(recent_mean_snrs) > rolling_window:
            recent_mean_snrs.pop(0)
        rolling_var = (
            pstdev(recent_mean_snrs) ** 2 if len(recent_mean_snrs) > 1 else 0.0
        )

        features.append(
            FeatureVector(
                fix_time=ep.fix_time,
                mean_snr=round(m, 3),
                max_snr=float(mx),
                var_snr=round(v, 3),
                hdop=ep.hdop if ep.hdop is not None else float("nan"),
                pdop=ep.pdop if ep.pdop is not None else float("nan"),
                tracked_satellites=ep.tracked_satellites or 0,
                fix_quality=ep.fix_quality or 0,
                delta_mean_snr=round(delta, 3),
                rolling_var_snr=round(rolling_var, 3),
            )
        )
        prev_mean_snr = m

    return features


def load_features(path: str | Path, rolling_window: int = 10) -> list[FeatureVector]:
    """Convenience wrapper: log file -> list of feature vectors."""
    return epochs_to_features(parse_log(path), rolling_window=rolling_window)


def write_csv(features: list[FeatureVector], out_path: str | Path) -> None:
    """Write feature vectors to CSV for inspection / downstream training."""
    if not features:
        raise ValueError("No feature vectors to write.")

    fieldnames = list(asdict(features[0]).keys())
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for fv in features:
            writer.writerow(asdict(fv))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Parse a raw NMEA log into per-epoch feature vectors."
    )
    parser.add_argument("log_path", help="Path to raw NMEA log file")
    parser.add_argument(
        "-o", "--output", default=None, help="Output CSV path (default: <log>_features.csv)"
    )
    parser.add_argument(
        "--rolling-window", type=int, default=10, help="Window size for rolling SNR variance"
    )
    args = parser.parse_args()

    log_path = Path(args.log_path)
    out_path = Path(args.output) if args.output else log_path.with_suffix("").with_name(
        log_path.stem + "_features.csv"
    )

    feats = load_features(log_path, rolling_window=args.rolling_window)
    write_csv(feats, out_path)

    print(f"Parsed {len(feats)} epochs from {log_path}")
    print(f"Wrote features to {out_path}")

    if feats:
        hdops = [f.hdop for f in feats if f.hdop == f.hdop]  # filter NaN
        sats = [f.tracked_satellites for f in feats]
        print(f"HDOP avg: {sum(hdops)/len(hdops):.2f}" if hdops else "HDOP: n/a")
        print(f"Satellite count avg: {sum(sats)/len(sats):.1f}")
