"""
NMEA Logger — logs raw NMEA sentences from the NEO-6M (via ESP32-S3 passthrough)
to a timestamped file, with a live per-line timestamp for later temporal-feature work.

Usage:
    python nmea_logger.py --port COM6 --baud 115200 --label clean

    --port   Serial port your ESP32-S3 shows up as (check Arduino IDE Tools > Port)
    --baud   Must match Serial.begin() in the sketch (115200 in your current sketch)
    --label  Optional tag added to the filename, e.g. "clean" or "degraded"

Stop logging any time with Ctrl+C — the file is flushed after every line, so
nothing is lost if you kill it abruptly.

Output: logs/nmea_<label>_<YYYYMMDD_HHMMSS>.log
Each line: <iso_timestamp> <raw NMEA sentence>
"""

import argparse
import datetime as dt
import os
import sys

try:
    import serial
except ImportError:
    sys.exit(
        "pyserial not installed. Run: pip install pyserial --break-system-packages"
    )


def main():
    parser = argparse.ArgumentParser(description="Log raw NMEA sentences to a timestamped file.")
    parser.add_argument("--port", required=True, help="Serial port, e.g. COM6 or /dev/ttyUSB0")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--label", default="log", help="Label for the output filename (e.g. clean, degraded)")
    parser.add_argument("--outdir", default="logs", help="Output directory (default: logs)")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(args.outdir, f"nmea_{args.label}_{timestamp}.log")

    print(f"Opening {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        sys.exit(f"Could not open serial port: {e}")

    print(f"Logging to {filename}")
    print("Press Ctrl+C to stop.\n")

    line_count = 0
    fix_count = 0

    try:
        with open(filename, "w", encoding="utf-8") as f:
            while True:
                raw = ser.readline()
                if not raw:
                    continue  # timeout with no data, keep waiting

                try:
                    line = raw.decode("ascii", errors="replace").strip()
                except Exception:
                    continue

                if not line.startswith("$"):
                    continue  # skip boot messages / non-NMEA noise

                now = dt.datetime.now().isoformat()
                f.write(f"{now} {line}\n")
                f.flush()

                line_count += 1
                print(line)

                # quick live indicator of fix status from GGA sentences
                if line.startswith("$GPGGA") or line.startswith("$GNGGA"):
                    fields = line.split(",")
                    if len(fields) > 6 and fields[6] not in ("", "0"):
                        fix_count += 1

    except KeyboardInterrupt:
        print(f"\nStopped. Logged {line_count} sentences ({fix_count} with a valid fix) to {filename}")
    finally:
        ser.close()


if __name__ == "__main__":
    main()