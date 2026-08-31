"""Phase 4: wire the anomaly detector's output into the EKF trust weight.

This is the project's headline contribution -- the point where `detect/`
and `fuse/` stop being two modules sitting next to each other and become
one integrity-aware estimator. For each epoch the trained detector
produces an anomaly flag; that flag is turned into a ``trust_weight``
and handed to :meth:`ExtendedKalmanFilter.update_gps`, so GPS is
down-weighted automatically exactly when the signal looks wrong.

The bridge (why it looks the way it does)
-----------------------------------------
The detectors are trained on *real* GNSS signal-quality features
(``mean_snr`` etc.) from TEXBAT scenarios. The EKF is validated on a
*synthetic* trajectory, which has no such features. There is no real
paired GPS+IMU+signal-quality dataset in this project, so Phase 4 uses a
**hybrid** setup:

  * Real detector, real input: the chosen detector is trained on clean
    TEXBAT features (``cleanStatic``) and run on a real spoofing
    scenario's feature CSV (``ds2`` / ``ds7`` / ...). The resulting
    per-epoch flags are genuine detector output -- including its real
    detection latency and its real false positives / misses.
  * Synthetic trajectory + synthetic GPS fault: a synthetic ground-truth
    path (``fuse.trajectory_sim``) with a synthetic "degraded GPS"
    window injected over a span that lines up with the real scenario's
    documented attack. Exact ground truth for the RMSE claim.
  * The real detector flags, aligned onto the synthetic timeline, drive
    ``trust_weight``.

So the only synthetic thing about the attack is the *shape* of the GPS
corruption -- the same concession Phase 3 already makes -- while the
detector's behaviour is real.

Three filters are run for comparison:
  1. naive          -- every GPS fix trusted at weight 1.0 (the EKF is
                       blind to which fixes are bad).
  2. integrity (real)   -- trust weight driven by the real detector flags.
  3. integrity (oracle) -- trust weight driven by the ground-truth
                       ``known_spoof_window`` label (a hypothetical
                       perfect, zero-latency detector). This is the
                       ceiling the integration could reach, and isolates
                       "is the wiring/strategy sound?" from "is the
                       current detector good enough?".

flag -> trust_weight mapping
----------------------------
Deliberately simple for this first cut (see :func:`flag_to_trust_weight`):

    flagged (anomalous) epoch -> ``flagged_trust_weight`` (default 0.05,
        the same constant Phase 3's validate.py used by hand)
    unflagged epoch           -> 1.0 (nominal)

A smoother score -> weight map (e.g. a sigmoid on the autoencoder's
reconstruction MSE, which ``detect.autoencoder.score_dataframe`` already
returns) and detection-latency compensation are the obvious next
refinements. They are not needed to demonstrate the integration, and the
results below show exactly why they will matter.

What this module does NOT do
---------------------------
* It does not modify `detect/`, `data/`, or the internals of
  `fuse/trajectory_sim.py` / `fuse/ekf.py`.
* It does not claim a real end-to-end field result -- the trajectory is
  synthetic. It demonstrates the *wiring* and quantifies the benefit
  *given* a flag stream, for both a real and an ideal detector.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from dataclasses import replace as _replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Allow `python src/gnss_integrity/pipeline.py ...` as well as `-m`.
_SRC_DIR = Path(__file__).resolve().parents[1]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from gnss_integrity.fuse.ekf import ExtendedKalmanFilter, ImuReading
from gnss_integrity.fuse.trajectory_sim import SimConfig, simulate

# Same value Phase 3's fuse/validate.py used as its hand-set degraded-window
# weight. Phase 4's whole point is that this is now *selected by the
# detector*, per epoch, instead of being applied over a window we hardcoded.
DEFAULT_FLAGGED_TRUST_WEIGHT = 0.05


def flag_to_trust_weight(
    flag: int | bool,
    flagged_weight: float = DEFAULT_FLAGGED_TRUST_WEIGHT,
) -> float:
    """Map a binary anomaly flag to an EKF GPS trust weight.

    1 / True  (anomalous) -> ``flagged_weight``   (trust this fix less)
    0 / False (clean)     -> 1.0                   (nominal trust)

    A continuous score -> weight mapping is a sensible future refinement;
    the binary form is enough to demonstrate the integration and works
    identically for both detectors.
    """
    return float(flagged_weight) if int(flag) else 1.0


@dataclass
class PipelineConfig:
    scenario: str = "ds2"                 # TEXBAT scenario folder under data_root
    detector: str = "isolation_forest"    # "isolation_forest" | "autoencoder"
    # IsolationForest contamination. The detect/ module default (0.01) is
    # extremely conservative -- on TEXBAT it fires on only ~3% of the
    # attack window, so integrity-aware fusion would be indistinguishable
    # from naive. 0.05 gives the detector usable recall; the cost is more
    # false positives on clean epochs, which the report quantifies.
    contamination: float = 0.05
    ae_epochs: int = 40                    # autoencoder only
    clean_csv: str = "data/texbat/cleanStatic/features.csv"
    data_root: str = "data/texbat"
    trajectory_kind: str = "turns"        # fuse.trajectory_sim profile
    # The synthetic run is a clean run-up (``pre_attack_s``) followed by a
    # degraded-GPS window that runs to the end of the run
    # (onset + ``post_attack_s``). There is no clean-recovery tail -- the
    # real TEXBAT attacks run to the end of the recording, so the honest
    # analogue is "attack, no recovery". ``post_attack_s`` is kept short
    # enough (~45 s) that dead reckoning through a *rejected* attack stays
    # below the injected bias magnitude; for much longer windows even a
    # perfect detector stops helping (drift catches up with the bias).
    pre_attack_s: float = 30.0
    post_attack_s: float = 45.0
    # Optional, off-by-default: run a *bounded* attack of this many seconds
    # followed by ``recovery_s`` of clean GPS, instead of "spoofed from
    # onset to end of run". This exists to isolate the "sustained attack,
    # no recovery" failure mode -- see the Phase 4 findings. The recovery
    # segment replays the real record's *clean* leading epochs (TEXBAT has
    # no real post-attack clean data), which lets the flag stream clear
    # and tests whether the EKF re-anchors once trust is restored. When
    # ``None`` (default) the behaviour above is unchanged.
    attack_duration_s: float | None = None
    recovery_s: float = 45.0
    flagged_trust_weight: float = DEFAULT_FLAGGED_TRUST_WEIGHT
    seed: int = 42


@dataclass
class PipelineResult:
    t: np.ndarray                 # (N,) synthetic time base, s
    truth: np.ndarray             # (N, 2) ground-truth position
    fused_naive: np.ndarray       # (N, 2) EKF, every fix trusted at 1.0
    fused_integrity: np.ndarray   # (N, 2) EKF, trust weight from real detector
    fused_oracle: np.ndarray      # (N, 2) EKF, trust weight from perfect flags
    gps_t: np.ndarray             # (M,) times of the GPS fixes actually applied
    flag_series: np.ndarray       # (M,) real detector flag at each applied fix
    oracle_series: np.ndarray     # (M,) ground-truth spoof label at each fix
    trust_series: np.ndarray      # (M,) trust weight used (real detector) per fix
    attack_window: tuple[float, float]   # (start, end) in synthetic time
    recovery_window: tuple[float, float] | None = None  # set only for a
    # bounded attack: (attack_end, run_end), a clean-GPS re-anchor period
    metrics: dict = field(default_factory=dict)


# --- detector bridge -------------------------------------------------------


def _detector_flag_stream(
    cfg: PipelineConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Train the chosen detector on clean data, run it on the scenario.

    Returns:
        rel_t        : (K,) epoch time, seconds from the scenario's first epoch
        flags        : (K,) 0/1 anomaly flag per epoch (real detector output)
        oracle_flags : (K,) 0/1 ground-truth spoof label (known_spoof_window)
        attack_start : documented attack onset (rel seconds)
    """
    eval_csv = Path(cfg.data_root) / cfg.scenario / "features.csv"
    if not eval_csv.exists():
        raise FileNotFoundError(f"scenario feature CSV not found: {eval_csv}")

    if cfg.detector == "isolation_forest":
        from gnss_integrity.detect import isolation_forest as det_mod

        clean = det_mod.load_feature_csv(cfg.clean_csv)
        detector = det_mod.train_detector(
            [clean], contamination=cfg.contamination, random_state=cfg.seed
        )
        eval_df = det_mod.load_feature_csv(eval_csv)
        raw = det_mod.score_dataframe(detector, eval_df)
        flags = np.asarray(raw).astype(int)

    elif cfg.detector == "autoencoder":
        try:
            from gnss_integrity.detect import autoencoder as det_mod
        except Exception as exc:  # tensorflow not installed / not declared
            raise RuntimeError(
                "the autoencoder detector requires tensorflow, which is not "
                "installed in this environment. Use --detector isolation_forest, "
                "or `pip install tensorflow`."
            ) from exc

        clean = det_mod.load_feature_csv(cfg.clean_csv)
        detector = det_mod.train_autoencoder(
            [clean], epochs=cfg.ae_epochs, random_state=cfg.seed
        )
        eval_df = det_mod.load_feature_csv(eval_csv)
        out = det_mod.score_dataframe(detector, eval_df)
        # autoencoder returns (mse, y_pred); take the binary prediction.
        flags = np.asarray(out[1] if isinstance(out, tuple) else out).astype(int)

    else:
        raise ValueError(
            f"unknown detector {cfg.detector!r} "
            f"(use 'isolation_forest' or 'autoencoder')"
        )

    if "known_spoof_window" not in eval_df.columns:
        raise ValueError(
            f"{eval_csv} has no 'known_spoof_window' column -- can't locate the "
            f"attack onset. Regenerate it with the current texbat_loader.py."
        )
    oracle_flags = eval_df["known_spoof_window"].to_numpy().astype(int)
    if not (oracle_flags == 1).any():
        raise ValueError(
            f"scenario {cfg.scenario!r} has no labelled attack window; pick a "
            f"spoofing scenario (ds2/ds3/ds7/...)."
        )

    fix_time = eval_df["fix_time"].to_numpy()
    rel_t = fix_time - fix_time[0]
    attack_start = float(rel_t[np.argmax(oracle_flags == 1)])
    return rel_t, flags, oracle_flags, attack_start


# --- EKF runs ------------------------------------------------------------


def _run_ekf(result, trust_fn) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step the EKF over a simulated scenario.

    ``trust_fn(t)`` returns the GPS trust weight for a fix at synthetic
    time ``t``. This is the only thing that differs between the naive and
    integrity-aware runs -- the stepping is otherwise identical to
    ``fuse.validate.run_ekf``.

    Returns (fused_positions (N,2), applied_fix_times (M,), applied_weights (M,)).
    """
    gt, imu, gps = result.ground_truth, result.imu, result.gps
    n = len(gt.t)
    dt = gt.dt

    ekf = ExtendedKalmanFilter(
        initial_state=[gt.x[0], gt.y[0], gt.vx[0], gt.vy[0], gt.heading[0]],
        gps_position_noise_std=result.config.gps_noise_std,
    )
    fused = np.empty((n, 2))
    fix_times: list[float] = []
    weights: list[float] = []

    p = 0
    for i in range(n):
        if i > 0:
            ekf.predict(
                dt,
                ImuReading(
                    accel_x=float(imu.accel_body[i, 0]),
                    accel_y=float(imu.accel_body[i, 1]),
                    yaw_rate=float(imu.yaw_rate[i]),
                ),
            )
        t = imu.t[i]
        while p < len(gps.t) and gps.t[p] <= t + dt / 2:
            if gps.available[p]:
                w = float(trust_fn(float(gps.t[p])))
                ekf.update_gps([gps.x[p], gps.y[p]], trust_weight=w)
                fix_times.append(float(gps.t[p]))
                weights.append(w)
            p += 1
        fused[i] = ekf.position

    return fused, np.asarray(fix_times), np.asarray(weights)


def _rmse(a: np.ndarray, b: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        a, b = a[mask], b[mask]
    if len(a) == 0:
        return float("nan")
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def run_pipeline(cfg: PipelineConfig | None = None) -> PipelineResult:
    """Run the full detector -> EKF integration and both comparison filters."""
    cfg = cfg or PipelineConfig()

    rel_t, flags, oracle_flags, attack_start = _detector_flag_stream(cfg)
    scenario_dur = float(rel_t[-1])

    w_start = max(0.0, attack_start - cfg.pre_attack_s)
    if cfg.attack_duration_s is None:
        # Default: degraded GPS from onset to the end of the run (the
        # honest analogue of TEXBAT's real, non-terminating attacks).
        w_end = min(scenario_dur, attack_start + cfg.post_attack_s)
        syn_dur = w_end - w_start
        attack_window = (attack_start - w_start, w_end - w_start)
        recovery_start: float | None = None
    else:
        # Bounded attack + a synthetic clean-GPS recovery tail. Used to
        # isolate the "sustained attack, no recovery" failure mode.
        atk_len = float(cfg.attack_duration_s)
        attack_window = (cfg.pre_attack_s, cfg.pre_attack_s + atk_len)
        recovery_start = cfg.pre_attack_s + atk_len
        syn_dur = recovery_start + cfg.recovery_s
    recovery_window = None if recovery_start is None else (recovery_start, syn_dur)

    sim_cfg = _replace(SimConfig(), duration_s=syn_dur, seed=cfg.seed)
    result = simulate(
        cfg.trajectory_kind,
        sim_cfg,
        dropout_windows=[],
        degraded_windows=[attack_window],
    )
    gt = result.ground_truth
    truth = gt.position()

    # Map a synthetic timestamp back to a real-scenario timestamp so the
    # detector flags can be looked up. This is exactly ``w_start + t`` in
    # the default case; in the bounded case the recovery segment replays
    # the real record's clean leading epochs (from real t = 0), so the
    # flag stream clears just as the synthetic fault does.
    def _real_time(t_syn: float) -> float:
        if recovery_start is not None and t_syn >= recovery_start:
            return t_syn - recovery_start
        return w_start + t_syn

    def _nearest(arr, t_syn):
        return int(arr[int(np.argmin(np.abs(rel_t - _real_time(t_syn))))])

    def trust_real(t_syn):
        return flag_to_trust_weight(_nearest(flags, t_syn), cfg.flagged_trust_weight)

    def trust_oracle(t_syn):
        return flag_to_trust_weight(_nearest(oracle_flags, t_syn), cfg.flagged_trust_weight)

    def trust_naive(_t):
        return 1.0

    fused_naive, _, _ = _run_ekf(result, trust_naive)
    fused_integrity, fix_t, fix_w = _run_ekf(result, trust_real)
    fused_oracle, _, _ = _run_ekf(result, trust_oracle)

    flag_series = np.array([_nearest(flags, t) for t in fix_t], dtype=int)
    oracle_series = np.array([_nearest(oracle_flags, t) for t in fix_t], dtype=int)

    atk0, atk1 = attack_window
    tt = gt.t
    in_attack = (tt >= atk0) & (tt <= atk1)
    pre_attack = tt < atk0
    post_recovery = tt > atk1  # all-False (-> RMSE nan) unless a bounded attack
    gps_in_attack = (fix_t >= atk0) & (fix_t <= atk1)
    gps_pre_attack = fix_t < atk0

    def _row(fused):
        return {
            "overall": _rmse(fused, truth),
            "pre": _rmse(fused, truth, pre_attack),
            "attack": _rmse(fused, truth, in_attack),
            "recovery": _rmse(fused, truth, post_recovery),
        }

    metrics = {
        "scenario": cfg.scenario,
        "detector": cfg.detector,
        "flagged_trust_weight": cfg.flagged_trust_weight,
        "naive": _row(fused_naive),
        "integrity_real": _row(fused_integrity),
        "integrity_oracle": _row(fused_oracle),
        "detector_recall_in_attack": float(flag_series[gps_in_attack].mean())
        if gps_in_attack.any() else float("nan"),
        "detector_fp_pre_attack": float(flag_series[gps_pre_attack].mean())
        if gps_pre_attack.any() else float("nan"),
        "detector_fp_recovery": float(flag_series[fix_t > atk1].mean())
        if (fix_t > atk1).any() else float("nan"),
        "n_gps_fixes": int(len(fix_t)),
        "bounded_attack": cfg.attack_duration_s is not None,
    }

    return PipelineResult(
        t=tt,
        truth=truth,
        fused_naive=fused_naive,
        fused_integrity=fused_integrity,
        fused_oracle=fused_oracle,
        gps_t=fix_t,
        flag_series=flag_series,
        oracle_series=oracle_series,
        trust_series=fix_w,
        attack_window=attack_window,
        recovery_window=recovery_window,
        metrics=metrics,
    )


# --- reporting --------------------------------------------------------


def print_report(res: PipelineResult) -> None:
    m = res.metrics
    a0, a1 = res.attack_window
    print("\n=== Phase 4: integrity-aware fusion ===")
    print(f"scenario         : {m['scenario']}  (real {m['detector']} flags)")
    print(f"synthetic run    : {res.t[-1]:.0f} s, {m['n_gps_fixes']} GPS fixes")
    if res.recovery_window is not None:
        r0, r1 = res.recovery_window
        print(f"attack window    : {a0:.0f}-{a1:.0f} s  (bounded synthetic degraded GPS)")
        print(f"recovery window  : {r0:.0f}-{r1:.0f} s  (clean GPS; flag stream "
              f"replays real clean epochs -- synthetic stand-in, TEXBAT has "
              f"no real post-attack clean data)")
    else:
        print(f"attack window    : {a0:.0f}-{a1:.0f} s  (synthetic degraded GPS to "
              f"end of run, co-timed with the real scenario's documented attack)")
    print(f"flag -> trust    : anomalous fix trusted at weight "
          f"{m['flagged_trust_weight']:g} (else 1.0)")
    print(f"real detector    : recall in attack = {m['detector_recall_in_attack']:.2f}, "
          f"false-positive rate before attack = {m['detector_fp_pre_attack']:.2f}")
    print()

    rows = [("overall", "overall"), ("pre-attack", "pre"), ("in attack", "attack")]
    if np.isfinite(m["naive"]["recovery"]):
        rows.append(("post-attack recov", "recovery"))
    hdr = f"{'RMSE (m)':18} {'naive':>9} {'integ (real)':>14} {'integ (oracle)':>16}"
    print(hdr)
    print("-" * len(hdr))
    for label, key in rows:
        print(f"{label:18} {m['naive'][key]:9.2f} {m['integrity_real'][key]:14.2f} "
              f"{m['integrity_oracle'][key]:16.2f}")
    print()

    n_atk = m["naive"]["attack"]
    r_atk = m["integrity_real"]["attack"]
    o_atk = m["integrity_oracle"]["attack"]

    def _verdict(name, val):
        if not (np.isfinite(val) and np.isfinite(n_atk)):
            return
        if val < 0.95 * n_atk:
            print(f"  {name}: {n_atk / max(val, 1e-9):.2f}x better than naive "
                  f"through the attack ({val:.1f} vs {n_atk:.1f} m).")
        elif val <= 1.05 * n_atk:
            print(f"  {name}: about the same as naive through the attack "
                  f"({val:.1f} vs {n_atk:.1f} m).")
        else:
            print(f"  {name}: WORSE than naive through the attack "
                  f"({val:.1f} vs {n_atk:.1f} m) -- reported as-is, not a bug.")

    _verdict("integrity (real detector)  ", r_atk)
    _verdict("integrity (oracle detector)", o_atk)

    if np.isfinite(m["naive"]["recovery"]):
        n_rec = m["naive"]["recovery"]
        r_rec = m["integrity_real"]["recovery"]
        tag = ("re-anchors: BETTER than naive" if r_rec < 0.95 * n_rec
               else "re-anchors: ~equal to naive" if r_rec <= 1.05 * n_rec
               else "re-anchors: still worse than naive")
        print(f"  recovery window -- {tag} ({r_rec:.1f} vs {n_rec:.1f} m). "
              f"A finite, bounded recovery number here means the EKF does "
              f"re-lock onto GPS once trust is restored.")

    print("\n  Read: the oracle row is the ceiling the wiring can reach; the gap "
          "\n  between it and the real-detector row is pure detector quality "
          "\n  (detection latency + false positives).")


def make_plot(res: PipelineResult, out_path: str) -> None:
    a0, a1 = res.attack_window
    t = res.t
    err_naive = np.linalg.norm(res.fused_naive - res.truth, axis=1)
    err_real = np.linalg.norm(res.fused_integrity - res.truth, axis=1)
    err_oracle = np.linalg.norm(res.fused_oracle - res.truth, axis=1)

    fig, (ax_err, ax_w) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1]},
    )

    ax_err.axvspan(a0, a1, color="red", alpha=0.10, label="attack window (degraded GPS)")
    if res.recovery_window is not None:
        ax_err.axvspan(*res.recovery_window, color="green", alpha=0.07,
                       label="recovery window (clean GPS)")
    ax_err.plot(t, err_naive, color="#1f77b4", linewidth=1.3,
                label="naive fusion (GPS always trusted)")
    ax_err.plot(t, err_real, color="#2ca02c", linewidth=1.6,
                label="integrity-aware (real detector flags)")
    ax_err.plot(t, err_oracle, color="#9467bd", linewidth=1.4, linestyle="--",
                label="integrity-aware (oracle / perfect flags)")
    ax_err.set_ylabel("position error vs truth (m)")
    ax_err.set_title(
        f"Naive vs integrity-aware fusion -- {res.metrics['scenario']} "
        f"({res.metrics['detector']} flags), synthetic trajectory"
    )
    ax_err.legend(loc="upper left", fontsize=8)

    # Trust-weight strip: makes detector latency and false positives visible.
    ax_w.axvspan(a0, a1, color="red", alpha=0.10)
    if res.recovery_window is not None:
        ax_w.axvspan(*res.recovery_window, color="green", alpha=0.07)
    ax_w.step(res.gps_t, res.trust_series, where="post", color="#2ca02c", linewidth=1.2)
    flagged = res.flag_series == 1
    ax_w.scatter(res.gps_t[flagged], res.trust_series[flagged], s=16,
                 color="red", zorder=5, label="epoch flagged by real detector")
    lo = float(res.trust_series.min()) if len(res.trust_series) else 0.05
    ax_w.set_ylim(-0.1, 1.15)
    ax_w.set_yticks([lo, 1.0])
    ax_w.set_ylabel("GPS trust weight\n(real detector)")
    ax_w.set_xlabel("time since start of synthetic run (s)")
    ax_w.legend(loc="center left", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


def run(cfg: PipelineConfig, out_path: str) -> PipelineResult:
    res = run_pipeline(cfg)
    print_report(res)
    make_plot(res, out_path)
    return res


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Phase 4: detector-driven integrity-aware GPS/IMU fusion."
    )
    parser.add_argument("--scenario", default="ds2",
                        help="TEXBAT scenario folder under data/texbat/ (ds2, ds3, ds7, ...)")
    parser.add_argument("--detector", default="isolation_forest",
                        choices=["isolation_forest", "autoencoder"])
    parser.add_argument("--contamination", type=float, default=0.05,
                        help="IsolationForest contamination (recall / false-positive knob)")
    parser.add_argument("--kind", default="turns", choices=["line", "turns"],
                        help="synthetic trajectory profile")
    parser.add_argument("--flagged-trust-weight", type=float,
                        default=DEFAULT_FLAGGED_TRUST_WEIGHT)
    parser.add_argument("--attack-duration", type=float, default=None,
                        help="if set, run a BOUNDED attack of this many seconds "
                             "followed by --recovery seconds of clean GPS "
                             "(default: attack runs to end of the synthetic run)")
    parser.add_argument("--recovery", type=float, default=45.0,
                        help="clean-GPS recovery seconds after a bounded attack "
                             "(only used with --attack-duration)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", default="pipeline_integrity_vs_naive.png")
    args = parser.parse_args()

    cfg = PipelineConfig(
        scenario=args.scenario,
        detector=args.detector,
        contamination=args.contamination,
        trajectory_kind=args.kind,
        flagged_trust_weight=args.flagged_trust_weight,
        attack_duration_s=args.attack_duration,
        recovery_s=args.recovery,
        seed=args.seed,
    )
    run(cfg, args.output)
