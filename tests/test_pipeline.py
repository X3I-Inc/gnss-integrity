"""
Tests for Phase 4 -- detector -> EKF trust-weight integration (pipeline.py).

Covers:
  * the flag -> trust_weight mapping in isolation,
  * an end-to-end smoke test that the pipeline runs and returns
    consistent, finite output,
  * the invariant that with a *perfect* (oracle) detector, integrity-
    aware fusion is no worse than naive fusion through the attack,
  * a test that pins the current, honest Phase 4 finding on the default
    (*sustained*, no-recovery) scenario: with the real IsolationForest
    detector, integrity-aware fusion is actually *worse* than naive
    in-window on ds2 -- detection latency + pre-attack false positives,
    not a wiring bug, and
  * the Phase-4.1 follow-up: on a *bounded* attack followed by a
    clean-GPS recovery period, that negative result largely reverses --
    the EKF re-anchors once trust is restored and overall RMSE returns to
    parity-or-better. This confirms "sustained attack, no recovery" was
    the dominant cause, so both results are pinned here side by side.

These use the committed TEXBAT feature CSVs (small, ~120 KB each) and
train a fast IsolationForest, so no network / large data is needed.
"""

from __future__ import annotations

import numpy as np
import pytest

from gnss_integrity.pipeline import (
    DEFAULT_FLAGGED_TRUST_WEIGHT,
    PipelineConfig,
    PipelineResult,
    flag_to_trust_weight,
    run_pipeline,
)


# --- flag -> trust_weight mapping ---------------------------------------


def test_flag_to_trust_weight_default():
    assert flag_to_trust_weight(1) == DEFAULT_FLAGGED_TRUST_WEIGHT
    assert flag_to_trust_weight(0) == 1.0
    # booleans and numpy ints must behave the same as plain ints
    assert flag_to_trust_weight(True) == DEFAULT_FLAGGED_TRUST_WEIGHT
    assert flag_to_trust_weight(np.int64(0)) == 1.0


def test_flag_to_trust_weight_custom_weight():
    assert flag_to_trust_weight(1, flagged_weight=0.2) == 0.2
    assert flag_to_trust_weight(0, flagged_weight=0.2) == 1.0
    # a flagged fix is always trusted strictly less than an unflagged one
    assert flag_to_trust_weight(1, 0.5) < flag_to_trust_weight(0, 0.5)


# --- end-to-end pipeline ----------------------------------------------


@pytest.fixture(scope="module")
def ds2_result() -> PipelineResult:
    # Default config == ds2, isolation_forest, sustained attack (degraded
    # GPS from onset to the end of the run -- no recovery period).
    return run_pipeline(PipelineConfig())


@pytest.fixture(scope="module")
def ds2_bounded_result() -> PipelineResult:
    # Bounded attack (25 s) + a 45 s clean-GPS recovery tail. Same
    # everything else as the default.
    return run_pipeline(PipelineConfig(attack_duration_s=25.0, recovery_s=45.0))


def test_pipeline_runs_and_output_is_consistent(ds2_result: PipelineResult):
    r = ds2_result
    n = len(r.t)
    assert n > 100
    for arr in (r.truth, r.fused_naive, r.fused_integrity, r.fused_oracle):
        assert arr.shape == (n, 2)
        assert np.all(np.isfinite(arr))

    m = len(r.gps_t)
    assert m > 10
    for arr in (r.flag_series, r.oracle_series, r.trust_series):
        assert arr.shape == (m,)
    # flags are binary; trust weights are one of the two mapped values
    assert set(np.unique(r.flag_series)).issubset({0, 1})
    assert set(np.unique(r.trust_series)).issubset(
        {1.0, r.metrics["flagged_trust_weight"]}
    )

    a0, a1 = r.attack_window
    assert 0.0 <= a0 < a1 <= r.t[-1] + 1e-6

    for key in ("naive", "integrity_real", "integrity_oracle"):
        row = r.metrics[key]
        assert np.isfinite(row["overall"])
        assert np.isfinite(row["attack"])


def test_detector_actually_fires_in_attack_window(ds2_result: PipelineResult):
    # If the detector never flagged anything, the whole comparison would
    # be vacuous (integrity == naive). Guard against that.
    m = ds2_result.metrics
    assert m["detector_recall_in_attack"] > 0.1
    # trust weight was actually pulled down for some fixes
    assert np.any(ds2_result.trust_series < 1.0)


def test_oracle_integrity_no_worse_than_naive_through_attack(ds2_result: PipelineResult):
    """With a perfect, zero-latency detector the integration must not hurt
    -- and in practice it clearly helps through a bounded attack."""
    m = ds2_result.metrics
    naive_atk = m["naive"]["attack"]
    oracle_atk = m["integrity_oracle"]["attack"]
    assert oracle_atk <= naive_atk  # "no worse" -- the invariant
    # document the size of the win we currently see (informational)
    assert oracle_atk < 0.9 * naive_atk


def test_real_detector_integrity_currently_underperforms(ds2_result: PipelineResult):
    """HONEST FINDING, pinned so it can't change silently.

    As of Phase 4, on the *default* config (ds2 + the "turns" trajectory),
    feeding the *real* IsolationForest flags into the trust weight makes
    fusion WORSE than naive through the attack: the detector needs a few
    seconds to lock on, and during that lag the biased fixes are still
    trusted at 1.0 and corrupt the velocity state, after which dead
    reckoning on a bad velocity drifts faster than the spoof bias itself.
    Pre-attack false positives add to the damage.

    This is not a wiring bug -- the oracle test passes, other configs
    (e.g. --scenario ds3 --kind line) come out ahead, and the *dominant*
    cause is the sustained/no-recovery attack profile: see
    ``test_bounded_attack_recovery_reanchors``, where a bounded attack
    with a recovery period brings the overall RMSE back to parity. If a
    detector/mapping improvement makes the default come out >= naive, this
    test SHOULD fail -- update it and the Phase 4 write-up together.
    """
    m = ds2_result.metrics
    naive_atk = m["naive"]["attack"]
    real_atk = m["integrity_real"]["attack"]
    assert real_atk > naive_atk, (
        f"real-detector integrity ({real_atk:.1f} m) is no longer worse than "
        f"naive ({naive_atk:.1f} m) -- the detector improved; update this test "
        f"and the Phase 4 narrative."
    )
    # ... but it should not be catastrophically worse -- guard against a
    # real regression (e.g. trust weight wired backwards).
    assert real_atk < 2.0 * naive_atk


def test_pipeline_deterministic():
    a = run_pipeline(PipelineConfig(seed=7))
    b = run_pipeline(PipelineConfig(seed=7))
    assert a.metrics["integrity_real"]["overall"] == pytest.approx(
        b.metrics["integrity_real"]["overall"]
    )


# --- bounded attack + recovery (Phase 4.1) ---------------------------


def test_bounded_attack_is_off_by_default(ds2_result: PipelineResult):
    """The bounded-attack knob must not touch the default behaviour."""
    assert PipelineConfig().attack_duration_s is None
    assert ds2_result.recovery_window is None
    # no recovery segment -> the recovery RMSE row is NaN, not a number
    assert not np.isfinite(ds2_result.metrics["naive"]["recovery"])
    # default attack still runs to the end of the synthetic run
    _a0, a1 = ds2_result.attack_window
    assert a1 == pytest.approx(ds2_result.t[-1], abs=1.0 / 50.0)


def test_bounded_attack_has_a_recovery_window(ds2_bounded_result: PipelineResult):
    r = ds2_bounded_result
    assert r.recovery_window is not None
    r0, r1 = r.recovery_window
    a0, a1 = r.attack_window
    assert a0 == pytest.approx(30.0) and a1 == pytest.approx(55.0)
    assert r0 == pytest.approx(55.0) and r1 == pytest.approx(100.0)
    for key in ("naive", "integrity_real", "integrity_oracle"):
        assert np.isfinite(r.metrics[key]["recovery"])
    # detector flags must genuinely clear during the replayed-clean recovery
    assert r.metrics["detector_fp_recovery"] == pytest.approx(0.0, abs=0.05)


def test_bounded_attack_recovery_reanchors(ds2_bounded_result: PipelineResult):
    """The headline Phase 4.1 result: with a bounded attack + recovery, the
    real-detector integrity filter re-anchors to GPS once trust is
    restored, and overall RMSE is back to parity-or-better vs naive --
    unlike the sustained-attack default, where it is ~23% worse.
    """
    m = ds2_bounded_result.metrics
    naive, real = m["naive"], m["integrity_real"]

    # 1. it re-locks onto GPS in the recovery window -- in fact better than
    #    naive there (naive is still shedding the bias it chased).
    assert real["recovery"] < naive["recovery"]

    # 2. overall RMSE is no longer a clear loss -- parity within 10%.
    assert real["overall"] <= 1.10 * naive["overall"]

    # 3. the oracle still shows the ceiling: clearly better in-attack.
    assert m["integrity_oracle"]["attack"] < 0.9 * naive["attack"]

    # 4. HONEST residual: a bounded attack does NOT fully fix the
    #    in-attack shortfall (causes 1-2 -- latency + false positives --
    #    still cost something). This is expected; pin it so a real
    #    improvement here is noticed.
    assert real["attack"] >= naive["attack"]


def test_bounded_recovery_shrinks_the_real_vs_naive_gap(
    ds2_result: PipelineResult, ds2_bounded_result: PipelineResult
):
    """Direct evidence that "sustained, no recovery" was the dominant
    cause: the real-vs-naive in-attack RMSE ratio is markedly closer to
    1.0 once a recovery period exists."""
    def ratio(res: PipelineResult) -> float:
        m = res.metrics
        return m["integrity_real"]["attack"] / m["naive"]["attack"]

    sustained_ratio = ratio(ds2_result)      # ~1.23 (23% worse)
    bounded_ratio = ratio(ds2_bounded_result)  # ~1.10 (10% worse)
    assert sustained_ratio > 1.15
    assert bounded_ratio < sustained_ratio - 0.05


def test_lower_trust_weight_rejects_harder(ds2_result: PipelineResult):
    """A smaller flagged weight => the oracle filter leans harder on dead
    reckoning => it departs further from the (biased) GPS during the
    attack. This confirms the weight actually controls the coupling."""
    soft = run_pipeline(PipelineConfig(flagged_trust_weight=0.5))
    hard = run_pipeline(PipelineConfig(flagged_trust_weight=0.005))
    # distance of the oracle estimate from the raw (biased) fixes is larger
    # when we trust flagged fixes less -- compare via RMSE-vs-truth spread
    soft_gap = abs(soft.metrics["integrity_oracle"]["attack"]
                   - soft.metrics["naive"]["attack"])
    hard_gap = abs(hard.metrics["integrity_oracle"]["attack"]
                   - hard.metrics["naive"]["attack"])
    assert hard_gap > soft_gap
