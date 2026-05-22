"""Unit tests for evaluation metrics.

These verify mathematical correctness against known values from textbooks
and reference implementations.
"""

from __future__ import annotations

import math

import pytest

from tennis_predictor.backtest.metrics import (
    accuracy,
    brier_score,
    evaluate_predictions,
    expected_calibration_error,
    log_loss,
)


@pytest.mark.unit
class TestLogLoss:
    """Log loss / cross-entropy."""

    def test_perfect_predictions(self) -> None:
        """Predictions of 1.0 for outcome=1 give ~0 log loss."""
        result = log_loss([0.999, 0.999, 0.999], [1, 1, 1])
        assert result < 0.01

    def test_terrible_predictions(self) -> None:
        """Predictions of 0 for outcome=1 should be huge log loss."""
        result = log_loss([0.001, 0.001, 0.001], [1, 1, 1])
        assert result > 5.0

    def test_random_guessing(self) -> None:
        """P=0.5 gives log_loss = ln(2) ≈ 0.693."""
        result = log_loss([0.5] * 100, [1, 0] * 50)
        assert abs(result - math.log(2)) < 0.001

    def test_perfect_calibration_at_70_30(self) -> None:
        """If we always predict 70% and 70% actually win, log_loss is fixed."""
        # 7 ones, 3 zeros, all predicted with p=0.7
        probs = [0.7] * 10
        outcomes = [1] * 7 + [0] * 3
        expected = -(7 * math.log(0.7) + 3 * math.log(0.3)) / 10
        assert abs(log_loss(probs, outcomes) - expected) < 1e-9

    def test_handles_zero_probability(self) -> None:
        """Clipping should prevent inf when p=0 and y=1."""
        result = log_loss([0.0], [1])
        assert math.isfinite(result)
        assert result > 30  # Very high, but not infinite

    def test_handles_one_probability(self) -> None:
        """Clipping should prevent inf when p=1 and y=0."""
        result = log_loss([1.0], [0])
        assert math.isfinite(result)
        assert result > 30

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="Length mismatch"):
            log_loss([0.5, 0.5], [1])

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="empty"):
            log_loss([], [])


@pytest.mark.unit
class TestBrierScore:
    """Brier score (squared error of probabilities)."""

    def test_perfect_predictions(self) -> None:
        """Perfect predictions give Brier = 0."""
        assert brier_score([1.0, 0.0, 1.0], [1, 0, 1]) == 0.0

    def test_worst_predictions(self) -> None:
        """Worst case: predict opposite. Brier = 1.0."""
        assert brier_score([0.0, 1.0], [1, 0]) == 1.0

    def test_random_guessing(self) -> None:
        """P=0.5 always gives Brier = 0.25."""
        result = brier_score([0.5] * 100, [1, 0] * 50)
        assert abs(result - 0.25) < 1e-9

    def test_specific_value(self) -> None:
        """Manual calculation: p=[0.8, 0.3], y=[1, 0]
        Brier = ((0.8-1)^2 + (0.3-0)^2) / 2 = (0.04 + 0.09) / 2 = 0.065"""
        result = brier_score([0.8, 0.3], [1, 0])
        assert abs(result - 0.065) < 1e-9


@pytest.mark.unit
class TestAccuracy:
    """Classification accuracy at 0.5 threshold."""

    def test_all_correct(self) -> None:
        assert accuracy([0.9, 0.1, 0.8], [1, 0, 1]) == 1.0

    def test_all_wrong(self) -> None:
        assert accuracy([0.9, 0.1], [0, 1]) == 0.0

    def test_threshold_at_0_5(self) -> None:
        """p > 0.5 → predicts positive; p <= 0.5 → predicts negative."""
        # p=0.5 is treated as negative prediction
        assert accuracy([0.5], [0]) == 1.0
        assert accuracy([0.5], [1]) == 0.0
        # p=0.51 is positive
        assert accuracy([0.51], [1]) == 1.0
        assert accuracy([0.51], [0]) == 0.0


@pytest.mark.unit
class TestExpectedCalibrationError:
    """Expected calibration error."""

    def test_perfect_calibration(self) -> None:
        """When predicted prob matches actual rate exactly, ECE = 0."""
        # 100 predictions at 0.7, exactly 70 of them positive
        probs = [0.7] * 100
        outcomes = [1] * 70 + [0] * 30
        ece, _ = expected_calibration_error(probs, outcomes, n_bins=10)
        assert ece < 0.001

    def test_complete_miscalibration(self) -> None:
        """Predict 90%, actually 10% positive → large ECE."""
        probs = [0.9] * 100
        outcomes = [1] * 10 + [0] * 90
        ece, _ = expected_calibration_error(probs, outcomes, n_bins=10)
        assert ece > 0.7

    def test_returns_bin_summaries(self) -> None:
        """Bins should reflect the data structure."""
        probs = [0.1, 0.2, 0.7, 0.8]
        outcomes = [0, 0, 1, 1]
        ece, bins = expected_calibration_error(probs, outcomes, n_bins=10)
        # Should have 4 non-empty bins (one per unique probability)
        assert len(bins) == 4
        # Total count across bins equals input length
        assert sum(b[2] for b in bins) == 4


@pytest.mark.unit
class TestEvaluatePredictions:
    """End-to-end evaluation."""

    def test_returns_all_metrics(self) -> None:
        """All fields populated, no errors."""
        probs = [0.6, 0.4, 0.7, 0.3, 0.8]
        outcomes = [1, 0, 1, 0, 1]
        m = evaluate_predictions(probs, outcomes)

        assert m.n_predictions == 5
        assert 0 <= m.accuracy <= 1
        assert m.log_loss >= 0
        assert 0 <= m.brier_score <= 1
        assert 0 <= m.calibration_error <= 1
        assert m.calibration_bins is not None
        assert len(m.calibration_bins) > 0

    def test_pretty_print_works(self) -> None:
        """pretty_print returns a string without crashing."""
        m = evaluate_predictions([0.6, 0.4], [1, 0])
        output = m.pretty_print()
        assert isinstance(output, str)
        assert "Accuracy" in output
        assert "Log loss" in output


@pytest.mark.unit
class TestKnownBenchmarks:
    """Sanity checks against tennis literature benchmarks.

    From Kovalchik (2016) and tennisabstract.com Elo backtests:
    - Random guessing on tennis: accuracy ~50%, log_loss ~0.693
    - Bookmaker closing lines: accuracy ~70%, log_loss ~0.55
    - Surface-Adjusted Elo: accuracy ~68%, log_loss ~0.59
    """

    def test_random_is_correctly_evaluated(self) -> None:
        """Verify random predictions give expected benchmark values."""
        import random
        random.seed(42)
        probs = [random.random() for _ in range(10000)]
        outcomes = [random.randint(0, 1) for _ in range(10000)]
        m = evaluate_predictions(probs, outcomes)

        # Random should be close to: acc=0.5, log_loss=1.0+ (no skill), brier ~0.33
        assert 0.45 < m.accuracy < 0.55  # Close to 50%
        assert m.log_loss > 0.9  # Worse than constant 0.5

    def test_constant_0_5_is_correctly_evaluated(self) -> None:
        """Always predicting 0.5 is the 'no information' baseline."""
        probs = [0.5] * 10000
        outcomes = [1] * 5000 + [0] * 5000
        m = evaluate_predictions(probs, outcomes)

        assert abs(m.accuracy - 0.5) < 0.01
        assert abs(m.log_loss - math.log(2)) < 0.01
        assert abs(m.brier_score - 0.25) < 0.01
