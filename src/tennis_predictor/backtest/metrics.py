"""Evaluation metrics for probabilistic predictions.

For betting models, accuracy alone is insufficient. We need:

- **Log Loss** (a.k.a. cross-entropy): penalizes confident wrong predictions
  exponentially. The "right" metric for probabilistic predictions.
  Lower is better. Pure guessing (P=0.5) gives log_loss = 0.693.

- **Brier Score**: mean squared error of probabilities. Bounded [0, 1],
  lower is better. Pure guessing gives Brier = 0.25.

- **Expected Calibration Error (ECE)**: are the predicted probabilities
  honest? If we say "70% confident" do we win 70% of those bets?
  Lower is better. A perfectly calibrated model has ECE = 0.

- **Accuracy**: simple win/lose count. Easy to interpret but misleading
  for evaluating probabilistic models.

References:
- Brier (1950): "Verification of forecasts expressed in terms of probability"
- Naeini et al (2015): "Obtaining well calibrated probabilities"
- DeGroot & Fienberg (1983): "The comparison and evaluation of forecasters"
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class EvaluationMetrics:
    """Container for all evaluation metrics on a set of predictions.

    All metrics are 'lower is better' except accuracy (higher is better).
    """

    n_predictions: int
    accuracy: float
    log_loss: float
    brier_score: float
    calibration_error: float

    # Reliability breakdown - one entry per probability bin
    calibration_bins: list[tuple[float, float, int]] | None = None
    # Each tuple: (mean_predicted, mean_actual, count)

    def pretty_print(self) -> str:
        """Format metrics for human reading."""
        lines = [
            f"N predictions:     {self.n_predictions:>10,}",
            f"Accuracy:          {self.accuracy:>10.4f}  ({self.accuracy * 100:.2f}%)",
            f"Log loss:          {self.log_loss:>10.4f}  (lower better; 0.693 = random)",
            f"Brier score:       {self.brier_score:>10.4f}  (lower better; 0.250 = random)",
            f"Calibration error: {self.calibration_error:>10.4f}  (lower better; 0.00 = perfect)",
        ]
        return "\n".join(lines)


def log_loss(predicted_probs: list[float], actual_outcomes: list[int]) -> float:
    """Compute binary log loss (cross-entropy).

    log_loss = -1/N * Σ [y * log(p) + (1-y) * log(1-p)]

    Args:
        predicted_probs: List of P(positive class), all in [0, 1]
        actual_outcomes: List of 0/1 outcomes (1 = positive class occurred)

    Returns:
        Log loss value. Lower is better. Bounded [0, ∞).
        Random guessing (P=0.5) gives ~0.693.

    Raises:
        ValueError: If lists have different lengths or invalid probabilities.
    """
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError(
            f"Length mismatch: {len(predicted_probs)} probs vs "
            f"{len(actual_outcomes)} outcomes"
        )
    if not predicted_probs:
        raise ValueError("Cannot compute log_loss on empty input")

    # Clip probabilities to avoid log(0). Standard epsilon is 1e-15.
    eps = 1e-15
    total = 0.0
    for p, y in zip(predicted_probs, actual_outcomes, strict=True):
        p_clipped = max(eps, min(1.0 - eps, p))
        if y == 1:
            total -= math.log(p_clipped)
        else:
            total -= math.log(1.0 - p_clipped)

    return total / len(predicted_probs)


def brier_score(predicted_probs: list[float], actual_outcomes: list[int]) -> float:
    """Compute Brier score (mean squared error of probabilities).

    brier = 1/N * Σ (p - y)^2

    Args:
        predicted_probs: List of P(positive class)
        actual_outcomes: List of 0/1 outcomes

    Returns:
        Brier score. Lower is better. Bounded [0, 1].
        Random guessing gives 0.25.
    """
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("Length mismatch")
    if not predicted_probs:
        raise ValueError("Empty input")

    total = sum(
        (p - y) ** 2
        for p, y in zip(predicted_probs, actual_outcomes, strict=True)
    )
    return total / len(predicted_probs)


def accuracy(predicted_probs: list[float], actual_outcomes: list[int]) -> float:
    """Compute classification accuracy (using 0.5 threshold).

    Args:
        predicted_probs: List of probabilities P(positive class)
        actual_outcomes: List of 0/1 outcomes

    Returns:
        Fraction of predictions where (p > 0.5) matches outcome.
    """
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("Length mismatch")
    if not predicted_probs:
        raise ValueError("Empty input")

    correct = sum(
        1 for p, y in zip(predicted_probs, actual_outcomes, strict=True)
        if (p > 0.5 and y == 1) or (p <= 0.5 and y == 0)
    )
    return correct / len(predicted_probs)


def expected_calibration_error(
    predicted_probs: list[float],
    actual_outcomes: list[int],
    n_bins: int = 10,
) -> tuple[float, list[tuple[float, float, int]]]:
    """Compute Expected Calibration Error (ECE).

    Bins predictions by predicted probability. For each bin, compares
    mean predicted prob to actual win rate. ECE is the weighted average
    of these gaps (weighted by bin size).

    A perfectly calibrated model: when it says "70% confident", actually
    wins 70% of the time → ECE = 0.

    Args:
        predicted_probs: List of P(positive class)
        actual_outcomes: List of 0/1 outcomes
        n_bins: Number of probability bins (default 10 = 10% buckets)

    Returns:
        (ece, bins) where bins is a list of (mean_predicted, mean_actual, count)
        tuples, one per bin (excluding empty bins).
    """
    if len(predicted_probs) != len(actual_outcomes):
        raise ValueError("Length mismatch")
    if not predicted_probs:
        raise ValueError("Empty input")

    n = len(predicted_probs)
    bin_edges = [i / n_bins for i in range(n_bins + 1)]

    # Assign predictions to bins
    bin_data: list[list[tuple[float, int]]] = [[] for _ in range(n_bins)]
    for p, y in zip(predicted_probs, actual_outcomes, strict=True):
        # Find bin index. p=1.0 should go in last bin
        bin_idx = min(int(p * n_bins), n_bins - 1)
        bin_data[bin_idx].append((p, y))

    # Compute ECE and bin summaries
    total_ece = 0.0
    bin_summaries: list[tuple[float, float, int]] = []

    for bin_items in bin_data:
        if not bin_items:
            continue
        bin_size = len(bin_items)
        mean_pred = sum(p for p, _ in bin_items) / bin_size
        mean_actual = sum(y for _, y in bin_items) / bin_size
        gap = abs(mean_pred - mean_actual)
        total_ece += (bin_size / n) * gap
        bin_summaries.append((mean_pred, mean_actual, bin_size))

    return total_ece, bin_summaries


def evaluate_predictions(
    predicted_probs: list[float],
    actual_outcomes: list[int],
    n_bins: int = 10,
) -> EvaluationMetrics:
    """Compute all evaluation metrics at once.

    Convenience function that runs all metric calculations on the same data.

    Args:
        predicted_probs: List of predicted probabilities P(winner wins)
        actual_outcomes: List of 1s (winner always wins in our setup)
                        OR mix of 1s and 0s if we're predicting arbitrary players
        n_bins: Calibration bins

    Returns:
        EvaluationMetrics with all values populated.
    """
    return EvaluationMetrics(
        n_predictions=len(predicted_probs),
        accuracy=accuracy(predicted_probs, actual_outcomes),
        log_loss=log_loss(predicted_probs, actual_outcomes),
        brier_score=brier_score(predicted_probs, actual_outcomes),
        calibration_error=expected_calibration_error(
            predicted_probs, actual_outcomes, n_bins
        )[0],
        calibration_bins=expected_calibration_error(
            predicted_probs, actual_outcomes, n_bins
        )[1],
    )
