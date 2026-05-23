"""Closing Line Value (CLV) and value-bet detection math.

This module is pure functions: no DB access, no external state. All logic
here is unit-testable offline.

Terminology recap:
    decimal_odds = the European-style decimal price (1.85, 2.50, etc.)
    implied_prob = 1 / decimal_odds
    vig          = sum of implied_probs across all outcomes - 1
                   (i.e. the bookmaker's margin)
    predicted_prob = the model's probability for an outcome (0..1)

Definitions used in this project:

  CLV = (predicted_prob - pinnacle_implied_prob) / pinnacle_implied_prob
        ^ relative gap: how much higher (positive) or lower (negative) our
          probability is vs the market's

  Edge = predicted_prob * decimal_odds - 1
       = expected profit (in units) per unit staked on a fair-priced bet
         at the given odds. Positive edge = +EV bet at face value.

  Value bet criteria (configurable):
    - edge >= min_edge          (typical: 0.05 = 5%)
    - predicted_prob >= min_prob (typical: 0.55 — avoids underdog noise)
    - decimal_odds >= min_odds   (typical: 1.60 — avoids heavy-favorite roll)

The thresholds are CONSERVATIVE on purpose. Sharp sports bettors target
edge values of 2-3% on Pinnacle closing lines; we use 5% here as a safety
margin against model overconfidence.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Default thresholds (mirror the project blueprint)
# ---------------------------------------------------------------------------

DEFAULT_MIN_EDGE = 0.05      # 5% edge required to consider a bet
DEFAULT_MIN_PROB = 0.55      # 55% predicted probability minimum
DEFAULT_MIN_ODDS = 1.60      # 1.60 minimum decimal odds (avoids -EV grind)


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def implied_prob(decimal_odds: float) -> float:
    """Convert decimal odds to implied probability.

    >>> implied_prob(2.00)
    0.5
    >>> implied_prob(1.50)
    0.6666666666666666
    """
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal_odds must be > 1.0, got {decimal_odds}")
    return 1.0 / decimal_odds


def compute_clv(
    predicted_prob: float,
    pinnacle_implied_prob: float,
) -> float:
    """Relative CLV: positive means we valued the outcome higher than the line.

    A bettor whose model consistently has positive CLV vs Pinnacle is
    statistically beating the closing line — the gold-standard signal of
    long-run edge in sports betting.

    Formula: (predicted_prob - pinnacle_implied_prob) / pinnacle_implied_prob

    >>> # Model says 60%, Pinnacle says 55% (decimal 1.818)
    >>> # CLV = (0.60 - 0.55) / 0.55 = +9.09% — strong positive
    >>> abs(compute_clv(0.60, 0.55) - 0.09090909) < 1e-6
    True

    >>> # Model agrees with Pinnacle exactly
    >>> compute_clv(0.55, 0.55)
    0.0

    >>> # Model says 50%, Pinnacle says 55% — we're below market
    >>> compute_clv(0.50, 0.55) < 0
    True
    """
    if not (0 <= predicted_prob <= 1):
        raise ValueError(f"predicted_prob out of [0,1]: {predicted_prob}")
    if not (0 < pinnacle_implied_prob <= 1):
        raise ValueError(f"pinnacle_implied_prob out of (0,1]: {pinnacle_implied_prob}")
    return (predicted_prob - pinnacle_implied_prob) / pinnacle_implied_prob


def compute_edge(predicted_prob: float, decimal_odds: float) -> float:
    """Expected value per unit staked at the given odds.

    Formula: predicted_prob * decimal_odds - 1

    Positive edge means a +EV bet at face value (before fees/vig
    considerations).

    >>> # Coin flip at 2.50 odds when we think it's 50/50: edge = 0.25
    >>> abs(compute_edge(0.50, 2.50) - 0.25) < 1e-9
    True

    >>> # Fair-priced bet: zero edge
    >>> compute_edge(0.50, 2.00)
    0.0

    >>> # Negative edge: we think 40% but odds imply 50%
    >>> compute_edge(0.40, 2.00) < 0
    True
    """
    if not (0 <= predicted_prob <= 1):
        raise ValueError(f"predicted_prob out of [0,1]: {predicted_prob}")
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal_odds must be > 1.0, got {decimal_odds}")
    return predicted_prob * decimal_odds - 1.0


# ---------------------------------------------------------------------------
# Value bet detection
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ValueBetCriteria:
    """Bundle of thresholds for value-bet classification."""
    min_edge: float = DEFAULT_MIN_EDGE
    min_prob: float = DEFAULT_MIN_PROB
    min_odds: float = DEFAULT_MIN_ODDS

    def __post_init__(self) -> None:
        if not (0 <= self.min_edge < 1):
            raise ValueError(f"min_edge must be in [0, 1): {self.min_edge}")
        if not (0 < self.min_prob <= 1):
            raise ValueError(f"min_prob must be in (0, 1]: {self.min_prob}")
        if self.min_odds <= 1.0:
            raise ValueError(f"min_odds must be > 1.0: {self.min_odds}")


def is_value_bet(
    predicted_prob: float,
    decimal_odds: float,
    criteria: ValueBetCriteria | None = None,
) -> bool:
    """Return True iff this (prob, odds) pair qualifies as a value bet.

    Three filters, all must pass:
      1. edge >= min_edge
      2. predicted_prob >= min_prob
      3. decimal_odds >= min_odds

    >>> # Edge 9.5%, prob 60%, odds 1.825 -- all pass at defaults
    >>> is_value_bet(0.60, 1.825)
    True

    >>> # Same prob/odds but too low (4%)
    >>> is_value_bet(0.55, 1.85, ValueBetCriteria(min_edge=0.05))
    False

    >>> # Edge OK but prob below 55% floor
    >>> is_value_bet(0.50, 2.50)
    False

    >>> # Edge OK, prob OK, but heavy favorite at 1.30 below min_odds
    >>> is_value_bet(0.85, 1.30)
    False
    """
    crit = criteria or ValueBetCriteria()
    edge = compute_edge(predicted_prob, decimal_odds)
    return (
        edge >= crit.min_edge
        and predicted_prob >= crit.min_prob
        and decimal_odds >= crit.min_odds
    )


# ---------------------------------------------------------------------------
# CLV summary stats (used by analysis CLI)
# ---------------------------------------------------------------------------

@dataclass
class CLVStats:
    """Aggregate CLV statistics over a set of predictions."""
    n_predictions: int = 0
    mean_clv: float = 0.0
    median_clv: float = 0.0
    pct_positive_clv: float = 0.0    # % of predictions with CLV > 0
    pct_significantly_positive: float = 0.0   # % with CLV > 0.02 (2%)
    n_value_bets: int = 0
    value_bet_rate: float = 0.0      # n_value_bets / n_predictions


def summarise_clv(
    clvs: list[float],
    value_bet_flags: list[bool] | None = None,
) -> CLVStats:
    """Compute aggregate CLV stats from per-prediction CLV values.

    >>> stats = summarise_clv([0.05, 0.03, -0.01, 0.10, 0.02])
    >>> stats.n_predictions
    5
    >>> abs(stats.mean_clv - 0.038) < 1e-9
    True
    >>> stats.pct_positive_clv == 80.0  # 4 of 5 positive
    True
    """
    if not clvs:
        return CLVStats()

    n = len(clvs)
    sorted_clvs = sorted(clvs)
    mean_clv = sum(clvs) / n
    median_clv = sorted_clvs[n // 2] if n % 2 else (
        (sorted_clvs[n // 2 - 1] + sorted_clvs[n // 2]) / 2
    )
    pct_pos = 100.0 * sum(1 for c in clvs if c > 0) / n
    pct_sig_pos = 100.0 * sum(1 for c in clvs if c > 0.02) / n

    n_vb = sum(value_bet_flags) if value_bet_flags else 0
    vb_rate = 100.0 * n_vb / n if value_bet_flags else 0.0

    return CLVStats(
        n_predictions=n,
        mean_clv=mean_clv,
        median_clv=median_clv,
        pct_positive_clv=pct_pos,
        pct_significantly_positive=pct_sig_pos,
        n_value_bets=n_vb,
        value_bet_rate=vb_rate,
    )
