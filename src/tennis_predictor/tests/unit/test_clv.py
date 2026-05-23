"""Unit tests for CLV math + value-bet detection.

All tests run offline against pure functions, no DB needed.
"""

from __future__ import annotations

import pytest

from tennis_predictor.backtest.clv import (
    DEFAULT_MIN_EDGE,
    DEFAULT_MIN_ODDS,
    DEFAULT_MIN_PROB,
    CLVStats,
    ValueBetCriteria,
    compute_clv,
    compute_edge,
    implied_prob,
    is_value_bet,
    summarise_clv,
)


# =============================================================================
# implied_prob()
# =============================================================================

class TestImpliedProb:
    def test_even_odds(self):
        assert implied_prob(2.0) == 0.5

    def test_favorite(self):
        assert implied_prob(1.5) == pytest.approx(2 / 3)

    def test_underdog(self):
        assert implied_prob(4.0) == 0.25

    def test_extreme_favorite(self):
        assert implied_prob(1.05) == pytest.approx(0.9524, rel=1e-3)

    def test_rejects_invalid_odds(self):
        with pytest.raises(ValueError, match="must be > 1.0"):
            implied_prob(1.0)
        with pytest.raises(ValueError):
            implied_prob(0.5)
        with pytest.raises(ValueError):
            implied_prob(0.0)


# =============================================================================
# compute_clv()
# =============================================================================

class TestComputeClv:
    def test_perfect_agreement_zero_clv(self):
        # Model and Pinnacle agree exactly
        assert compute_clv(0.55, 0.55) == 0.0

    def test_positive_clv_we_value_higher(self):
        # Model says 60%, Pinnacle says 55% -> +9.09%
        clv = compute_clv(0.60, 0.55)
        assert clv == pytest.approx(0.0909, rel=1e-3)

    def test_negative_clv_we_value_lower(self):
        # Model says 50%, Pinnacle says 55%
        clv = compute_clv(0.50, 0.55)
        assert clv < 0
        assert clv == pytest.approx(-0.0909, rel=1e-3)

    def test_large_positive_clv(self):
        # Heavy disagreement: 80% vs 50%
        clv = compute_clv(0.80, 0.50)
        assert clv == pytest.approx(0.60)

    def test_large_negative_clv(self):
        # Model thinks 30% when Pinnacle says 70%
        clv = compute_clv(0.30, 0.70)
        assert clv == pytest.approx(-0.5714, rel=1e-3)

    def test_zero_predicted_prob(self):
        # Model 100% sure the other side wins -> CLV = -1
        clv = compute_clv(0.0, 0.50)
        assert clv == -1.0

    def test_rejects_invalid_predicted(self):
        with pytest.raises(ValueError, match="predicted_prob"):
            compute_clv(-0.1, 0.5)
        with pytest.raises(ValueError):
            compute_clv(1.5, 0.5)

    def test_rejects_invalid_pinnacle(self):
        with pytest.raises(ValueError, match="pinnacle_implied"):
            compute_clv(0.5, 0.0)
        with pytest.raises(ValueError):
            compute_clv(0.5, -0.1)
        with pytest.raises(ValueError):
            compute_clv(0.5, 1.5)


# =============================================================================
# compute_edge()
# =============================================================================

class TestComputeEdge:
    def test_zero_edge_fair_market(self):
        # 50% chance at 2.00 odds = exactly fair
        assert compute_edge(0.50, 2.0) == 0.0

    def test_positive_edge_underdog(self):
        # Model thinks 50% at 2.50 odds (implied 40%)
        edge = compute_edge(0.50, 2.50)
        assert edge == pytest.approx(0.25)

    def test_negative_edge(self):
        # Model thinks 40% but odds are 2.00 (implied 50%)
        edge = compute_edge(0.40, 2.00)
        assert edge == pytest.approx(-0.20)

    def test_classic_value_situation(self):
        # Pinnacle's "fair" line on a 60/40 spot: 1.667 / 2.500
        # If we think it's 65/35 instead, our edge on the favorite:
        # 0.65 * 1.667 - 1 = +8.36%
        edge = compute_edge(0.65, 1.667)
        assert edge == pytest.approx(0.0836, rel=1e-2)

    def test_rejects_invalid_odds(self):
        with pytest.raises(ValueError, match="decimal_odds"):
            compute_edge(0.5, 1.0)
        with pytest.raises(ValueError):
            compute_edge(0.5, 0.99)

    def test_rejects_invalid_prob(self):
        with pytest.raises(ValueError, match="predicted_prob"):
            compute_edge(-0.1, 2.0)
        with pytest.raises(ValueError):
            compute_edge(1.1, 2.0)


# =============================================================================
# ValueBetCriteria
# =============================================================================

class TestValueBetCriteria:
    def test_defaults_match_project_blueprint(self):
        c = ValueBetCriteria()
        assert c.min_edge == 0.05
        assert c.min_prob == 0.55
        assert c.min_odds == 1.60

    def test_custom_values(self):
        c = ValueBetCriteria(min_edge=0.07, min_prob=0.60, min_odds=1.80)
        assert c.min_edge == 0.07
        assert c.min_prob == 0.60
        assert c.min_odds == 1.80

    def test_rejects_invalid_edge(self):
        with pytest.raises(ValueError):
            ValueBetCriteria(min_edge=-0.01)
        with pytest.raises(ValueError):
            ValueBetCriteria(min_edge=1.0)

    def test_rejects_invalid_prob(self):
        with pytest.raises(ValueError):
            ValueBetCriteria(min_prob=0.0)
        with pytest.raises(ValueError):
            ValueBetCriteria(min_prob=1.1)

    def test_rejects_invalid_odds(self):
        with pytest.raises(ValueError):
            ValueBetCriteria(min_odds=1.0)

    def test_is_immutable(self):
        c = ValueBetCriteria()
        with pytest.raises((AttributeError, Exception)):
            c.min_edge = 0.10  # frozen


# =============================================================================
# is_value_bet()
# =============================================================================

class TestIsValueBet:
    def test_textbook_value_bet(self):
        # Model 60% on a market at 1.825 implies 54.8%
        # Edge = 0.60 * 1.825 - 1 = +9.5%, well above 5%
        # Prob 60% >= 55%, odds 1.825 >= 1.60
        assert is_value_bet(0.60, 1.825) is True

    def test_edge_just_below_threshold(self):
        # 0.55 * 1.85 - 1 = 0.0175 ~= 1.75%, below 5%
        assert is_value_bet(0.55, 1.85) is False

    def test_prob_below_floor(self):
        # 50% prob fails the prob floor even if edge is positive
        # 0.50 * 2.50 = 1.25, edge = 0.25 (large), but 0.50 < 0.55 floor
        assert is_value_bet(0.50, 2.50) is False

    def test_odds_below_floor(self):
        # 85% at 1.30 odds = edge 10.5% large, prob 85% high, but odds 1.30 < 1.60
        assert is_value_bet(0.85, 1.30) is False

    def test_custom_criteria_passes(self):
        # Loosen min_edge to 1% - the 1.75% case becomes a value bet
        c = ValueBetCriteria(min_edge=0.01)
        assert is_value_bet(0.55, 1.85, c) is True

    def test_custom_criteria_tightens(self):
        # Tighten min_edge to 12% - typical 9.5% bet no longer qualifies
        c = ValueBetCriteria(min_edge=0.12)
        assert is_value_bet(0.60, 1.825, c) is False

    def test_all_three_must_pass(self):
        # Edge OK, prob below floor by tiny amount
        # 0.549 * 1.85 = 1.0157, edge 1.57%
        assert is_value_bet(0.549, 1.85) is False  # prob 0.549 < 0.55


# =============================================================================
# summarise_clv()
# =============================================================================

class TestSummariseClv:
    def test_empty_returns_zero_stats(self):
        s = summarise_clv([])
        assert s.n_predictions == 0
        assert s.mean_clv == 0.0

    def test_simple_summary(self):
        s = summarise_clv([0.05, 0.03, -0.01, 0.10, 0.02])
        assert s.n_predictions == 5
        assert s.mean_clv == pytest.approx(0.038)
        assert s.median_clv == pytest.approx(0.03)
        assert s.pct_positive_clv == 80.0  # 4 of 5
        assert s.pct_significantly_positive == 60.0  # 3 of 5 above 2%

    def test_median_even_count(self):
        # 4 values: median = avg of middle two
        s = summarise_clv([0.01, 0.03, 0.05, 0.07])
        assert s.median_clv == pytest.approx(0.04)  # (0.03 + 0.05) / 2

    def test_includes_value_bet_stats(self):
        clvs = [0.05, 0.03, -0.01, 0.10, 0.02]
        flags = [True, False, False, True, False]
        s = summarise_clv(clvs, flags)
        assert s.n_value_bets == 2
        assert s.value_bet_rate == 40.0  # 2 of 5

    def test_all_negative_clvs(self):
        s = summarise_clv([-0.05, -0.10, -0.02])
        assert s.pct_positive_clv == 0.0
        assert s.mean_clv < 0

    def test_all_positive_clvs(self):
        s = summarise_clv([0.05, 0.10, 0.02])
        assert s.pct_positive_clv == 100.0
        assert s.pct_significantly_positive > 0


# =============================================================================
# Cross-function consistency
# =============================================================================

class TestConsistency:
    """CLV and edge should be related when implied_prob is derived from odds."""

    def test_zero_clv_implies_zero_edge_minus_vig(self):
        # If model agrees with Pinnacle, then edge = -vig (you pay the margin)
        # Model 0.50, Pinnacle book at 1.90/1.90 (vig ~5.3%)
        # edge = 0.50 * 1.90 - 1 = -0.05
        edge = compute_edge(0.50, 1.90)
        assert edge < 0   # expected: lose the vig

    def test_positive_clv_implies_positive_edge_at_market_odds(self):
        # If model has CLV > vig, then edge should be positive
        # Pinnacle: 1.90, implied 0.5263
        # Model: 0.60 (CLV +14% relative)
        # Edge: 0.60 * 1.90 - 1 = +0.14 (positive)
        clv = compute_clv(0.60, 1 / 1.90)
        edge = compute_edge(0.60, 1.90)
        assert clv > 0
        assert edge > 0
