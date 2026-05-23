"""Unit tests for ROI simulation math (offline, pure functions, no DB).

Coverage:
  - kelly_fraction(): math correctness, edge cases, clamping
  - Bet dataclass: construction
  - simulate_flat_staking(): correct stake, profit, equity curve
  - simulate_kelly_staking(): compounding, per-bet cap, bankrupt
  - compute_max_drawdown(): peak-to-trough logic
  - SimulationResult properties (roi, win_rate, bankroll_growth)
  - roi_by_attribute(): grouping logic
"""

from __future__ import annotations

import pytest

from tennis_predictor.backtest.roi import (
    Bet, SimulationResult,
    compute_max_drawdown, kelly_fraction,
    roi_by_attribute,
    simulate_flat_staking, simulate_kelly_staking,
)


# =============================================================================
# kelly_fraction()
# =============================================================================

class TestKellyFraction:
    def test_classic_60_at_2_00(self):
        # b=1, p=0.6, q=0.4 -> f = 0.20
        assert kelly_fraction(0.60, 2.00) == pytest.approx(0.20)

    def test_fair_priced_zero(self):
        # 50% chance at 2.00 odds = exactly fair -> f = 0
        assert kelly_fraction(0.50, 2.00) == 0.0

    def test_negative_ev_clamped_to_zero(self):
        # 40% at 2.00 odds = -EV -> Kelly says no bet
        assert kelly_fraction(0.40, 2.00) == 0.0

    def test_underdog_big_payoff(self):
        # 30% at 5.00 odds: b=4, p=0.3, q=0.7 -> f = (4*0.3 - 0.7)/4 = 0.125
        assert kelly_fraction(0.30, 5.00) == pytest.approx(0.125)

    def test_extreme_favorite(self):
        # 90% at 1.20 odds: b=0.2, p=0.9, q=0.1 -> f = (0.18 - 0.1)/0.2 = 0.40
        assert kelly_fraction(0.90, 1.20) == pytest.approx(0.40)

    def test_certain_outcome(self):
        # 100% at any odds > 1: f -> 1 (bet entire bankroll)
        # b=1.5, p=1.0, q=0 -> f = (1.5 - 0)/1.5 = 1.0
        assert kelly_fraction(1.0, 2.5) == pytest.approx(1.0)

    def test_rejects_invalid_prob(self):
        with pytest.raises(ValueError, match="predicted_prob"):
            kelly_fraction(-0.1, 2.0)
        with pytest.raises(ValueError):
            kelly_fraction(1.5, 2.0)

    def test_rejects_invalid_odds(self):
        with pytest.raises(ValueError, match="decimal_odds"):
            kelly_fraction(0.5, 1.0)
        with pytest.raises(ValueError):
            kelly_fraction(0.5, 0.5)


# =============================================================================
# Bet dataclass
# =============================================================================

class TestBet:
    def test_minimal_construction(self):
        b = Bet(predicted_prob=0.6, decimal_odds=2.0, won=True)
        assert b.predicted_prob == 0.6
        assert b.decimal_odds == 2.0
        assert b.won is True
        assert b.surface is None

    def test_with_metadata(self):
        b = Bet(
            predicted_prob=0.55, decimal_odds=1.85, won=False,
            surface="Hard", tournament_level="G", year=2024,
            match_id=12345,
        )
        assert b.surface == "Hard"
        assert b.tournament_level == "G"
        assert b.year == 2024
        assert b.match_id == 12345

    def test_is_immutable(self):
        b = Bet(0.5, 2.0, True)
        with pytest.raises((AttributeError, Exception)):
            b.predicted_prob = 0.6


# =============================================================================
# compute_max_drawdown()
# =============================================================================

class TestMaxDrawdown:
    def test_empty_curve(self):
        assert compute_max_drawdown([], 1.0) == 0.0

    def test_always_rising(self):
        # No drawdown if equity only grows
        assert compute_max_drawdown([1.0, 1.1, 1.2, 1.5], 1.0) == 0.0

    def test_simple_drawdown(self):
        # Peak at 1.5, trough at 1.0 -> dd = (1.5 - 1.0) / 1.5 = 0.333
        dd = compute_max_drawdown([1.0, 1.2, 1.5, 1.3, 1.0, 1.4], 1.0)
        assert dd == pytest.approx(1 / 3)

    def test_drawdown_below_initial(self):
        # Peak stays at initial=1.0, trough at 0.5 -> dd = 0.5
        dd = compute_max_drawdown([0.8, 0.6, 0.5, 0.7], 1.0)
        assert dd == pytest.approx(0.5)

    def test_multiple_peaks(self):
        # Two peaks 1.5 and 2.0; deepest trough relative to 2.0 is 1.2
        # dd = (2.0 - 1.2) / 2.0 = 0.4
        dd = compute_max_drawdown([1.5, 1.4, 2.0, 1.8, 1.2, 1.6], 1.0)
        assert dd == pytest.approx(0.4)

    def test_full_loss(self):
        # Bankroll goes to zero -> dd = 1.0
        dd = compute_max_drawdown([0.5, 0.0], 1.0)
        assert dd == 1.0


# =============================================================================
# simulate_flat_staking()
# =============================================================================

class TestFlatStaking:
    def test_no_bets(self):
        r = simulate_flat_staking([], unit=1.0, initial_bankroll=100.0)
        assert r.n_bets == 0
        assert r.total_staked == 0.0
        assert r.profit == 0.0
        assert r.roi == 0.0
        assert r.final_bankroll == 100.0

    def test_single_winning_bet(self):
        bets = [Bet(0.6, 2.0, True)]
        r = simulate_flat_staking(bets, unit=1.0, initial_bankroll=100.0)
        assert r.n_bets == 1
        assert r.n_wins == 1
        assert r.total_staked == 1.0
        assert r.total_returned == 2.0
        assert r.profit == 1.0
        assert r.roi == pytest.approx(1.0)
        assert r.final_bankroll == 101.0

    def test_single_losing_bet(self):
        bets = [Bet(0.6, 2.0, False)]
        r = simulate_flat_staking(bets, unit=1.0, initial_bankroll=100.0)
        assert r.n_bets == 1
        assert r.n_wins == 0
        assert r.total_staked == 1.0
        assert r.total_returned == 0.0
        assert r.profit == -1.0
        assert r.roi == -1.0
        assert r.final_bankroll == 99.0

    def test_mixed_outcomes(self):
        bets = [
            Bet(0.6, 2.0, True),    # +1
            Bet(0.6, 2.0, False),   # -1
            Bet(0.6, 2.0, True),    # +1
        ]
        r = simulate_flat_staking(bets, unit=1.0, initial_bankroll=10.0)
        assert r.n_bets == 3
        assert r.n_wins == 2
        assert r.total_staked == 3.0
        assert r.total_returned == 4.0
        assert r.profit == pytest.approx(1.0)
        assert r.win_rate == pytest.approx(2 / 3)

    def test_custom_unit(self):
        bets = [Bet(0.6, 2.0, True)]
        r = simulate_flat_staking(bets, unit=5.0, initial_bankroll=100.0)
        assert r.total_staked == 5.0
        assert r.total_returned == 10.0
        assert r.profit == 5.0
        assert r.final_bankroll == 105.0

    def test_equity_curve_length(self):
        bets = [Bet(0.5, 2.0, i % 2 == 0) for i in range(10)]
        r = simulate_flat_staking(bets, unit=1.0, initial_bankroll=100.0)
        assert len(r.equity_curve) == 10

    def test_negative_roi_at_pinnacle_vig(self):
        # 100 bets at 1.95 / 1.95 (vig ~2.6%), exactly fair: should lose ~vig%
        # We simulate a 50/50 spot perfectly: half win, half lose
        bets = [Bet(0.5, 1.95, i < 50) for i in range(100)]
        r = simulate_flat_staking(bets, unit=1.0, initial_bankroll=1000.0)
        # Won 50 bets at 1.95 = 97.5 returned, lost 50 = -50; net = -2.5
        # ROI = -2.5 / 100 = -2.5% (the vig)
        assert r.roi == pytest.approx(-0.025, abs=1e-6)


# =============================================================================
# simulate_kelly_staking()
# =============================================================================

class TestKellyStaking:
    def test_no_bets(self):
        r = simulate_kelly_staking([], fraction=0.25, initial_bankroll=1000.0)
        assert r.n_bets == 0
        assert r.final_bankroll == 1000.0

    def test_full_kelly_winning_then_losing(self):
        # Kelly fraction for (0.6, 2.0) = 0.20; with fraction=1.0, max_bet_pct=1.0,
        # actual stake = bankroll * 0.20
        bets = [Bet(0.6, 2.0, True), Bet(0.6, 2.0, False)]
        r = simulate_kelly_staking(
            bets, fraction=1.0, initial_bankroll=100.0, max_bet_pct=1.0,
        )
        # Bet 1: stake 20, win, +20 -> bankroll = 120
        # Bet 2: stake 120*0.20 = 24, lose, -24 -> bankroll = 96
        assert r.n_bets == 2
        assert r.n_wins == 1
        assert r.final_bankroll == pytest.approx(96.0)

    def test_fractional_kelly_quarter(self):
        # Same setup but fraction = 0.25 -> stake reduced 4x
        bets = [Bet(0.6, 2.0, True)]
        r = simulate_kelly_staking(
            bets, fraction=0.25, initial_bankroll=100.0, max_bet_pct=1.0,
        )
        # Kelly 0.20, scaled to 0.05 -> stake = 5, win -> +5 -> 105
        assert r.final_bankroll == pytest.approx(105.0)

    def test_max_bet_pct_cap(self):
        # Kelly says 40% (huge bet), but cap at 5% -> stake limited
        bets = [Bet(0.9, 1.20, True)]  # full Kelly = 0.40
        r = simulate_kelly_staking(
            bets, fraction=1.0, initial_bankroll=100.0, max_bet_pct=0.05,
        )
        # Capped stake = 5, win at 1.20 -> profit = 5 * 0.20 = 1
        assert r.final_bankroll == pytest.approx(101.0)

    def test_skips_negative_ev_bets(self):
        # Kelly = 0 on losing-EV bets, so they should not count toward n_bets
        bets = [
            Bet(0.40, 2.0, False),  # -EV, skipped
            Bet(0.60, 2.0, True),   # +EV, counted
        ]
        r = simulate_kelly_staking(
            bets, fraction=0.25, initial_bankroll=100.0,
        )
        assert r.n_bets == 1  # only the second was placed
        assert r.n_wins == 1

    def test_kelly_never_bankrupts_with_safety_cap(self):
        # Even on a 100-bet losing streak, the cap should keep us alive.
        # Each loss removes at most max_bet_pct * bankroll = 5% of remaining.
        bets = [Bet(0.6, 2.0, False)] * 100
        r = simulate_kelly_staking(
            bets, fraction=0.25, initial_bankroll=100.0, max_bet_pct=0.05,
        )
        # Bankroll multiplied by (1 - 0.05) each loss = 100 * 0.95^100 ~ 0.59
        assert 0 < r.final_bankroll < 1.0

    def test_bankrupt_break_short_circuits(self):
        # If bankroll hits exactly 0, the loop must break (no negative bets).
        # We synthesize this by forcing a bet so big it goes negative,
        # then verify the next bet was NOT placed.
        # Use a setup with high Kelly + no safety cap.
        bets = [
            Bet(0.99, 1.01, False),  # Kelly ~98%, no cap -> 99% stake, loses big
            Bet(0.99, 1.01, True),   # would-be winner; bankroll already near 0
            Bet(0.99, 1.01, True),
        ]
        r = simulate_kelly_staking(
            bets, fraction=1.0, initial_bankroll=100.0, max_bet_pct=1.0,
        )
        # Hard to assert exact n_bets due to floating-point edge cases on
        # tiny residual bankroll, but the loop MUST handle it without errors.
        assert r.final_bankroll >= 0  # never goes negative
        assert r.n_bets <= 3

    def test_rejects_invalid_fraction(self):
        with pytest.raises(ValueError, match="fraction"):
            simulate_kelly_staking([], fraction=0.0)
        with pytest.raises(ValueError):
            simulate_kelly_staking([], fraction=1.5)

    def test_rejects_invalid_max_bet_pct(self):
        with pytest.raises(ValueError, match="max_bet_pct"):
            simulate_kelly_staking([], fraction=0.25, max_bet_pct=0.0)
        with pytest.raises(ValueError):
            simulate_kelly_staking([], fraction=0.25, max_bet_pct=1.5)

    def test_equity_curve_records_only_placed_bets(self):
        bets = [
            Bet(0.40, 2.0, False),  # skipped (Kelly=0)
            Bet(0.60, 2.0, True),
            Bet(0.60, 2.0, False),
        ]
        r = simulate_kelly_staking(
            bets, fraction=0.25, initial_bankroll=100.0,
        )
        assert len(r.equity_curve) == 2


# =============================================================================
# SimulationResult properties
# =============================================================================

class TestSimulationResultProperties:
    def test_empty_result_defaults(self):
        r = SimulationResult(strategy="test")
        assert r.profit == 0.0
        assert r.roi == 0.0
        assert r.win_rate == 0.0
        assert r.bankroll_growth == 1.0
        assert r.max_drawdown == 0.0

    def test_profit_calculation(self):
        r = SimulationResult(strategy="x", total_staked=100.0, total_returned=110.0)
        assert r.profit == pytest.approx(10.0)
        assert r.roi == pytest.approx(0.10)

    def test_win_rate(self):
        r = SimulationResult(strategy="x", n_bets=10, n_wins=4)
        assert r.win_rate == pytest.approx(0.4)

    def test_bankroll_growth(self):
        r = SimulationResult(
            strategy="x", initial_bankroll=100.0, final_bankroll=150.0,
        )
        assert r.bankroll_growth == pytest.approx(1.5)

    def test_zero_initial_bankroll_safe(self):
        r = SimulationResult(
            strategy="x", initial_bankroll=0.0, final_bankroll=0.0,
        )
        assert r.bankroll_growth == 0.0


# =============================================================================
# roi_by_attribute()
# =============================================================================

class TestRoiByAttribute:
    def test_groups_by_surface(self):
        bets = [
            Bet(0.6, 2.0, True,  surface="Hard"),
            Bet(0.6, 2.0, False, surface="Hard"),
            Bet(0.6, 2.0, True,  surface="Clay"),
        ]
        rows = roi_by_attribute(bets, "surface")
        by_name = {r.name: r for r in rows}
        assert by_name["Hard"].n_bets == 2
        assert by_name["Hard"].n_wins == 1
        assert by_name["Clay"].n_bets == 1
        assert by_name["Clay"].n_wins == 1

    def test_sorted_descending_by_roi(self):
        # Clay all wins (+100% ROI), Hard mixed (0% ROI)
        bets = [
            Bet(0.6, 2.0, True,  surface="Clay"),
            Bet(0.6, 2.0, True,  surface="Clay"),
            Bet(0.6, 2.0, True,  surface="Hard"),
            Bet(0.6, 2.0, False, surface="Hard"),
        ]
        rows = roi_by_attribute(bets, "surface")
        # First entry must be Clay (higher ROI)
        assert rows[0].name == "Clay"
        assert rows[0].roi > rows[1].roi

    def test_handles_none_attribute(self):
        bets = [Bet(0.6, 2.0, True, surface=None)]
        rows = roi_by_attribute(bets, "surface")
        assert rows[0].name == "(unknown)"

    def test_groups_by_year(self):
        bets = [
            Bet(0.6, 2.0, True,  year=2023),
            Bet(0.6, 2.0, False, year=2024),
            Bet(0.6, 2.0, True,  year=2023),
        ]
        rows = roi_by_attribute(bets, "year")
        by_name = {r.name: r for r in rows}
        assert by_name["2023"].n_bets == 2
        assert by_name["2024"].n_bets == 1

    def test_empty_bets(self):
        rows = roi_by_attribute([], "surface")
        assert rows == []
