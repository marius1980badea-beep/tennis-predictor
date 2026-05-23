"""ROI simulation for staking strategies on identified value bets.

This module is pure functions: no DB access, no external state. All logic
here is unit-testable offline.

Three staking strategies are supported:

  - flat_staking: 1 unit per bet, regardless of edge or bankroll
  - kelly: bankroll * kelly_fraction * fraction (fractional Kelly)
  - kelly_fraction defaults: 1/4 (conservative), 1/8 (very conservative)

For each strategy we compute:
  - Final ROI in % (profit / total_staked)
  - Equity curve (bankroll over time, normalised to start=1.0)
  - Max drawdown (largest peak-to-trough fall in equity)
  - Number of bets, win rate, average odds

Reference:
  - Kelly criterion: f* = (b * p - q) / b, where
      b = decimal_odds - 1  (net odds)
      p = predicted probability of win
      q = 1 - p
    Positive f* means a +EV bet at face value; we then SCALE this by
    a safety fraction (1/4 or 1/8) to reduce variance.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Kelly math
# ---------------------------------------------------------------------------

def kelly_fraction(predicted_prob: float, decimal_odds: float) -> float:
    """Classic Kelly fraction.

    Formula: f* = (b*p - q) / b
      where b = decimal_odds - 1 (net odds)
            p = predicted_prob
            q = 1 - p

    Returns 0.0 if the bet would be -EV (i.e. Kelly says "don't bet").

    >>> # 60% chance at 2.00 odds: b=1, p=0.6, q=0.4
    >>> # f* = (1*0.6 - 0.4) / 1 = 0.20 = bet 20% of bankroll
    >>> abs(kelly_fraction(0.60, 2.00) - 0.20) < 1e-9
    True

    >>> # Fair-priced bet: zero Kelly
    >>> kelly_fraction(0.50, 2.00)
    0.0

    >>> # Negative-EV bet: clamped to 0
    >>> kelly_fraction(0.40, 2.00)
    0.0
    """
    if not (0 <= predicted_prob <= 1):
        raise ValueError(f"predicted_prob out of [0,1]: {predicted_prob}")
    if decimal_odds <= 1.0:
        raise ValueError(f"decimal_odds must be > 1.0, got {decimal_odds}")

    b = decimal_odds - 1.0
    p = predicted_prob
    q = 1.0 - p
    f = (b * p - q) / b
    return max(f, 0.0)


# ---------------------------------------------------------------------------
# Bet representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Bet:
    """One placed bet, with everything needed for simulation."""
    predicted_prob: float       # model's probability for THIS side
    decimal_odds: float         # bookmaker odds for THIS side
    won: bool                   # outcome: did this side win?
    # Free-form metadata for grouping in reports (surface, year, etc.)
    surface: str | None = None
    tournament_level: str | None = None
    year: int | None = None
    match_id: int | None = None


# ---------------------------------------------------------------------------
# Simulation results
# ---------------------------------------------------------------------------

@dataclass
class SimulationResult:
    """Aggregate output from a single staking strategy run."""

    strategy: str
    n_bets: int = 0
    n_wins: int = 0
    total_staked: float = 0.0
    total_returned: float = 0.0  # gross winnings (including stake on wins)

    # Bankroll trajectory: list of bankroll values AFTER each bet (Kelly only).
    # For flat staking, equity_curve tracks (cumulative_profit + initial_bankroll).
    equity_curve: list[float] = field(default_factory=list)

    # Final bankroll (last value of equity_curve, or initial if no bets).
    final_bankroll: float = 1.0
    initial_bankroll: float = 1.0

    @property
    def profit(self) -> float:
        """Net profit (returned - staked)."""
        return self.total_returned - self.total_staked

    @property
    def roi(self) -> float:
        """Return on Investment: profit / total_staked.

        Returns 0.0 if no bets were placed.
        """
        if self.total_staked == 0:
            return 0.0
        return self.profit / self.total_staked

    @property
    def win_rate(self) -> float:
        """Fraction of bets that won."""
        if self.n_bets == 0:
            return 0.0
        return self.n_wins / self.n_bets

    @property
    def bankroll_growth(self) -> float:
        """Final / initial bankroll (1.0 = breakeven, 2.0 = doubled, 0.5 = lost half)."""
        if self.initial_bankroll == 0:
            return 0.0
        return self.final_bankroll / self.initial_bankroll

    @property
    def max_drawdown(self) -> float:
        """Largest peak-to-trough decline in equity_curve, as a fraction.

        E.g. 0.30 means at some point the bankroll fell 30% from a previous high.
        """
        return compute_max_drawdown(self.equity_curve, self.initial_bankroll)


def compute_max_drawdown(equity_curve: list[float], initial: float) -> float:
    """Max peak-to-trough drawdown in an equity curve.

    >>> compute_max_drawdown([1.0, 1.2, 1.5, 1.3, 1.0, 1.4], 1.0)
    0.3333333333333333
    >>> compute_max_drawdown([], 1.0)
    0.0
    >>> compute_max_drawdown([1.0, 1.1, 1.2], 1.0)  # always rising
    0.0
    """
    if not equity_curve:
        return 0.0

    peak = initial
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        if peak > 0:
            dd = (peak - value) / peak
            if dd > max_dd:
                max_dd = dd
    return max_dd


# ---------------------------------------------------------------------------
# Flat staking
# ---------------------------------------------------------------------------

def simulate_flat_staking(
    bets: list[Bet],
    unit: float = 1.0,
    initial_bankroll: float = 1000.0,
) -> SimulationResult:
    """Bet `unit` on every bet in the list.

    Equity curve tracks running (initial_bankroll + cumulative_profit).
    This is independent of bankroll size (we never resize stake).

    >>> bets = [
    ...     Bet(0.6, 2.0, True),    # win +1
    ...     Bet(0.6, 2.0, False),   # lose -1
    ...     Bet(0.6, 2.0, True),    # win +1
    ... ]
    >>> r = simulate_flat_staking(bets, unit=1.0, initial_bankroll=10.0)
    >>> r.n_bets, r.n_wins
    (3, 2)
    >>> r.total_staked, r.total_returned
    (3.0, 4.0)
    >>> abs(r.profit - 1.0) < 1e-9
    True
    """
    result = SimulationResult(
        strategy=f"flat_{unit}u",
        initial_bankroll=initial_bankroll,
        final_bankroll=initial_bankroll,
    )

    bankroll = initial_bankroll
    for bet in bets:
        stake = unit
        result.total_staked += stake
        if bet.won:
            # Decimal odds include the stake in the payout
            gross_return = stake * bet.decimal_odds
            result.total_returned += gross_return
            bankroll += (gross_return - stake)
            result.n_wins += 1
        else:
            bankroll -= stake
        result.equity_curve.append(bankroll)
        result.n_bets += 1

    result.final_bankroll = bankroll
    return result


# ---------------------------------------------------------------------------
# Fractional Kelly staking
# ---------------------------------------------------------------------------

def simulate_kelly_staking(
    bets: list[Bet],
    fraction: float = 0.25,
    initial_bankroll: float = 1000.0,
    max_bet_pct: float = 0.05,
) -> SimulationResult:
    """Fractional Kelly with a per-bet cap.

    Stake = bankroll * kelly_fraction(p, odds) * fraction
    Capped at `max_bet_pct` of current bankroll (safety against blow-ups).

    Bankroll compounds: each win/loss scales future stakes.

    Args:
        bets: ordered list of Bet (chronological order matters for compounding)
        fraction: Kelly fraction multiplier (0.25 = quarter Kelly, 0.125 = eighth)
        initial_bankroll: starting bankroll
        max_bet_pct: cap on any single bet, as fraction of current bankroll
                     (default 5% — prevents catastrophic single-bet loss)

    >>> bets = [Bet(0.6, 2.0, True), Bet(0.6, 2.0, False)]
    >>> r = simulate_kelly_staking(bets, fraction=1.0, initial_bankroll=100.0,
    ...                             max_bet_pct=1.0)
    >>> # First bet: full Kelly = 0.20; stake = 100*0.20 = 20, wins to 120
    >>> # Second bet: 120 * 0.20 = 24, loses, bankroll = 96
    >>> r.n_bets, r.n_wins
    (2, 1)
    >>> abs(r.final_bankroll - 96.0) < 1e-6
    True
    """
    if not (0 < fraction <= 1):
        raise ValueError(f"fraction must be in (0, 1]: {fraction}")
    if not (0 < max_bet_pct <= 1):
        raise ValueError(f"max_bet_pct must be in (0, 1]: {max_bet_pct}")

    strategy_label = (
        f"kelly_1/{int(round(1.0 / fraction))}"
        if fraction not in (0, 1)
        else ("kelly_full" if fraction == 1 else "kelly_0")
    )
    result = SimulationResult(
        strategy=strategy_label,
        initial_bankroll=initial_bankroll,
        final_bankroll=initial_bankroll,
    )

    bankroll = initial_bankroll
    for bet in bets:
        if bankroll <= 0:
            # Bankrupt; stop placing bets but keep iterating to record bets skipped
            break

        full_kelly = kelly_fraction(bet.predicted_prob, bet.decimal_odds)
        scaled = full_kelly * fraction
        scaled = min(scaled, max_bet_pct)  # cap per-bet exposure

        stake = bankroll * scaled
        if stake <= 0:
            # Kelly says don't bet; skip recording it (no contribution to ROI)
            continue

        result.total_staked += stake
        if bet.won:
            gross_return = stake * bet.decimal_odds
            result.total_returned += gross_return
            bankroll += (gross_return - stake)
            result.n_wins += 1
        else:
            bankroll -= stake

        result.equity_curve.append(bankroll)
        result.n_bets += 1

    result.final_bankroll = bankroll
    return result


# ---------------------------------------------------------------------------
# Aggregated reporting helpers
# ---------------------------------------------------------------------------

@dataclass
class SubsliceROI:
    """ROI numbers for one subslice (e.g. one surface, one year)."""
    name: str
    n_bets: int
    n_wins: int
    total_staked: float
    profit: float

    @property
    def roi(self) -> float:
        return self.profit / self.total_staked if self.total_staked else 0.0

    @property
    def win_rate(self) -> float:
        return self.n_wins / self.n_bets if self.n_bets else 0.0


def roi_by_attribute(
    bets: list[Bet],
    attribute: str,
    unit: float = 1.0,
) -> list[SubsliceROI]:
    """Group flat-staking ROI by Bet attribute (surface, year, tournament_level).

    Returns a list sorted descending by ROI.

    >>> bets = [
    ...     Bet(0.6, 2.0, True,  surface="Hard"),
    ...     Bet(0.6, 2.0, False, surface="Hard"),
    ...     Bet(0.6, 2.0, True,  surface="Clay"),
    ... ]
    >>> rows = roi_by_attribute(bets, "surface")
    >>> {r.name: r.n_bets for r in rows} == {"Hard": 2, "Clay": 1}
    True
    """
    groups: dict[object, list[Bet]] = {}
    for b in bets:
        key = getattr(b, attribute, None)
        groups.setdefault(key, []).append(b)

    rows = []
    for key, group_bets in groups.items():
        sub_result = simulate_flat_staking(group_bets, unit=unit)
        rows.append(SubsliceROI(
            name=str(key) if key is not None else "(unknown)",
            n_bets=sub_result.n_bets,
            n_wins=sub_result.n_wins,
            total_staked=sub_result.total_staked,
            profit=sub_result.profit,
        ))
    rows.sort(key=lambda r: r.roi, reverse=True)
    return rows
