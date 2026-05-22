"""Surface-Adjusted Elo v2 — improved baseline.

Three improvements over v1 (elo.py):

1. **Mean reversion**: ratings far from the population mean (1750 for tennis)
   are pulled gently back toward it each match. This prevents the rating
   spirals that occur when a hot player keeps beating weaker opposition.
   Mathematical formulation: after each rating update, apply
       R_new = mean + (1 - α) * (R_new - mean)
   where α is small (e.g., 0.005). At equilibrium, K=24 update on a 50/50
   match still moves you, but extreme ratings can't run away indefinitely.

2. **Time decay** between matches: a player inactive for 3+ months has their
   rating pulled slightly toward the mean for each subsequent active week.
   Captures: layoffs reduce match sharpness, age-related decline, etc.

3. **Conservative K for big spreads**: when the rating gap is already large
   (>250 points), reduce K-factor. Rationale: the model is already saying
   "this should be a one-sided match", so we shouldn't update aggressively
   regardless of result — variance dominates in lopsided matchups.

Backward compatibility: keeps the same PlayerEloState shape so the manager,
backtest, and predict CLIs all keep working without changes.

References:
- Lakatos (2018): "Modelling tennis match outcomes using Elo and ML methods"
  - Discussion of mean reversion in elo
- Glickman & Stern (1998): "A state-space model for NHL hockey scores"
  - Time-decay framework adapted for sports
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Literal

Surface = Literal["Hard", "Clay", "Grass", "Carpet", "Overall"]


@dataclass
class EloConfigV2:
    """Configuration for v2 Elo with mean reversion, time decay, K-spread adjustment.

    All v1 parameters preserved; v2-specific are at the bottom.
    """

    # ============ v1 parameters (unchanged) ============
    initial_rating: float = 1500.0

    k_initial: float = 40.0
    k_mid: float = 24.0
    k_stable: float = 16.0
    k_threshold_low: int = 20
    k_threshold_high: int = 50

    level_multipliers: dict[str, float] = field(default_factory=lambda: {
        "G": 1.1, "M": 1.05, "F": 1.05, "A": 1.0, "D": 1.0,
        "PM": 1.05, "I": 1.0, "O": 1.0, "C": 0.85, "S": 0.7,
    })

    mov_min_multiplier: float = 0.8
    mov_max_multiplier: float = 1.2

    surface_prior_weight: float = 0.3
    surface_prior_threshold: int = 10

    retirement_k_multiplier: float = 0.5

    # ============ v2 new parameters ============

    # Mean reversion: ratings pulled toward this anchor each match
    population_mean: float = 1750.0
    # Strength of pull per match (0.005 = 0.5% pull, very gentle)
    mean_reversion_strength: float = 0.005

    # Time decay: regress toward mean when player is inactive
    # Half-life in days for "freshness" of a rating
    inactivity_threshold_days: int = 90  # No decay before this
    inactivity_decay_per_week: float = 0.003  # 0.3% pull per week of inactivity

    # K reduction for large spreads
    large_spread_threshold: float = 250.0  # Points
    large_spread_k_multiplier: float = 0.7  # Reduce K by 30% when gap is huge


@dataclass
class PlayerEloStateV2:
    """In-memory state with last activity tracking per surface.

    Compatible interface with v1 PlayerEloState but adds per-surface
    last_match_date dict for time decay calculations.
    """

    player_id: str
    ratings: dict[Surface, float] = field(default_factory=dict)
    matches_played: dict[Surface, int] = field(default_factory=dict)

    # New: per-surface last match date (for time decay)
    last_match_date_per_surface: dict[Surface, date] = field(default_factory=dict)

    # Kept for compatibility with v1 interface
    last_match_date: date | None = None

    def get_rating(self, surface: Surface, config: EloConfigV2) -> float:
        return self.ratings.get(surface, config.initial_rating)

    def get_matches_played(self, surface: Surface) -> int:
        return self.matches_played.get(surface, 0)

    def get_effective_rating(self, surface: Surface, config: EloConfigV2) -> float:
        """Surface rating blended with Overall for new-to-surface players."""
        surface_matches = self.get_matches_played(surface)
        surface_rating = self.get_rating(surface, config)

        if surface == "Overall" or surface_matches >= config.surface_prior_threshold:
            return surface_rating

        overall_rating = self.get_rating("Overall", config)
        weight_surface = surface_matches / config.surface_prior_threshold
        weight_overall = config.surface_prior_weight * (1 - weight_surface)
        total_weight = weight_surface + weight_overall

        if total_weight == 0:
            return overall_rating

        return (
            surface_rating * weight_surface + overall_rating * weight_overall
        ) / total_weight


# ============================================================================
# Math primitives (same as v1 since they're well-tested)
# ============================================================================


def expected_score(rating_a: float, rating_b: float) -> float:
    """Elo win probability formula. Identical to v1."""
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def k_factor(matches_played: int, config: EloConfigV2) -> float:
    """K-factor by experience. Same as v1."""
    if matches_played < config.k_threshold_low:
        return config.k_initial
    elif matches_played < config.k_threshold_high:
        return config.k_mid
    else:
        return config.k_stable


def margin_of_victory_multiplier(
    winner_games: int,
    loser_games: int,
    config: EloConfigV2,
) -> float:
    """MoV adjustment. Same as v1."""
    total = winner_games + loser_games
    if total == 0:
        return 1.0

    win_ratio = winner_games / total
    raw_multiplier = 1.0 + (win_ratio - 0.5)

    return max(
        config.mov_min_multiplier,
        min(config.mov_max_multiplier, raw_multiplier),
    )


def parse_score_for_games(score: str | None) -> tuple[int, int]:
    """Parse Sackmann score string. Same as v1."""
    if not score or score.strip() in ("W/O", "DEF"):
        return 0, 0

    winner_total = 0
    loser_total = 0
    sets = score.replace("RET", "").replace("DEF", "").split()

    for set_score in sets:
        clean = set_score.split("(")[0]
        parts = clean.split("-")
        if len(parts) != 2:
            continue
        try:
            winner_total += int(parts[0])
            loser_total += int(parts[1])
        except ValueError:
            continue

    return winner_total, loser_total


# ============================================================================
# v2 NEW: Time decay
# ============================================================================


def apply_time_decay(
    state: PlayerEloStateV2,
    surface: Surface,
    current_date: date,
    config: EloConfigV2,
) -> None:
    """Pull rating toward population mean based on inactivity.

    If player hasn't played on this surface for > threshold_days, regress
    their rating toward population_mean by a small fraction per week of
    inactivity beyond the threshold.

    No effect if surface was never played (no last_match_date_per_surface entry).
    """
    last_date = state.last_match_date_per_surface.get(surface)
    if last_date is None:
        return

    days_inactive = (current_date - last_date).days
    if days_inactive <= config.inactivity_threshold_days:
        return

    weeks_inactive_beyond_threshold = (
        days_inactive - config.inactivity_threshold_days
    ) / 7.0

    # Cap total decay at reasonable amount (e.g. 50% pull at most)
    total_pull = min(0.5, weeks_inactive_beyond_threshold * config.inactivity_decay_per_week)

    current_rating = state.get_rating(surface, config)
    new_rating = current_rating + total_pull * (config.population_mean - current_rating)
    state.ratings[surface] = new_rating


# ============================================================================
# v2 update_ratings
# ============================================================================


def update_ratings(
    winner_state: PlayerEloStateV2,
    loser_state: PlayerEloStateV2,
    surface: Surface,
    match_date: date,
    score: str | None = None,
    tournament_level: str | None = None,
    is_retirement: bool = False,
    is_walkover: bool = False,
    config: EloConfigV2 | None = None,
) -> tuple[float, float, float]:
    """Update Elo ratings with v2 improvements applied in order:
    1. Time decay (apply BEFORE prediction so stale ratings are corrected)
    2. Standard Elo update
    3. Mean reversion (apply AFTER update so we pull recent changes toward mean)
    4. K reduction for large rating gaps
    """
    if config is None:
        config = EloConfigV2()

    if is_walkover:
        return 0.5, 0.0, 0.0

    # Step 1: Apply time decay before computing the prediction
    apply_time_decay(winner_state, surface, match_date, config)
    apply_time_decay(loser_state, surface, match_date, config)
    apply_time_decay(winner_state, "Overall", match_date, config)
    apply_time_decay(loser_state, "Overall", match_date, config)

    # Use effective rating (surface blended with overall for new players)
    winner_rating = winner_state.get_effective_rating(surface, config)
    loser_rating = loser_state.get_effective_rating(surface, config)

    expected_winner = expected_score(winner_rating, loser_rating)

    # Compute base K
    winner_matches = winner_state.get_matches_played(surface)
    loser_matches = loser_state.get_matches_played(surface)
    k = (k_factor(winner_matches, config) + k_factor(loser_matches, config)) / 2.0

    # Tournament weight
    if tournament_level and tournament_level in config.level_multipliers:
        k *= config.level_multipliers[tournament_level]

    # MoV adjustment
    if score:
        winner_games, loser_games = parse_score_for_games(score)
        mov_mult = margin_of_victory_multiplier(winner_games, loser_games, config)
        k *= mov_mult

    # Retirement: less reliable
    if is_retirement:
        k *= config.retirement_k_multiplier

    # v2 NEW: Reduce K for large rating spreads (variance dominates)
    rating_gap = abs(winner_rating - loser_rating)
    if rating_gap > config.large_spread_threshold:
        k *= config.large_spread_k_multiplier

    # Standard Elo update
    delta = k * (1.0 - expected_winner)

    new_winner_rating = winner_rating + delta
    new_loser_rating = loser_rating - delta

    # v2 NEW: Mean reversion - pull both ratings toward population mean
    mean = config.population_mean
    pull = config.mean_reversion_strength
    new_winner_rating = new_winner_rating + pull * (mean - new_winner_rating)
    new_loser_rating = new_loser_rating + pull * (mean - new_loser_rating)

    winner_state.ratings[surface] = new_winner_rating
    loser_state.ratings[surface] = new_loser_rating
    winner_state.matches_played[surface] = winner_matches + 1
    loser_state.matches_played[surface] = loser_matches + 1

    # Track last match date PER SURFACE for time decay
    winner_state.last_match_date_per_surface[surface] = match_date
    loser_state.last_match_date_per_surface[surface] = match_date

    # Update Overall (with same v2 improvements)
    overall_winner = winner_state.get_rating("Overall", config)
    overall_loser = loser_state.get_rating("Overall", config)
    expected_overall = expected_score(overall_winner, overall_loser)
    overall_k = k * 0.7  # bias toward surface
    overall_delta = overall_k * (1.0 - expected_overall)
    new_ow = overall_winner + overall_delta
    new_ol = overall_loser - overall_delta

    # Mean reversion on Overall too
    new_ow = new_ow + pull * (mean - new_ow)
    new_ol = new_ol + pull * (mean - new_ol)

    winner_state.ratings["Overall"] = new_ow
    loser_state.ratings["Overall"] = new_ol
    winner_state.matches_played["Overall"] = winner_state.get_matches_played("Overall") + 1
    loser_state.matches_played["Overall"] = loser_state.get_matches_played("Overall") + 1

    winner_state.last_match_date_per_surface["Overall"] = match_date
    loser_state.last_match_date_per_surface["Overall"] = match_date
    winner_state.last_match_date = match_date
    loser_state.last_match_date = match_date

    return expected_winner, delta, k


def predict_match_probability(
    player_a_state: PlayerEloStateV2,
    player_b_state: PlayerEloStateV2,
    surface: Surface,
    config: EloConfigV2 | None = None,
    prediction_date: date | None = None,
) -> float:
    """Predict P(player_a beats player_b) on surface.

    If prediction_date is provided, applies time decay first so stale
    ratings are corrected (e.g., for live prediction on a player returning
    from a layoff).
    """
    if config is None:
        config = EloConfigV2()

    if prediction_date:
        apply_time_decay(player_a_state, surface, prediction_date, config)
        apply_time_decay(player_b_state, surface, prediction_date, config)

    rating_a = player_a_state.get_effective_rating(surface, config)
    rating_b = player_b_state.get_effective_rating(surface, config)

    return expected_score(rating_a, rating_b)
