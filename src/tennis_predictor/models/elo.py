"""Surface-Adjusted Elo rating engine for tennis.

Implements the Elo rating system with tennis-specific adaptations:

1. **Per-surface ratings**: Hard/Clay/Grass tracked separately, since player
   skill varies dramatically by surface (e.g., Nadal on clay vs grass).

2. **Overall rating**: Blended estimate used as prior for new players or
   players who haven't played a surface recently.

3. **Dynamic K-factor**: Higher K early in career (rapid learning), decays
   as experience grows for stability.

4. **Tournament weight**: Grand Slams contribute slightly more than regular tour.

5. **Margin of victory**: A 6-0 6-0 win provides more information than 7-6 7-6.
   We use the ratio of games won as a multiplicative adjustment.

Mathematical foundations:
- Expected score: E_A = 1 / (1 + 10^((R_B - R_A) / 400))
- Rating update: R_A_new = R_A + K * (S_A - E_A)
  where S_A = 1 if A wins, 0 if A loses

References:
- Elo, Arpad E. (1978). "The Rating of Chessplayers, Past and Present"
- Kovalchik (2016). "Searching for the GOAT of tennis win prediction"
- tennisabstract.com Elo methodology
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal

Surface = Literal["Hard", "Clay", "Grass", "Carpet", "Overall"]
TournamentLevel = Literal["G", "M", "A", "D", "F", "C", "S", "O", "PM", "I"]


@dataclass
class EloConfig:
    """Configuration for the Elo calculation.

    Defaults are tuned based on tennis literature and known-good values.
    All parameters can be overridden for experimentation / hyperparameter search.
    """

    # Starting rating for new players
    initial_rating: float = 1500.0

    # K-factor decay: more volatile early, stable later
    k_initial: float = 40.0  # First 20 matches
    k_mid: float = 24.0      # Matches 21-50
    k_stable: float = 16.0   # Matches 51+
    k_threshold_low: int = 20
    k_threshold_high: int = 50

    # Tournament-level multipliers
    # Grand Slams provide more information (5-set, top fields)
    level_multipliers: dict[str, float] = field(default_factory=lambda: {
        "G": 1.1,    # Grand Slam
        "M": 1.05,   # Masters 1000 / WTA 1000
        "F": 1.05,   # Tour Finals
        "A": 1.0,    # ATP 500 / WTA 500
        "D": 1.0,    # ATP 250 / WTA 250
        "PM": 1.05,  # WTA Premier Mandatory (legacy)
        "I": 1.0,    # WTA International (legacy)
        "O": 1.0,    # Olympics
        "C": 0.85,   # Challenger - less reliable, smaller field
        "S": 0.7,    # Satellite/ITF - much weaker field
    })

    # Margin-of-victory adjustment range (capped to prevent extreme swings)
    mov_min_multiplier: float = 0.8
    mov_max_multiplier: float = 1.2

    # Surface vs Overall blending: weight of Overall rating when a player
    # has few matches on a given surface (acts as prior / smoothing)
    surface_prior_weight: float = 0.3
    surface_prior_threshold: int = 10  # Use blending below this match count

    # Retirement penalty: walkovers/retirements have less information
    retirement_k_multiplier: float = 0.5


@dataclass
class PlayerEloState:
    """In-memory state for a player's Elo ratings across all surfaces.

    Used during walk-forward backtest to avoid hitting the DB for every match.
    """

    player_id: str
    ratings: dict[Surface, float] = field(default_factory=dict)
    matches_played: dict[Surface, int] = field(default_factory=dict)
    last_match_date: date | None = None

    def get_rating(self, surface: Surface, config: EloConfig) -> float:
        """Get rating for surface; default to initial_rating if never played."""
        return self.ratings.get(surface, config.initial_rating)

    def get_matches_played(self, surface: Surface) -> int:
        return self.matches_played.get(surface, 0)

    def get_effective_rating(self, surface: Surface, config: EloConfig) -> float:
        """Get rating blended with Overall if surface has few matches.

        This gives more stable predictions for players new to a surface.
        """
        surface_matches = self.get_matches_played(surface)
        surface_rating = self.get_rating(surface, config)

        if surface == "Overall" or surface_matches >= config.surface_prior_threshold:
            return surface_rating

        # Blend surface with overall (acts as smoothing prior)
        overall_rating = self.get_rating("Overall", config)
        weight_surface = surface_matches / config.surface_prior_threshold
        weight_overall = config.surface_prior_weight * (1 - weight_surface)
        total_weight = weight_surface + weight_overall

        if total_weight == 0:
            return overall_rating

        return (
            surface_rating * weight_surface + overall_rating * weight_overall
        ) / total_weight


def expected_score(rating_a: float, rating_b: float) -> float:
    """Compute expected score (win probability) of player A vs player B.

    Standard Elo formula:
        E_A = 1 / (1 + 10^((R_B - R_A) / 400))

    Args:
        rating_a: Player A's current rating
        rating_b: Player B's current rating

    Returns:
        Probability between 0 and 1 that A beats B
    """
    return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


def k_factor(matches_played: int, config: EloConfig) -> float:
    """Compute K-factor based on player experience.

    Players with few matches get larger K (faster learning).
    Veterans get smaller K (rating is already well-established).
    """
    if matches_played < config.k_threshold_low:
        return config.k_initial
    elif matches_played < config.k_threshold_high:
        return config.k_mid
    else:
        return config.k_stable


def margin_of_victory_multiplier(
    winner_games: int,
    loser_games: int,
    config: EloConfig,
) -> float:
    """Compute multiplier for margin of victory.

    A dominant win (e.g. 12-2 in games) provides more information than
    a close win (e.g. 13-11). We use the ratio of games won, clipped to
    [mov_min, mov_max] to prevent extreme adjustments.

    Args:
        winner_games: Total games won by winner across all sets
        loser_games: Total games won by loser across all sets

    Returns:
        Multiplier in [mov_min_multiplier, mov_max_multiplier]
    """
    total = winner_games + loser_games
    if total == 0:
        return 1.0

    win_ratio = winner_games / total
    # Map 0.5 (close) -> 1.0, 1.0 (bagel) -> 1.5 ideally, but cap
    # Linear stretch: ratio 0.5 -> 1.0, ratio 1.0 -> 1.5
    raw_multiplier = 1.0 + (win_ratio - 0.5)

    return max(
        config.mov_min_multiplier,
        min(config.mov_max_multiplier, raw_multiplier),
    )


def parse_score_for_games(score: str | None) -> tuple[int, int]:
    """Parse a Sackmann score string to count total games won by winner / loser.

    Score format examples:
        "6-4 6-3"           -> (12, 7)
        "7-6(3) 4-6 6-2"    -> (17, 14)
        "6-1 3-2 RET"       -> (9, 3)   (retirement, partial set counted)
        "W/O"               -> (0, 0)   (walkover - no games played)

    Args:
        score: Raw score string from Sackmann data

    Returns:
        (winner_games, loser_games) - games won by each player
    """
    if not score or score.strip() in ("W/O", "DEF"):
        return 0, 0

    winner_total = 0
    loser_total = 0

    # Split into sets by whitespace, strip RET/W/O markers
    sets = score.replace("RET", "").replace("DEF", "").split()

    for set_score in sets:
        # Strip tiebreak parentheses, e.g. "7-6(3)" -> "7-6"
        clean = set_score.split("(")[0]
        parts = clean.split("-")
        if len(parts) != 2:
            continue
        try:
            w = int(parts[0])
            l = int(parts[1])
            winner_total += w
            loser_total += l
        except ValueError:
            continue

    return winner_total, loser_total


def update_ratings(
    winner_state: PlayerEloState,
    loser_state: PlayerEloState,
    surface: Surface,
    match_date: date,
    score: str | None = None,
    tournament_level: str | None = None,
    is_retirement: bool = False,
    is_walkover: bool = False,
    config: EloConfig | None = None,
) -> tuple[float, float, float]:
    """Update Elo ratings for both players after a match.

    Updates BOTH the surface-specific and Overall ratings.
    Modifies winner_state and loser_state in place.

    Args:
        winner_state: PlayerEloState for the match winner
        loser_state: PlayerEloState for the match loser
        surface: Hard / Clay / Grass / Carpet
        match_date: When the match was played
        score: Score string for margin-of-victory adjustment
        tournament_level: G/M/A/D etc for tournament weight
        is_retirement: True if match ended in retirement
        is_walkover: True if walkover (no match played)
        config: Elo configuration (uses defaults if None)

    Returns:
        (winner_expected_prob, rating_delta, k_used) - useful for logging/debugging
    """
    if config is None:
        config = EloConfig()

    # Walkovers contain no on-court information - skip rating update
    if is_walkover:
        return 0.5, 0.0, 0.0

    # Use effective rating (surface blended with overall for new players)
    winner_rating = winner_state.get_effective_rating(surface, config)
    loser_rating = loser_state.get_effective_rating(surface, config)

    # Expected probability of winner winning (before result is known)
    expected_winner = expected_score(winner_rating, loser_rating)

    # Determine K-factor (use winner's experience as base, average with loser's)
    winner_matches = winner_state.get_matches_played(surface)
    loser_matches = loser_state.get_matches_played(surface)
    k = (k_factor(winner_matches, config) + k_factor(loser_matches, config)) / 2.0

    # Apply tournament level multiplier
    if tournament_level and tournament_level in config.level_multipliers:
        k *= config.level_multipliers[tournament_level]

    # Apply margin-of-victory adjustment from score
    if score:
        winner_games, loser_games = parse_score_for_games(score)
        mov_mult = margin_of_victory_multiplier(winner_games, loser_games, config)
        k *= mov_mult

    # Retirements: less reliable signal, reduce K
    if is_retirement:
        k *= config.retirement_k_multiplier

    # Compute rating change: surprise factor (actual - expected) * K
    # Winner gets +delta, loser gets -delta (zero-sum)
    delta = k * (1.0 - expected_winner)

    # Update surface-specific ratings
    winner_state.ratings[surface] = winner_rating + delta
    loser_state.ratings[surface] = loser_rating - delta
    winner_state.matches_played[surface] = winner_matches + 1
    loser_state.matches_played[surface] = loser_matches + 1

    # Also update Overall ratings (with smaller weight since surface is primary)
    overall_winner = winner_state.get_rating("Overall", config)
    overall_loser = loser_state.get_rating("Overall", config)
    expected_overall = expected_score(overall_winner, overall_loser)
    overall_delta = k * 0.7 * (1.0 - expected_overall)  # 0.7 = surface bias

    winner_state.ratings["Overall"] = overall_winner + overall_delta
    loser_state.ratings["Overall"] = overall_loser - overall_delta
    winner_state.matches_played["Overall"] = (
        winner_state.get_matches_played("Overall") + 1
    )
    loser_state.matches_played["Overall"] = (
        loser_state.get_matches_played("Overall") + 1
    )

    # Track last match date (for inactivity detection later)
    winner_state.last_match_date = match_date
    loser_state.last_match_date = match_date

    return expected_winner, delta, k


def predict_match_probability(
    player_a_state: PlayerEloState,
    player_b_state: PlayerEloState,
    surface: Surface,
    config: EloConfig | None = None,
) -> float:
    """Predict probability that player A beats player B on given surface.

    Uses effective ratings (surface blended with overall for new players).

    Args:
        player_a_state: First player's Elo state
        player_b_state: Second player's Elo state
        surface: Match surface
        config: Elo configuration

    Returns:
        Probability between 0 and 1 that A beats B
    """
    if config is None:
        config = EloConfig()

    rating_a = player_a_state.get_effective_rating(surface, config)
    rating_b = player_b_state.get_effective_rating(surface, config)

    return expected_score(rating_a, rating_b)
