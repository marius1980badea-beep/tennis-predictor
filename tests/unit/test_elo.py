"""Unit tests for Elo math functions.

These verify the mathematical correctness of the rating system.
"""

from __future__ import annotations

from datetime import date

import pytest

from tennis_predictor.models.elo import (
    EloConfig,
    PlayerEloState,
    expected_score,
    k_factor,
    margin_of_victory_multiplier,
    parse_score_for_games,
    predict_match_probability,
    update_ratings,
)


@pytest.mark.unit
class TestExpectedScore:
    """Tests for the expected score (win probability) calculation."""

    def test_equal_ratings_50_50(self) -> None:
        """Two equal players should have 50% expected win prob each."""
        assert expected_score(1500, 1500) == 0.5

    def test_higher_rating_favored(self) -> None:
        """Player with higher rating should be favored."""
        prob = expected_score(1700, 1500)
        assert prob > 0.5
        assert prob < 1.0

    def test_400_point_gap(self) -> None:
        """A 400-point gap means ~91% win probability for stronger player.

        This is a fundamental Elo property: 400 points = 10x more likely to win.
        Verify P(stronger wins) / P(weaker wins) = 10.
        """
        p_strong = expected_score(1900, 1500)
        p_weak = 1 - p_strong
        assert abs(p_strong / p_weak - 10.0) < 0.01

    def test_symmetry(self) -> None:
        """P(A beats B) + P(B beats A) = 1.0"""
        p_ab = expected_score(1700, 1500)
        p_ba = expected_score(1500, 1700)
        assert abs(p_ab + p_ba - 1.0) < 1e-9

    def test_extreme_gap(self) -> None:
        """Very large rating gap approaches but never reaches 1.0."""
        prob = expected_score(2400, 1200)
        assert prob > 0.99
        assert prob < 1.0


@pytest.mark.unit
class TestKFactor:
    """K-factor varies with experience."""

    def test_new_player_high_k(self) -> None:
        """First match: high K for fast learning."""
        config = EloConfig()
        assert k_factor(0, config) == config.k_initial
        assert k_factor(5, config) == config.k_initial

    def test_mid_career(self) -> None:
        """Mid-career: moderate K."""
        config = EloConfig()
        assert k_factor(25, config) == config.k_mid

    def test_veteran(self) -> None:
        """Veteran: stable K."""
        config = EloConfig()
        assert k_factor(100, config) == config.k_stable
        assert k_factor(500, config) == config.k_stable

    def test_thresholds_exact(self) -> None:
        """Boundary conditions at exact thresholds."""
        config = EloConfig(k_threshold_low=20, k_threshold_high=50)
        assert k_factor(19, config) == config.k_initial
        assert k_factor(20, config) == config.k_mid
        assert k_factor(49, config) == config.k_mid
        assert k_factor(50, config) == config.k_stable


@pytest.mark.unit
class TestMarginOfVictoryMultiplier:
    """Margin of victory adjustment."""

    def test_close_match_neutral(self) -> None:
        """Equal-ish games -> close to 1.0 multiplier."""
        # 13-11 (e.g. 6-4 7-6) -> ratio ~0.54
        mult = margin_of_victory_multiplier(13, 11, EloConfig())
        assert 0.95 < mult < 1.10

    def test_dominant_win_capped(self) -> None:
        """Bagel (6-0 6-0) caps at mov_max_multiplier."""
        config = EloConfig()
        mult = margin_of_victory_multiplier(12, 0, config)
        assert mult == config.mov_max_multiplier

    def test_zero_games(self) -> None:
        """No games played (e.g. walkover) returns neutral 1.0."""
        mult = margin_of_victory_multiplier(0, 0, EloConfig())
        assert mult == 1.0

    def test_within_range(self) -> None:
        """All multipliers are within configured range."""
        config = EloConfig()
        for wg in range(0, 30):
            for lg in range(0, 30):
                mult = margin_of_victory_multiplier(wg, lg, config)
                assert config.mov_min_multiplier <= mult <= config.mov_max_multiplier


@pytest.mark.unit
class TestParseScore:
    """Score string parsing."""

    def test_standard_3_set(self) -> None:
        """Best-of-3 win: 6-4 6-3 -> winner 12, loser 7."""
        w, l = parse_score_for_games("6-4 6-3")
        assert w == 12
        assert l == 7

    def test_5_set_grand_slam(self) -> None:
        """Best-of-5: 6-4 4-6 6-3 6-2 -> 22, 15."""
        w, l = parse_score_for_games("6-4 4-6 6-3 6-2")
        assert w == 22
        assert l == 15

    def test_tiebreak_parens(self) -> None:
        """Tiebreak scores in parens are stripped."""
        w, l = parse_score_for_games("7-6(3) 7-6(5)")
        assert w == 14
        assert l == 12

    def test_retirement(self) -> None:
        """RET marker is handled gracefully."""
        w, l = parse_score_for_games("6-1 3-2 RET")
        assert w == 9
        assert l == 3

    def test_walkover(self) -> None:
        """Walkover returns 0, 0."""
        assert parse_score_for_games("W/O") == (0, 0)
        assert parse_score_for_games("DEF") == (0, 0)

    def test_none_input(self) -> None:
        """None / empty input."""
        assert parse_score_for_games(None) == (0, 0)
        assert parse_score_for_games("") == (0, 0)

    def test_malformed_score(self) -> None:
        """Malformed sets are silently skipped."""
        w, l = parse_score_for_games("6-4 garbage 6-2")
        assert w == 12
        assert l == 6


@pytest.mark.unit
class TestPlayerEloState:
    """Player state management."""

    def test_default_rating(self) -> None:
        """Unrated player gets initial_rating."""
        config = EloConfig()
        state = PlayerEloState(player_id="test")
        assert state.get_rating("Hard", config) == config.initial_rating

    def test_explicit_rating(self) -> None:
        config = EloConfig()
        state = PlayerEloState(player_id="test")
        state.ratings["Hard"] = 1750
        assert state.get_rating("Hard", config) == 1750

    def test_effective_rating_uses_overall_for_new(self) -> None:
        """Player with few surface matches blends with Overall."""
        config = EloConfig()
        state = PlayerEloState(player_id="test")
        state.ratings["Overall"] = 1800
        state.matches_played["Overall"] = 100
        # No Clay history at all -> should pull strongly from Overall
        rating = state.get_effective_rating("Clay", config)
        # Without any clay data, effective should be close to overall (1800)
        # but not exactly equal due to blending formula
        assert 1700 < rating <= 1800

    def test_effective_rating_stable_with_many_matches(self) -> None:
        """Once player has many surface matches, effective = surface."""
        config = EloConfig(surface_prior_threshold=10)
        state = PlayerEloState(player_id="test")
        state.ratings["Overall"] = 1800
        state.ratings["Clay"] = 2000
        state.matches_played["Overall"] = 100
        state.matches_played["Clay"] = 50  # Above threshold
        assert state.get_effective_rating("Clay", config) == 2000


@pytest.mark.unit
class TestUpdateRatings:
    """End-to-end rating update behavior."""

    def test_upset_increases_winner_rating_more(self) -> None:
        """Underdog winning -> bigger swing than favorite winning."""
        config = EloConfig()
        # Setup: favorite (high rating) vs underdog (low rating)
        favorite = PlayerEloState(player_id="fav")
        favorite.ratings["Hard"] = 1800
        favorite.ratings["Overall"] = 1800
        favorite.matches_played["Hard"] = 50
        favorite.matches_played["Overall"] = 50

        underdog = PlayerEloState(player_id="ud")
        underdog.ratings["Hard"] = 1500
        underdog.ratings["Overall"] = 1500
        underdog.matches_played["Hard"] = 50
        underdog.matches_played["Overall"] = 50

        # Underdog wins (upset)
        _, delta_upset, _ = update_ratings(
            winner_state=underdog,
            loser_state=favorite,
            surface="Hard",
            match_date=date(2024, 1, 1),
            score="7-5 6-4",
            tournament_level="A",
            config=config,
        )

        # Setup again, favorite wins this time
        favorite2 = PlayerEloState(player_id="fav2")
        favorite2.ratings["Hard"] = 1800
        favorite2.ratings["Overall"] = 1800
        favorite2.matches_played["Hard"] = 50
        favorite2.matches_played["Overall"] = 50

        underdog2 = PlayerEloState(player_id="ud2")
        underdog2.ratings["Hard"] = 1500
        underdog2.ratings["Overall"] = 1500
        underdog2.matches_played["Hard"] = 50
        underdog2.matches_played["Overall"] = 50

        _, delta_expected, _ = update_ratings(
            winner_state=favorite2,
            loser_state=underdog2,
            surface="Hard",
            match_date=date(2024, 1, 1),
            score="7-5 6-4",
            tournament_level="A",
            config=config,
        )

        # Upset delta should be much larger than expected-result delta
        assert delta_upset > delta_expected * 2

    def test_walkover_no_change(self) -> None:
        """Walkover should not change ratings."""
        config = EloConfig()
        winner = PlayerEloState(player_id="w")
        winner.ratings["Hard"] = 1700
        loser = PlayerEloState(player_id="l")
        loser.ratings["Hard"] = 1500

        update_ratings(
            winner_state=winner,
            loser_state=loser,
            surface="Hard",
            match_date=date(2024, 1, 1),
            is_walkover=True,
            config=config,
        )

        assert winner.ratings["Hard"] == 1700
        assert loser.ratings["Hard"] == 1500

    def test_zero_sum(self) -> None:
        """Winner's gain equals loser's loss (within float tolerance)."""
        config = EloConfig()
        winner = PlayerEloState(player_id="w")
        winner.ratings["Hard"] = 1600
        winner.ratings["Overall"] = 1600
        winner.matches_played["Hard"] = 30
        winner.matches_played["Overall"] = 30

        loser = PlayerEloState(player_id="l")
        loser.ratings["Hard"] = 1600
        loser.ratings["Overall"] = 1600
        loser.matches_played["Hard"] = 30
        loser.matches_played["Overall"] = 30

        winner_before = winner.ratings["Hard"]
        loser_before = loser.ratings["Hard"]

        update_ratings(
            winner_state=winner,
            loser_state=loser,
            surface="Hard",
            match_date=date(2024, 1, 1),
            score="6-3 6-3",
            tournament_level="A",
            config=config,
        )

        winner_gain = winner.ratings["Hard"] - winner_before
        loser_loss = loser_before - loser.ratings["Hard"]
        assert abs(winner_gain - loser_loss) < 0.001

    def test_match_count_increments(self) -> None:
        """Both players' match counts increase by 1 on the surface played."""
        config = EloConfig()
        w = PlayerEloState(player_id="w")
        l = PlayerEloState(player_id="l")

        update_ratings(
            winner_state=w, loser_state=l, surface="Clay",
            match_date=date(2024, 1, 1), score="6-3 6-3",
            tournament_level="A", config=config,
        )

        assert w.matches_played["Clay"] == 1
        assert l.matches_played["Clay"] == 1
        assert w.matches_played.get("Hard", 0) == 0  # Untouched
        # Overall also increments
        assert w.matches_played["Overall"] == 1


@pytest.mark.unit
class TestPredictMatchProbability:
    """Prediction interface for live use."""

    def test_predicts_higher_rated_favorite(self) -> None:
        config = EloConfig()
        a = PlayerEloState(player_id="a")
        a.ratings["Hard"] = 1800
        a.matches_played["Hard"] = 50
        b = PlayerEloState(player_id="b")
        b.ratings["Hard"] = 1500
        b.matches_played["Hard"] = 50

        prob_a_wins = predict_match_probability(a, b, "Hard", config)
        assert prob_a_wins > 0.7

    def test_symmetric_predictions(self) -> None:
        """P(A beats B) + P(B beats A) = 1.0"""
        config = EloConfig()
        a = PlayerEloState(player_id="a")
        a.ratings["Hard"] = 1750
        a.matches_played["Hard"] = 50
        b = PlayerEloState(player_id="b")
        b.ratings["Hard"] = 1600
        b.matches_played["Hard"] = 50

        p_a = predict_match_probability(a, b, "Hard", config)
        p_b = predict_match_probability(b, a, "Hard", config)
        assert abs(p_a + p_b - 1.0) < 1e-9
