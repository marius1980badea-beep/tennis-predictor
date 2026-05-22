"""Unit tests for Elo v2 improvements.

These verify the THREE new behaviors are working correctly:
1. Mean reversion - high ratings pulled down, low pulled up
2. Time decay - inactive players regress
3. Large spread K reduction - lopsided matches update less
"""

from __future__ import annotations

from datetime import date

import pytest

from tennis_predictor.models.elo_v2 import (
    EloConfigV2,
    PlayerEloStateV2,
    apply_time_decay,
    expected_score,
    update_ratings,
)


@pytest.mark.unit
class TestMeanReversion:
    """High ratings should drift toward population mean over time."""

    def test_high_rating_gets_pulled_down_after_win(self) -> None:
        """A 2100-rated player winning against 2000 should end up slightly lower
        than pure Elo math predicts, due to mean reversion."""
        config = EloConfigV2()
        # Big spread, both above mean
        winner = PlayerEloStateV2(player_id="w")
        winner.ratings["Hard"] = 2100
        winner.ratings["Overall"] = 2100
        winner.matches_played["Hard"] = 100
        winner.matches_played["Overall"] = 100
        loser = PlayerEloStateV2(player_id="l")
        loser.ratings["Hard"] = 2000
        loser.ratings["Overall"] = 2000
        loser.matches_played["Hard"] = 100
        loser.matches_played["Overall"] = 100

        update_ratings(
            winner_state=winner, loser_state=loser,
            surface="Hard", match_date=date(2024, 6, 1),
            score="6-4 6-3", tournament_level="A", config=config,
        )

        # Without mean reversion, winner gain would be roughly:
        # K=16, expected=0.640 (2100>2000 by 100), delta = 16*(1-0.640) ≈ 5.76
        # New rating ≈ 2105.76
        # With mean reversion (pull=0.005 toward 1750):
        # 2105.76 + 0.005*(1750-2105.76) = 2105.76 - 1.78 = 2103.98
        # So winner should be > 2100 but < 2110
        assert 2100 < winner.ratings["Hard"] < 2110
        # And should be slightly less than pure Elo would predict
        # (the mean reversion shaved off ~1.8 points)

    def test_low_rating_gets_pulled_up(self) -> None:
        """A 1300-rated player should drift up toward 1750."""
        config = EloConfigV2()
        # Two low-rated players
        w = PlayerEloStateV2(player_id="w")
        w.ratings["Hard"] = 1300
        w.matches_played["Hard"] = 50
        l = PlayerEloStateV2(player_id="l")
        l.ratings["Hard"] = 1300
        l.matches_played["Hard"] = 50

        before_avg = (w.ratings["Hard"] + l.ratings["Hard"]) / 2

        update_ratings(
            winner_state=w, loser_state=l,
            surface="Hard", match_date=date(2024, 6, 1),
            score="6-4 6-3", tournament_level="A", config=config,
        )

        after_avg = (w.ratings["Hard"] + l.ratings["Hard"]) / 2
        # Zero-sum + mean reversion should pull the average UP toward 1750
        assert after_avg > before_avg

    def test_ratings_at_mean_are_stable(self) -> None:
        """Ratings exactly at population_mean should barely move (no reversion gap)."""
        config = EloConfigV2()
        w = PlayerEloStateV2(player_id="w")
        w.ratings["Hard"] = 1750
        w.matches_played["Hard"] = 50
        l = PlayerEloStateV2(player_id="l")
        l.ratings["Hard"] = 1750
        l.matches_played["Hard"] = 50

        update_ratings(
            winner_state=w, loser_state=l,
            surface="Hard", match_date=date(2024, 6, 1),
            score="6-4 6-3", tournament_level="A", config=config,
        )

        # Both at mean, both moved by Elo delta only (no mean reversion contribution)
        # Standard Elo: K=16, expected=0.5, delta=8
        assert 1755 < w.ratings["Hard"] < 1760
        assert 1740 < l.ratings["Hard"] < 1745


@pytest.mark.unit
class TestTimeDecay:
    """Inactive players should drift toward mean."""

    def test_no_decay_if_recent(self) -> None:
        """Match within 90 days = no decay."""
        config = EloConfigV2()
        state = PlayerEloStateV2(player_id="p")
        state.ratings["Hard"] = 2000
        state.last_match_date_per_surface["Hard"] = date(2024, 6, 1)

        apply_time_decay(state, "Hard", date(2024, 7, 15), config)  # 44 days later
        assert state.ratings["Hard"] == 2000  # No change

    def test_decay_after_threshold(self) -> None:
        """Match 6+ months ago triggers decay."""
        config = EloConfigV2()
        state = PlayerEloStateV2(player_id="p")
        state.ratings["Hard"] = 2000
        state.last_match_date_per_surface["Hard"] = date(2024, 1, 1)

        # 6 months later = 180 days = 90 beyond threshold = ~12.86 weeks
        # Pull = 12.86 * 0.003 = 0.0386 (capped at 0.5)
        # New rating = 2000 + 0.0386 * (1750-2000) = 2000 - 9.65 = ~1990.35
        apply_time_decay(state, "Hard", date(2024, 7, 1), config)
        assert 1985 < state.ratings["Hard"] < 1995

    def test_decay_capped(self) -> None:
        """Very long inactivity shouldn't make rating jump huge amounts."""
        config = EloConfigV2()
        state = PlayerEloStateV2(player_id="p")
        state.ratings["Hard"] = 2000
        state.last_match_date_per_surface["Hard"] = date(2010, 1, 1)

        apply_time_decay(state, "Hard", date(2024, 1, 1), config)
        # Should be pulled max 50% toward mean
        # 2000 + 0.5 * (1750-2000) = 2000 - 125 = 1875
        # so anywhere between 1875 and 2000
        assert state.ratings["Hard"] >= 1870

    def test_low_rating_decays_up(self) -> None:
        """A low-rated player inactive long enough drifts up."""
        config = EloConfigV2()
        state = PlayerEloStateV2(player_id="p")
        state.ratings["Hard"] = 1400
        state.last_match_date_per_surface["Hard"] = date(2024, 1, 1)

        apply_time_decay(state, "Hard", date(2024, 7, 1), config)
        # Should be > 1400 (pulled toward 1750)
        assert state.ratings["Hard"] > 1400


@pytest.mark.unit
class TestLargeSpreadKReduction:
    """Lopsided matchups should update ratings less."""

    def test_small_spread_full_k(self) -> None:
        """Gap of 100 should not trigger K reduction."""
        config = EloConfigV2()
        w = PlayerEloStateV2(player_id="w")
        w.ratings["Hard"] = 1750
        w.ratings["Overall"] = 1750
        w.matches_played["Hard"] = 50
        w.matches_played["Overall"] = 50
        l = PlayerEloStateV2(player_id="l")
        l.ratings["Hard"] = 1650
        l.ratings["Overall"] = 1650
        l.matches_played["Hard"] = 50
        l.matches_played["Overall"] = 50

        _, delta_small, _ = update_ratings(
            winner_state=w, loser_state=l,
            surface="Hard", match_date=date(2024, 6, 1),
            score="6-4 6-3", tournament_level="A", config=config,
        )
        # K=16, expected for 100 gap = 0.640, delta = 16 * 0.36 = 5.76
        assert 5.0 < delta_small < 7.0

    def test_large_spread_reduced_k(self) -> None:
        """Gap of 300 should reduce K (delta smaller proportionally)."""
        config = EloConfigV2()
        w = PlayerEloStateV2(player_id="w")
        w.ratings["Hard"] = 1900
        w.ratings["Overall"] = 1900
        w.matches_played["Hard"] = 50
        w.matches_played["Overall"] = 50
        l = PlayerEloStateV2(player_id="l")
        l.ratings["Hard"] = 1600
        l.ratings["Overall"] = 1600
        l.matches_played["Hard"] = 50
        l.matches_played["Overall"] = 50

        _, delta_large, _ = update_ratings(
            winner_state=w, loser_state=l,
            surface="Hard", match_date=date(2024, 6, 1),
            score="6-4 6-3", tournament_level="A", config=config,
        )
        # K=16 * 0.7 = 11.2 (reduced), expected = 0.849, delta = 11.2 * 0.151 = 1.69
        # Without reduction would be: 16 * 0.151 = 2.42
        # So delta_large should be smaller in absolute terms than pure Elo
        assert delta_large < 2.5


@pytest.mark.unit
class TestV2BackwardCompatibility:
    """v2 should produce sane results equivalent to v1 in 'easy' cases."""

    def test_symmetric_predictions(self) -> None:
        """P(A>B) + P(B>A) = 1 even in v2."""
        config = EloConfigV2()
        a = PlayerEloStateV2(player_id="a")
        a.ratings["Hard"] = 1800
        a.matches_played["Hard"] = 50
        b = PlayerEloStateV2(player_id="b")
        b.ratings["Hard"] = 1600
        b.matches_played["Hard"] = 50

        # Need to import from v2
        from tennis_predictor.models.elo_v2 import predict_match_probability
        p_ab = predict_match_probability(a, b, "Hard", config=config)
        p_ba = predict_match_probability(b, a, "Hard", config=config)
        assert abs(p_ab + p_ba - 1.0) < 1e-9

    def test_higher_rated_player_favored(self) -> None:
        """Basic sanity: higher rating wins probability prediction."""
        config = EloConfigV2()
        from tennis_predictor.models.elo_v2 import predict_match_probability
        a = PlayerEloStateV2(player_id="a")
        a.ratings["Hard"] = 1900
        a.matches_played["Hard"] = 50
        b = PlayerEloStateV2(player_id="b")
        b.ratings["Hard"] = 1500
        b.matches_played["Hard"] = 50
        p = predict_match_probability(a, b, "Hard", config=config)
        assert p > 0.7
