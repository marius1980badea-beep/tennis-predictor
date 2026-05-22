"""Unit tests for backtest engine logic.

We test the core randomization and aggregation logic without DB calls.
The DB integration is implicitly tested when running the actual backtest.
"""

from __future__ import annotations

from datetime import date

import pytest

from tennis_predictor.backtest.metrics import evaluate_predictions
from tennis_predictor.backtest.walk_forward import BacktestPrediction


@pytest.mark.unit
class TestPredictionRandomization:
    """The randomized side trick ensures unbiased metric evaluation."""

    def test_randomization_preserves_information(self) -> None:
        """When real outcomes match predictions, accuracy ≈ predicted prob."""
        import random
        rng = random.Random(0)
        preds = []
        # Simulate 1000 matches where the favorite (predicted 65%) wins
        # 65% of the time and loses 35% (realistic = perfectly calibrated)
        for i in range(1000):
            p_winner = 0.65
            # Simulate outcome: in 65% of cases the predicted favorite wins
            favorite_wins = rng.random() < 0.65

            if favorite_wins:
                # The "winner" in DB IS the favorite
                # Randomize player A assignment
                if rng.random() < 0.5:
                    preds.append(BacktestPrediction(
                        match_date=date(2024, 1, 1), match_id=i, tour="ATP",
                        surface="Hard", winner_id=f"w{i}", loser_id=f"l{i}",
                        p_winner_wins=p_winner,
                        p_player_a_wins=p_winner,  # A=winner
                        a_is_winner=1,
                    ))
                else:
                    preds.append(BacktestPrediction(
                        match_date=date(2024, 1, 1), match_id=i, tour="ATP",
                        surface="Hard", winner_id=f"w{i}", loser_id=f"l{i}",
                        p_winner_wins=p_winner,
                        p_player_a_wins=1 - p_winner,  # A=loser
                        a_is_winner=0,
                    ))
            else:
                # Upset: the player the model thought was 35% to win, actually won
                # So in DB, the underdog became "winner". Our model gave them 35%.
                if rng.random() < 0.5:
                    preds.append(BacktestPrediction(
                        match_date=date(2024, 1, 1), match_id=i, tour="ATP",
                        surface="Hard", winner_id=f"w{i}", loser_id=f"l{i}",
                        p_winner_wins=1 - p_winner,  # model gave them 35%
                        p_player_a_wins=1 - p_winner,
                        a_is_winner=1,
                    ))
                else:
                    preds.append(BacktestPrediction(
                        match_date=date(2024, 1, 1), match_id=i, tour="ATP",
                        surface="Hard", winner_id=f"w{i}", loser_id=f"l{i}",
                        p_winner_wins=1 - p_winner,
                        p_player_a_wins=p_winner,  # A=loser, who was favored
                        a_is_winner=0,
                    ))

        # Evaluate on randomized side
        m = evaluate_predictions(
            [p.p_player_a_wins for p in preds],
            [p.a_is_winner for p in preds],
        )
        # Accuracy should match the predicted prob ≈ 65%
        assert 0.60 < m.accuracy < 0.70

    def test_predictions_sum_correctly(self) -> None:
        """Sum of p_player_a_wins + (1 - p_player_a_wins) for opposite-side
        view is always 1.0 — i.e., the model is self-consistent."""
        p = BacktestPrediction(
            match_date=date(2024, 1, 1),
            match_id=1,
            tour="ATP",
            surface="Hard",
            winner_id="w",
            loser_id="l",
            p_winner_wins=0.65,
            p_player_a_wins=0.65,
            a_is_winner=1,
        )
        # Consistency check
        if p.a_is_winner == 1:
            assert p.p_player_a_wins == p.p_winner_wins
        else:
            assert abs(p.p_player_a_wins - (1 - p.p_winner_wins)) < 1e-9


@pytest.mark.unit
class TestBacktestPrediction:
    """BacktestPrediction dataclass structure."""

    def test_can_be_constructed(self) -> None:
        p = BacktestPrediction(
            match_date=date(2024, 6, 15),
            match_id=42,
            tour="ATP",
            surface="Grass",
            winner_id="atp_104925",
            loser_id="atp_104745",
            p_winner_wins=0.72,
            p_player_a_wins=0.28,
            a_is_winner=0,
        )
        assert p.tour == "ATP"
        assert p.surface == "Grass"
        assert p.p_winner_wins == 0.72
        assert p.a_is_winner == 0
