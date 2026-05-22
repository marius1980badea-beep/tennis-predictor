"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def sample_sackmann_match_row():
    """Sample row mimicking Sackmann ATP CSV format."""
    import pandas as pd

    return pd.Series({
        "tourney_id": "580",
        "tourney_name": "Australian Open",
        "surface": "Hard",
        "draw_size": 128,
        "tourney_level": "G",
        "tourney_date": "20240114",
        "match_num": 1,
        "winner_id": "104925",
        "winner_seed": "1",
        "winner_entry": None,
        "winner_name": "Novak Djokovic",
        "winner_hand": "R",
        "winner_ht": 188,
        "winner_ioc": "SRB",
        "winner_age": 36.5,
        "loser_id": "207989",
        "loser_seed": None,
        "loser_entry": "Q",
        "loser_name": "Some Player",
        "loser_hand": "R",
        "loser_ht": 185,
        "loser_ioc": "USA",
        "loser_age": 24.0,
        "score": "6-1 6-2 6-3",
        "best_of": 5,
        "round": "R128",
        "minutes": 105,
        "w_ace": 8,
        "w_df": 2,
        "w_svpt": 65,
        "w_1stIn": 42,
        "w_1stWon": 35,
        "w_2ndWon": 14,
        "w_SvGms": 15,
        "w_bpSaved": 2,
        "w_bpFaced": 3,
        "l_ace": 3,
        "l_df": 5,
        "l_svpt": 75,
        "l_1stIn": 45,
        "l_1stWon": 28,
        "l_2ndWon": 12,
        "l_SvGms": 15,
        "l_bpSaved": 4,
        "l_bpFaced": 10,
        "winner_rank": 1,
        "winner_rank_points": 11055,
        "loser_rank": 156,
        "loser_rank_points": 425,
    })
