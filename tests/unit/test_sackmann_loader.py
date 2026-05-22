"""Unit tests for Sackmann loader helper functions.

These tests don't require database access - they test pure transformation logic.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from tennis_predictor.data.loaders.sackmann import (
    _normalize_hand,
    _parse_yyyymmdd,
    _safe_int,
    _safe_str,
)


@pytest.mark.unit
class TestSafeStr:
    """Tests for _safe_str helper."""

    def test_normal_string(self) -> None:
        assert _safe_str("hello") == "hello"

    def test_strips_whitespace(self) -> None:
        assert _safe_str("  hello  ") == "hello"

    def test_none(self) -> None:
        assert _safe_str(None) is None

    def test_nan(self) -> None:
        assert _safe_str(np.nan) is None

    def test_empty_string(self) -> None:
        assert _safe_str("") is None

    def test_whitespace_only(self) -> None:
        assert _safe_str("   ") is None

    def test_numeric_input(self) -> None:
        assert _safe_str(42) == "42"


@pytest.mark.unit
class TestSafeInt:
    """Tests for _safe_int helper."""

    def test_normal_int(self) -> None:
        assert _safe_int(42) == 42

    def test_float(self) -> None:
        assert _safe_int(42.0) == 42

    def test_string_numeric(self) -> None:
        assert _safe_int("42") == 42

    def test_none(self) -> None:
        assert _safe_int(None) is None

    def test_nan(self) -> None:
        assert _safe_int(np.nan) is None

    def test_non_numeric_seed(self) -> None:
        """Seeds in Sackmann data can be 'WC', 'Q', 'LL', etc."""
        assert _safe_int("WC") is None
        assert _safe_int("Q") is None
        assert _safe_int("LL") is None


@pytest.mark.unit
class TestNormalizeHand:
    """Tests for _normalize_hand helper."""

    def test_right(self) -> None:
        assert _normalize_hand("R") == "R"
        assert _normalize_hand("r") == "R"

    def test_left(self) -> None:
        assert _normalize_hand("L") == "L"

    def test_unknown(self) -> None:
        assert _normalize_hand("U") == "U"

    def test_none(self) -> None:
        assert _normalize_hand(None) is None

    def test_invalid(self) -> None:
        assert _normalize_hand("X") is None
        assert _normalize_hand("BOTH") is None


@pytest.mark.unit
class TestParseYyyymmdd:
    """Tests for _parse_yyyymmdd helper."""

    def test_valid_date(self) -> None:
        assert _parse_yyyymmdd("20240114") == date(2024, 1, 14)

    def test_int_input(self) -> None:
        # Sackmann sometimes has dates as integers when CSV is parsed loosely
        assert _parse_yyyymmdd(20240114) == date(2024, 1, 14)

    def test_none(self) -> None:
        assert _parse_yyyymmdd(None) is None

    def test_nan(self) -> None:
        assert _parse_yyyymmdd(np.nan) is None

    def test_wrong_length(self) -> None:
        assert _parse_yyyymmdd("2024") is None
        assert _parse_yyyymmdd("202401140") is None

    def test_invalid_date(self) -> None:
        assert _parse_yyyymmdd("20241332") is None  # Month 13
        assert _parse_yyyymmdd("abcdefgh") is None


@pytest.mark.unit
class TestBuildMatchRecord:
    """Tests for the match record builder."""

    def test_basic_match_record(self, sample_sackmann_match_row: pd.Series) -> None:
        """Verify a complete match row is transformed correctly."""
        from tennis_predictor.data.loaders.sackmann import SackmannLoader

        # We instantiate but don't call settings-dependent methods
        loader = SackmannLoader.__new__(SackmannLoader)
        loader.tour = "ATP"  # type: ignore[attr-defined]
        loader.player_id_prefix = "atp_"  # type: ignore[attr-defined]
        loader.SURFACE_MAP = SackmannLoader.SURFACE_MAP
        loader.LEVEL_MAP = SackmannLoader.LEVEL_MAP

        match, winner_stats, loser_stats = loader._build_match_record(
            sample_sackmann_match_row, year=2024
        )

        assert match is not None
        assert match["winner_id"] == "atp_104925"
        assert match["loser_id"] == "atp_207989"
        assert match["surface"] == "Hard"
        assert match["best_of"] == 5
        assert match["round"] == "R128"
        assert match["match_date"] == date(2024, 1, 14)
        assert match["retirement"] is False
        assert match["walkover"] is False
        assert match["score"] == "6-1 6-2 6-3"
        assert match["source_match_id"] == "2024_580_1"

        assert winner_stats is not None
        assert winner_stats["player_id"] == "atp_104925"
        assert winner_stats["is_winner"] is True
        assert winner_stats["aces"] == 8
        assert winner_stats["serve_points"] == 65

        assert loser_stats is not None
        assert loser_stats["is_winner"] is False
        assert loser_stats["aces"] == 3

    def test_retirement_detected(self, sample_sackmann_match_row: pd.Series) -> None:
        """RET in score should set retirement flag."""
        from tennis_predictor.data.loaders.sackmann import SackmannLoader

        loader = SackmannLoader.__new__(SackmannLoader)
        loader.tour = "ATP"
        loader.player_id_prefix = "atp_"
        loader.SURFACE_MAP = SackmannLoader.SURFACE_MAP
        loader.LEVEL_MAP = SackmannLoader.LEVEL_MAP

        row = sample_sackmann_match_row.copy()
        row["score"] = "6-1 3-2 RET"

        match, _, _ = loader._build_match_record(row, year=2024)
        assert match is not None
        assert match["retirement"] is True
        assert match["walkover"] is False

    def test_walkover_detected(self, sample_sackmann_match_row: pd.Series) -> None:
        """W/O in score should set walkover flag."""
        from tennis_predictor.data.loaders.sackmann import SackmannLoader

        loader = SackmannLoader.__new__(SackmannLoader)
        loader.tour = "ATP"
        loader.player_id_prefix = "atp_"
        loader.SURFACE_MAP = SackmannLoader.SURFACE_MAP
        loader.LEVEL_MAP = SackmannLoader.LEVEL_MAP

        row = sample_sackmann_match_row.copy()
        row["score"] = "W/O"

        match, _, _ = loader._build_match_record(row, year=2024)
        assert match is not None
        assert match["walkover"] is True

    def test_missing_winner_id_returns_none(
        self, sample_sackmann_match_row: pd.Series
    ) -> None:
        """Rows without winner_id should be skipped."""
        from tennis_predictor.data.loaders.sackmann import SackmannLoader

        loader = SackmannLoader.__new__(SackmannLoader)
        loader.tour = "ATP"
        loader.player_id_prefix = "atp_"
        loader.SURFACE_MAP = SackmannLoader.SURFACE_MAP
        loader.LEVEL_MAP = SackmannLoader.LEVEL_MAP

        row = sample_sackmann_match_row.copy()
        row["winner_id"] = np.nan

        match, _, _ = loader._build_match_record(row, year=2024)
        assert match is None

    def test_same_winner_and_loser_returns_none(
        self, sample_sackmann_match_row: pd.Series
    ) -> None:
        """Defensive: same player as winner and loser should be skipped."""
        from tennis_predictor.data.loaders.sackmann import SackmannLoader

        loader = SackmannLoader.__new__(SackmannLoader)
        loader.tour = "ATP"
        loader.player_id_prefix = "atp_"
        loader.SURFACE_MAP = SackmannLoader.SURFACE_MAP
        loader.LEVEL_MAP = SackmannLoader.LEVEL_MAP

        row = sample_sackmann_match_row.copy()
        row["loser_id"] = row["winner_id"]

        match, _, _ = loader._build_match_record(row, year=2024)
        assert match is None
