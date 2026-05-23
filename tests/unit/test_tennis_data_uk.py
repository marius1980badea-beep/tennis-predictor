"""Unit tests for the tennis-data.co.uk loader.

These tests exercise URL construction, probability math, and the parsing
pipeline on synthetic DataFrames -- no network or DB access required.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tennis_predictor.data.loaders.tennis_data_uk import (
    BOOKMAKER_COLUMNS,
    EARLIEST_YEAR,
    OddsRow,
    build_url,
    compute_implied,
    download_year_file,
    extract_match_metadata,
    extract_odds_rows,
)
from tennis_predictor.data.loaders.load_odds import parse_year_spec


# =============================================================================
# URL construction
# =============================================================================

class TestBuildUrl:
    def test_atp_xlsx_modern(self):
        assert build_url(2024, "ATP") == "http://www.tennis-data.co.uk/2024/2024.xlsx"

    def test_atp_xlsx_at_boundary_year_2013(self):
        # 2013 is the first xlsx year (per http://www.tennis-data.co.uk/alldata.php)
        assert build_url(2013, "ATP") == "http://www.tennis-data.co.uk/2013/2013.xlsx"

    def test_atp_xls_legacy(self):
        assert build_url(2010, "ATP") == "http://www.tennis-data.co.uk/2010/2010.xls"

    def test_atp_xls_at_boundary_year_2012(self):
        # 2012 is the last xls year
        assert build_url(2012, "ATP") == "http://www.tennis-data.co.uk/2012/2012.xls"

    def test_wta_xlsx_modern(self):
        assert build_url(2024, "WTA") == "http://www.tennis-data.co.uk/2024w/2024.xlsx"

    def test_wta_xls_legacy(self):
        assert build_url(2010, "WTA") == "http://www.tennis-data.co.uk/2010w/2010.xls"

    def test_invalid_tour_raises(self):
        with pytest.raises(ValueError, match="ATP or WTA"):
            build_url(2024, "ITF")


# =============================================================================
# download_year_file: argument validation (no real network)
# =============================================================================

class TestDownloadYearFile:
    def test_rejects_pre_atp_earliest(self, tmp_path):
        with pytest.raises(ValueError, match="2001 onward"):
            download_year_file(2000, "ATP", tmp_path)

    def test_rejects_pre_wta_earliest(self, tmp_path):
        with pytest.raises(ValueError, match="2007 onward"):
            download_year_file(2005, "WTA", tmp_path)

    def test_earliest_year_constants_match_data_source(self):
        # If this fails, tennis-data.co.uk schema changed -- update constants.
        assert EARLIEST_YEAR["ATP"] == 2001
        assert EARLIEST_YEAR["WTA"] == 2007

    def test_uses_cache_when_file_exists(self, tmp_path):
        # Pre-create a cached file and verify the loader returns it without HTTP.
        cached = tmp_path / "atp_2024.xlsx"
        cached.write_bytes(b"fake excel content")
        # If this triggered a real download, it would fail -- the test passes
        # because the cache is honoured and no network call happens.
        result = download_year_file(2024, "ATP", tmp_path)
        assert result == cached
        assert cached.read_bytes() == b"fake excel content"  # not overwritten


# =============================================================================
# compute_implied: probability and vig math
# =============================================================================

class TestComputeImplied:
    def test_pinnacle_typical_market(self):
        # Roughly even market, Pinnacle-like 2-3% margin
        w, l, v = compute_implied(1.91, 1.91)
        assert w == pytest.approx(1 / 1.91)
        assert l == pytest.approx(1 / 1.91)
        assert v == pytest.approx(2 / 1.91 - 1.0)
        assert 0.04 < v < 0.05  # ~4.7% vig

    def test_pinnacle_tight_market(self):
        # Pinnacle-tight market on a Slam final: 1.97 / 2.00 -> vig ~0.76%
        _, _, v = compute_implied(1.97, 2.00)
        # vig = 1/1.97 + 1/2.00 - 1 = 0.5076 + 0.5 - 1 = 0.0076
        assert 0.005 < v < 0.015

    def test_soft_book_wider_margin(self):
        # Bet365-style market with bigger margin
        _, _, v = compute_implied(1.70, 2.00)
        assert 0.04 < v < 0.10

    def test_max_can_have_near_zero_vig(self):
        # MaxW/MaxL is best across books -- can approach zero vig
        _, _, v = compute_implied(2.05, 2.05)
        assert v < 0.0  # actually slight arb in this exact case

    def test_perfectly_even_zero_vig(self):
        _, _, v = compute_implied(2.0, 2.0)
        assert v == pytest.approx(0.0)

    def test_implied_probs_sum_with_vig(self):
        w, l, v = compute_implied(1.50, 3.20)
        assert w + l == pytest.approx(1.0 + v)

    def test_rejects_odds_at_or_below_one(self):
        with pytest.raises(ValueError, match="odds must be > 1.0"):
            compute_implied(1.0, 2.0)
        with pytest.raises(ValueError):
            compute_implied(2.0, 0.99)
        with pytest.raises(ValueError):
            compute_implied(0.5, 0.5)

    def test_extreme_favourite(self):
        # 1.05 / 11.00 -- heavy favourite, plausible book quote
        w, l, v = compute_implied(1.05, 11.00)
        assert w > 0.95
        assert l < 0.10
        assert v > 0


# =============================================================================
# Metadata extraction
# =============================================================================

def _atp_sample_row(**overrides) -> pd.Series:
    base = {
        "Date":       pd.Timestamp("2024-06-09"),
        "Winner":     "Alcaraz C.",
        "Loser":      "Zverev A.",
        "Tournament": "French Open",
        "Series":     "Grand Slam",
        "Court":      "Outdoor",
        "Surface":    "Clay",
        "Round":      "The Final",
        "Best of":    5,
        "WRank":      3,
        "LRank":      4,
        "Comment":    "Completed",
    }
    base.update(overrides)
    return pd.Series(base)


class TestExtractMatchMetadata:
    def test_atp_basic(self):
        meta = extract_match_metadata(_atp_sample_row(), tour="ATP", year=2024)
        assert meta is not None
        assert meta["winner_name"] == "Alcaraz C."
        assert meta["loser_name"] == "Zverev A."
        assert meta["match_date"] == date(2024, 6, 9)
        assert meta["series_or_tier"] == "Grand Slam"
        assert meta["surface"] == "Clay"
        assert meta["best_of"] == 5
        assert meta["tour"] == "ATP"
        assert meta["source_year"] == 2024
        assert meta["source"] == "tennis-data.co.uk"

    def test_missing_winner_returns_none(self):
        row = _atp_sample_row(Winner=None)
        assert extract_match_metadata(row, tour="ATP", year=2024) is None

    def test_missing_loser_returns_none(self):
        row = _atp_sample_row(Loser=None)
        assert extract_match_metadata(row, tour="ATP", year=2024) is None

    def test_missing_date_returns_none(self):
        row = _atp_sample_row(Date=None)
        assert extract_match_metadata(row, tour="ATP", year=2024) is None

    def test_nan_date_returns_none(self):
        row = _atp_sample_row(Date=pd.NaT)
        assert extract_match_metadata(row, tour="ATP", year=2024) is None

    def test_wta_uses_tier_when_series_absent(self):
        row = _atp_sample_row().drop(labels=["Series"])
        row["Tier"] = "Grand Slam"
        meta = extract_match_metadata(row, tour="WTA", year=2024)
        assert meta is not None
        assert meta["series_or_tier"] == "Grand Slam"
        assert meta["tour"] == "WTA"

    def test_unranked_player_handled(self):
        row = _atp_sample_row(WRank="NR")  # tennis-data uses literal "NR" sometimes
        meta = extract_match_metadata(row, tour="ATP", year=2024)
        assert meta is not None
        assert meta["winner_rank"] is None  # coerced to None

    def test_empty_tournament_falls_back_to_unknown(self):
        row = _atp_sample_row(Tournament=None)
        meta = extract_match_metadata(row, tour="ATP", year=2024)
        assert meta is not None
        assert meta["tournament_name"] == "UNKNOWN"


# =============================================================================
# Odds extraction pipeline
# =============================================================================

def _sample_df_two_matches() -> pd.DataFrame:
    """One match with 4 books, one match with PS only."""
    return pd.DataFrame([
        {
            "Date":       pd.Timestamp("2024-06-09"),
            "Winner":     "Alcaraz C.", "Loser": "Zverev A.",
            "Tournament": "French Open", "Series": "Grand Slam",
            "Court":      "Outdoor", "Surface": "Clay", "Round": "The Final",
            "Best of":    5, "WRank": 3, "LRank": 4, "Comment": "Completed",
            "PSW":  1.85, "PSL":  2.00,
            "B365W": 1.80, "B365L": 2.05,
            "AvgW":  1.82, "AvgL":  2.02,
            "MaxW":  1.95, "MaxL":  2.10,
        },
        {
            "Date":       pd.Timestamp("2024-06-08"),
            "Winner":     "Djokovic N.", "Loser": "Sinner J.",
            "Tournament": "French Open", "Series": "Grand Slam",
            "Court":      "Outdoor", "Surface": "Clay", "Round": "Semifinal",
            "Best of":    5, "WRank": 1, "LRank": 2, "Comment": "Completed",
            "PSW":  1.60, "PSL":  2.40,
        },
    ])


class TestExtractOddsRows:
    def test_emits_one_row_per_match_bookmaker(self):
        df = _sample_df_two_matches()
        rows, report = extract_odds_rows(df, tour="ATP", year=2024)
        # Match 1: PS, B365, AVG, MAX = 4 rows. Match 2: PS only = 1 row. Total = 5.
        assert len(rows) == 5
        assert report.raw_matches == 2
        assert report.skipped_matches == 0
        assert report.pinnacle_rows == 2
        assert report.bet365_rows == 1
        assert report.avg_rows == 1

    def test_pinnacle_row_has_correct_implied_probs(self):
        df = _sample_df_two_matches()
        rows, _ = extract_odds_rows(df, tour="ATP", year=2024)
        ps_rows = [r for r in rows if r.bookmaker_code == "PS"]
        match1_ps = next(r for r in ps_rows if r.winner_name == "Alcaraz C.")
        assert match1_ps.winner_odds == 1.85
        assert match1_ps.loser_odds == 2.00
        assert match1_ps.winner_implied_prob == pytest.approx(1 / 1.85)
        assert match1_ps.loser_implied_prob == pytest.approx(1 / 2.00)
        assert match1_ps.vig == pytest.approx(1 / 1.85 + 1 / 2.00 - 1.0)

    def test_skips_row_with_no_odds_columns_at_all(self):
        df = pd.DataFrame([{
            "Date":   pd.Timestamp("2024-01-01"),
            "Winner": "A.", "Loser": "B.", "Tournament": "X",
            # No odds columns
        }])
        rows, report = extract_odds_rows(df, tour="ATP", year=2024)
        assert rows == []
        assert report.skipped_matches == 1

    def test_skips_row_with_missing_essentials(self):
        df = pd.DataFrame([{
            "Date":   pd.Timestamp("2024-01-01"),
            "Winner": None, "Loser": "B.", "Tournament": "X",
            "PSW":    1.85, "PSL": 2.00,
        }])
        rows, report = extract_odds_rows(df, tour="ATP", year=2024)
        assert rows == []
        assert report.skipped_matches == 1

    def test_invalid_odds_increments_parse_errors(self):
        df = pd.DataFrame([{
            "Date":   pd.Timestamp("2024-01-01"),
            "Winner": "A.", "Loser": "B.", "Tournament": "X",
            "PSW":    0.95,   # impossible (< 1.0)
            "PSL":    2.00,
            "B365W":  1.80,   # valid -- this one should produce a row
            "B365L":  2.05,
        }])
        rows, report = extract_odds_rows(df, tour="ATP", year=2024)
        # B365 row inserts; PS row rejected
        assert len(rows) == 1
        assert rows[0].bookmaker_code == "B365"
        assert report.parse_errors == 1

    def test_handles_missing_bookmaker_columns_silently(self):
        # Older years may lack some bookmakers -- shouldn't crash.
        df = pd.DataFrame([{
            "Date":   pd.Timestamp("2003-05-01"),
            "Winner": "Federer R.", "Loser": "Hewitt L.",
            "Tournament": "Hamburg Masters", "Series": "Masters",
            "PSW":    1.50, "PSL": 2.60,
            # B365, AvgW, MaxW etc. all missing
        }])
        rows, report = extract_odds_rows(df, tour="ATP", year=2003)
        assert len(rows) == 1
        assert rows[0].bookmaker_code == "PS"
        assert report.bet365_rows == 0
        assert report.avg_rows == 0

    def test_oddsrow_is_immutable(self):
        df = _sample_df_two_matches()
        rows, _ = extract_odds_rows(df, tour="ATP", year=2024)
        with pytest.raises((AttributeError, Exception)):
            rows[0].winner_odds = 999.0  # frozen dataclass

    def test_vig_warning_counter_increments(self):
        # PSW/PSL with huge vig (>4%) should trip the Pinnacle range warning.
        df = pd.DataFrame([{
            "Date":   pd.Timestamp("2024-01-01"),
            "Winner": "A.", "Loser": "B.", "Tournament": "X",
            "PSW":    1.50,
            "PSL":    1.80,
            # vig = 1/1.5 + 1/1.8 - 1 = 0.6667 + 0.5556 - 1 = 0.2222 (way too high)
        }])
        rows, report = extract_odds_rows(df, tour="ATP", year=2024)
        assert len(rows) == 1
        assert report.vig_warnings == 1


# =============================================================================
# CLI: year-spec parsing
# =============================================================================

class TestParseYearSpec:
    def test_single_year(self):
        assert parse_year_spec("2024") == [2024]

    def test_range(self):
        assert parse_year_spec("2003-2007") == [2003, 2004, 2005, 2006, 2007]

    def test_comma_list(self):
        assert parse_year_spec("2020,2022,2024") == [2020, 2022, 2024]

    def test_mixed_range_and_comma(self):
        assert parse_year_spec("2020-2022,2024") == [2020, 2021, 2022, 2024]

    def test_dedupes_and_sorts(self):
        assert parse_year_spec("2024,2020-2022,2021") == [2020, 2021, 2022, 2024]

    def test_handles_whitespace(self):
        assert parse_year_spec(" 2020 , 2022 ") == [2020, 2022]

    def test_rejects_invalid_token(self):
        import click as _click
        with pytest.raises(_click.BadParameter):
            parse_year_spec("not-a-year")

    def test_rejects_backwards_range(self):
        import click as _click
        with pytest.raises(_click.BadParameter):
            parse_year_spec("2024-2020")


# =============================================================================
# Schema sanity: bookmaker codes match what we expect in the DB
# =============================================================================

class TestBookmakerCodes:
    def test_pinnacle_present(self):
        # The benchmark book -- must be present or all CLV work is broken.
        assert "PS" in BOOKMAKER_COLUMNS
        assert BOOKMAKER_COLUMNS["PS"] == ("PSW", "PSL")

    def test_bet365_present(self):
        # Second-most-important book (widest coverage in dataset).
        assert "B365" in BOOKMAKER_COLUMNS
        assert BOOKMAKER_COLUMNS["B365"] == ("B365W", "B365L")

    def test_pseudo_bookmakers_present(self):
        assert "AVG" in BOOKMAKER_COLUMNS
        assert "MAX" in BOOKMAKER_COLUMNS

    def test_all_codes_have_w_and_l_columns(self):
        # Every code maps to a (winner_col, loser_col) pair, both non-empty.
        for code, (w, l) in BOOKMAKER_COLUMNS.items():
            assert w and l, f"{code} missing column"
            assert w != l, f"{code} has identical W/L columns"
