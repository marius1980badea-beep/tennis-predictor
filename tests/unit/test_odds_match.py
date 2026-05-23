"""Unit tests for v2 fuzzy matching (player-ID pivot)."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from tennis_predictor.data.matching.odds_match import (
    PlayerResolver,
    TOURNAMENT_ALIASES,
    _bucket_label,
    compact_sackmann_name,
    normalize,
    resolve_via_player_ids,
    tournament_similarity,
)


# =============================================================================
# Normalisation (carried over from v1)
# =============================================================================

class TestNormalize:
    def test_lowercases(self):
        assert normalize("Roland Garros") == "roland garros"

    def test_strips_dots(self):
        assert normalize("Djokovic N.") == "djokovic n"

    def test_strips_diacritics(self):
        assert normalize("Garín C.") == "garin c"

    def test_dashes_become_spaces(self):
        assert normalize("Pierre-Hugues Herbert") == "pierre hugues herbert"

    def test_handles_none(self):
        assert normalize(None) == ""


class TestCompactSackmann:
    def test_two_token_name(self):
        assert compact_sackmann_name("Novak Djokovic") == "djokovic n"

    def test_multi_token_surname(self):
        assert compact_sackmann_name("Roberto Bautista Agut") == "bautista agut r"

    def test_handles_diacritics(self):
        assert compact_sackmann_name("Cristian Garín") == "garin c"


class TestTournamentSimilarity:
    def test_perfect_match(self):
        assert tournament_similarity("French Open", "French Open") == pytest.approx(1.0)

    def test_alias_french_open(self):
        assert tournament_similarity("French Open", "Roland Garros") == pytest.approx(1.0)

    def test_alias_atp_finals(self):
        sim = tournament_similarity("ATP Finals", "Nitto ATP Finals")
        assert sim == pytest.approx(1.0)


class TestTournamentAliases:
    def test_all_aliases_normalized(self):
        for group in TOURNAMENT_ALIASES:
            for name in group:
                assert name == normalize(name), \
                    f"alias not normalized: {name!r}"


# =============================================================================
# PlayerResolver
# =============================================================================

def _players_df(rows: list[tuple[int | str, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"player_id": pid, "name_full": name} for pid, name in rows]
    )


class TestPlayerResolver:
    def test_exact_match_djokovic(self):
        df = _players_df([(1, "Novak Djokovic")])
        r = PlayerResolver(df)
        hit = r.resolve("Djokovic N.")
        assert hit is not None
        assert hit.player_id == 1
        assert hit.matched_via == "exact"
        assert hit.confidence == 1.0

    def test_exact_match_multi_word_surname(self):
        df = _players_df([(42, "Roberto Bautista Agut")])
        r = PlayerResolver(df)
        hit = r.resolve("Bautista Agut R.")
        assert hit is not None
        assert hit.player_id == 42
        assert hit.confidence == 1.0

    def test_exact_match_diacritics_in_full_name(self):
        df = _players_df([(7, "Cristian Garín")])
        r = PlayerResolver(df)
        hit = r.resolve("Garin C.")  # td name without diacritic
        assert hit is not None
        assert hit.player_id == 7

    def test_exact_match_diacritics_in_td_name(self):
        df = _players_df([(7, "Cristian Garin")])  # Sackmann without diacritic
        r = PlayerResolver(df)
        hit = r.resolve("Garín C.")
        assert hit is not None
        assert hit.player_id == 7

    def test_miss_returns_none(self):
        df = _players_df([(1, "Novak Djokovic")])
        r = PlayerResolver(df)
        assert r.resolve("Federer R.") is None

    def test_cached_lookups(self):
        # Repeat resolutions hit cache, not the underlying lookup
        df = _players_df([(1, "Novak Djokovic")])
        r = PlayerResolver(df)
        h1 = r.resolve("Djokovic N.")
        h2 = r.resolve("Djokovic N.")
        assert h1 is h2  # same object from cache

    def test_string_player_id_preserved(self):
        # Sackmann uses "atp_12345" style IDs. Make sure we don't break them.
        df = _players_df([("atp_999", "Novak Djokovic")])
        r = PlayerResolver(df)
        hit = r.resolve("Djokovic N.")
        assert hit is not None
        # Resolver stores as int if possible, otherwise raw string
        assert hit.player_id in (999, "atp_999")

    def test_fuzzy_fallback_when_exact_misses(self):
        # Slight typo case
        df = _players_df([(1, "Carlos Alcaraz Garfia")])
        r = PlayerResolver(df, fuzzy_threshold=0.7)
        # "Alcaraz C." -> compact "alcaraz c"
        # Sackmann full -> compact "alcaraz garfia c"
        # These overlap on "alcaraz" and "c" -- should fuzzy-match
        hit = r.resolve("Alcaraz C.")
        assert hit is not None
        assert hit.player_id == 1
        assert hit.matched_via in ("exact", "fuzzy")

    def test_fuzzy_rejects_too_different(self):
        df = _players_df([(1, "Novak Djokovic")])
        r = PlayerResolver(df, fuzzy_threshold=0.85)
        # "Random Person" is not similar enough
        assert r.resolve("Smith J.") is None

    def test_compact_name_collision_keeps_first(self):
        # Both Marko Djokovic and Mihailo Djokovic compact to "djokovic m"
        df = _players_df([
            (100, "Marko Djokovic"),
            (200, "Mihailo Djokovic"),
        ])
        r = PlayerResolver(df)
        hit = r.resolve("Djokovic M.")
        assert hit is not None
        assert hit.player_id == 100  # first one wins

    def test_handles_empty_name_full(self):
        df = pd.DataFrame([
            {"player_id": 1, "name_full": "Novak Djokovic"},
            {"player_id": 2, "name_full": None},
            {"player_id": 3, "name_full": ""},
        ])
        r = PlayerResolver(df)
        # Just verify it builds without crash and finds Djokovic
        assert r.resolve("Djokovic N.").player_id == 1
        assert len(r) == 1  # only the valid one was indexed


# =============================================================================
# resolve_via_player_ids
# =============================================================================

def _matches_df(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if not df.empty:
        df["match_date"] = df["match_date"].apply(
            lambda d: d if isinstance(d, date) else pd.Timestamp(d).date()
        )
    return df


def _resolver_for(player_rows: list[tuple[int, str]]) -> PlayerResolver:
    return PlayerResolver(_players_df(player_rows))


class TestResolveViaPlayerIds:
    def test_direct_match_in_window(self):
        resolver = _resolver_for([
            (1, "Novak Djokovic"),
            (2, "Rafael Nadal"),
        ])
        matches = _matches_df([
            {"match_id": 100, "winner_id": 1, "loser_id": 2,
             "match_date": date(2024, 6, 9), "tournament_name": "Roland Garros"},
        ])
        result = resolve_via_player_ids(
            td_winner="Djokovic N.", td_loser="Nadal R.",
            td_tournament="French Open", td_date=date(2024, 6, 9),
            matches_df=matches, player_resolver=resolver,
        )
        assert result is not None
        assert result.match_id == 100
        assert result.date_diff_days == 0
        assert result.confidence == 1.0

    def test_match_with_tournament_start_date_offset(self):
        # The CORE FIX: Sackmann date is 5 days before actual play date.
        # v1 (±1 day window) would fail this. v2 (±14 days) succeeds.
        resolver = _resolver_for([
            (1, "Novak Djokovic"),
            (2, "Rafael Nadal"),
        ])
        matches = _matches_df([
            {"match_id": 100, "winner_id": 1, "loser_id": 2,
             "match_date": date(2024, 6, 3),  # Sackmann: tournament start
             "tournament_name": "Roland Garros"},
        ])
        result = resolve_via_player_ids(
            td_winner="Djokovic N.", td_loser="Nadal R.",
            td_tournament="French Open",
            td_date=date(2024, 6, 9),  # tennis-data: actual final date
            matches_df=matches, player_resolver=resolver,
            date_window_days=14,
        )
        assert result is not None
        assert result.match_id == 100
        assert result.date_diff_days == 6

    def test_outside_window_returns_none(self):
        resolver = _resolver_for([
            (1, "Novak Djokovic"),
            (2, "Rafael Nadal"),
        ])
        matches = _matches_df([
            {"match_id": 100, "winner_id": 1, "loser_id": 2,
             "match_date": date(2024, 5, 1),
             "tournament_name": "Madrid"},
        ])
        result = resolve_via_player_ids(
            td_winner="Djokovic N.", td_loser="Nadal R.",
            td_tournament="French Open", td_date=date(2024, 6, 9),
            matches_df=matches, player_resolver=resolver,
            date_window_days=14,
        )
        assert result is None  # 39 days off, well outside window

    def test_swapped_winner_loser_still_matches(self):
        # Defensive: if for some reason tennis-data has swapped W/L
        resolver = _resolver_for([
            (1, "Novak Djokovic"),
            (2, "Rafael Nadal"),
        ])
        matches = _matches_df([
            {"match_id": 100, "winner_id": 2, "loser_id": 1,  # Nadal won in Sackmann
             "match_date": date(2024, 6, 9),
             "tournament_name": "Roland Garros"},
        ])
        result = resolve_via_player_ids(
            td_winner="Djokovic N.",  # tennis-data has Djokovic winning
            td_loser="Nadal R.",
            td_tournament="French Open", td_date=date(2024, 6, 9),
            matches_df=matches, player_resolver=resolver,
        )
        assert result is not None
        assert result.match_id == 100

    def test_unresolvable_player_returns_none(self):
        resolver = _resolver_for([(1, "Novak Djokovic")])  # no Nadal
        matches = _matches_df([])
        result = resolve_via_player_ids(
            td_winner="Djokovic N.", td_loser="Nadal R.",
            td_tournament="French Open", td_date=date(2024, 6, 9),
            matches_df=matches, player_resolver=resolver,
        )
        assert result is None

    def test_multiple_candidates_tournament_tiebreak(self):
        # Same two players, 2 matches in window. Tournament name should disambiguate.
        resolver = _resolver_for([
            (1, "Novak Djokovic"),
            (2, "Rafael Nadal"),
        ])
        matches = _matches_df([
            {"match_id": 100, "winner_id": 1, "loser_id": 2,
             "match_date": date(2024, 6, 1),
             "tournament_name": "Some Other Event"},
            {"match_id": 200, "winner_id": 1, "loser_id": 2,
             "match_date": date(2024, 6, 5),
             "tournament_name": "Roland Garros"},  # this one should win
        ])
        result = resolve_via_player_ids(
            td_winner="Djokovic N.", td_loser="Nadal R.",
            td_tournament="French Open", td_date=date(2024, 6, 9),
            matches_df=matches, player_resolver=resolver,
            date_window_days=14,
        )
        assert result is not None
        assert result.match_id == 200

    def test_no_match_in_window_distinct_from_no_candidates(self):
        # Players resolve, but there's no Sackmann match between them
        resolver = _resolver_for([
            (1, "Novak Djokovic"),
            (2, "Rafael Nadal"),
        ])
        matches = _matches_df([
            # A match exists, but between different players
            {"match_id": 999, "winner_id": 99, "loser_id": 88,
             "match_date": date(2024, 6, 9),
             "tournament_name": "Roland Garros"},
        ])
        result = resolve_via_player_ids(
            td_winner="Djokovic N.", td_loser="Nadal R.",
            td_tournament="French Open", td_date=date(2024, 6, 9),
            matches_df=matches, player_resolver=resolver,
        )
        assert result is None


# =============================================================================
# Bucket labels
# =============================================================================

class TestBucketLabel:
    def test_perfect(self):
        assert _bucket_label(1.0) == "1.00"

    def test_high(self):
        assert _bucket_label(0.97) == "0.95-0.99"

    def test_mid(self):
        assert _bucket_label(0.88) == "0.85-0.94"

    def test_low(self):
        assert _bucket_label(0.75) == "0.70-0.84"

    def test_below_threshold(self):
        assert _bucket_label(0.55) == "below_0.70"
