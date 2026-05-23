"""Fuzzy matching between tennis-data.co.uk odds and Sackmann ``matches``.

**v2 strategy: pivot via player IDs.**

The original v1 strategy compared player names per candidate match, with a
small (±1 day) date window. Diagnosis on real 2013 ATP data showed why this
broke: Sackmann assigns the *tournament start date* to all matches in a
tournament, while tennis-data.co.uk uses the *actual match date*. A 1-day
window misses most matches because the dates can differ by 5-10 days.

v2 fixes this by matching in two phases:

  Phase 1 (one-time): build a compact-name -> player_id lookup from `players`.
                      Resolves "Djokovic N." -> 12345 via direct dict hit or
                      fuzzy fallback (rapidfuzz.process.extractOne).

  Phase 2 (per row):  resolve winner_id and loser_id, then look up matches
                      with exact (winner_id, loser_id) AND date in a wide
                      window (±14 days). If multiple hits, tiebreak with
                      tournament_name similarity and date proximity.

Benefits:
  - Exact ID match eliminates name false-positives.
  - Wide date window handles the Sackmann/tennis-data convention mismatch.
  - Resolver is cached, so common players ("Djokovic N.") resolve once
    across thousands of rows.
  - Performance: O(rows × log players) instead of O(rows × candidates × players).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tournament aliases (used as a tiebreaker only -- main signal is player IDs)
# ---------------------------------------------------------------------------

TOURNAMENT_ALIASES: list[set[str]] = [
    {"french open", "roland garros", "internationaux de france"},
    {"australian open", "aus open"},
    {"wimbledon", "the championships"},
    {"atp finals", "tour finals", "world tour finals",
     "year end championships",
     "barclays atp world tour finals", "nitto atp finals"},
    {"masters cup", "tennis masters cup"},
    {"grand prix hassan ii", "moroccan open"},
]


# ---------------------------------------------------------------------------
# Name normalisation
# ---------------------------------------------------------------------------

def normalize(text_value: str | None) -> str:
    """Canonical form: lower-case, no diacritics, no dots/dashes, single spaces.

    >>> normalize("Bautista Agut R.")
    'bautista agut r'
    >>> normalize("Garín C.")
    'garin c'
    """
    if text_value is None:
        return ""
    s = unidecode(str(text_value))
    s = s.lower()
    for ch in (".", ",", "'", "`"):
        s = s.replace(ch, "")
    s = s.replace("-", " ")
    return " ".join(s.split())


def compact_sackmann_name(full_name: str | None) -> str:
    """Convert Sackmann ``name_full`` to compact form matching tennis-data.

    >>> compact_sackmann_name("Novak Djokovic")
    'djokovic n'
    >>> compact_sackmann_name("Roberto Bautista Agut")
    'bautista agut r'
    """
    parts = normalize(full_name).split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    first_initial = parts[0][0]
    last_name = " ".join(parts[1:])
    return f"{last_name} {first_initial}"


def tournament_similarity(td_name: str, sackmann_name: str) -> float:
    """Tournament name similarity with alias support."""
    td_norm = normalize(td_name)
    sk_norm = normalize(sackmann_name)
    if not td_norm or not sk_norm:
        return 0.0
    for group in TOURNAMENT_ALIASES:
        if td_norm in group and sk_norm in group:
            return 1.0
    return fuzz.token_set_ratio(td_norm, sk_norm) / 100.0


# ---------------------------------------------------------------------------
# Player resolver: tennis-data name -> Sackmann player_id
# ---------------------------------------------------------------------------

@dataclass
class PlayerLookupHit:
    """Result of resolving one tennis-data player name."""
    player_id: int
    sackmann_full: str
    confidence: float       # 1.0 for exact compact match, <1.0 for fuzzy
    matched_via: str        # "exact" | "fuzzy" | "miss"


class PlayerResolver:
    """O(1) resolver from tennis-data.co.uk compact name to Sackmann player_id.

    Built once per tour, then queried per odds row. Internal cache keyed by
    raw tennis-data string makes repeat lookups (very common: "Djokovic N."
    appears in hundreds of matches) cost nothing.
    """

    def __init__(
        self,
        players_df: pd.DataFrame,
        fuzzy_threshold: float = 0.85,
    ) -> None:
        """Build resolver from a DataFrame with columns: player_id, name_full."""
        self._fuzzy_threshold = fuzzy_threshold
        self._lookup: dict[str, tuple[int, str]] = {}
        self._lookup_keys: list[str] = []  # rapidfuzz prefers list over dict.keys()
        self._cache: dict[str, Optional[PlayerLookupHit]] = {}
        self._build(players_df)

    def _build(self, players_df: pd.DataFrame) -> None:
        for _, row in players_df.iterrows():
            full_name = row["name_full"]
            if full_name is None or pd.isna(full_name):
                continue
            compact = compact_sackmann_name(str(full_name))
            if not compact:
                continue
            pid = int(row["player_id"]) if "player_id" not in row or not isinstance(row["player_id"], str) else row["player_id"]
            # Note: player_id can be string in Sackmann ("atp_12345"). Cast appropriately.
            try:
                pid_val: object = int(row["player_id"])
            except (TypeError, ValueError):
                pid_val = row["player_id"]
            # On collisions (same compact form for two players), keep the first
            # seen and log a warning. Collisions are rare but possible (e.g.
            # "Marko Djokovic" and "Mihailo Djokovic" both -> "djokovic m").
            if compact not in self._lookup:
                self._lookup[compact] = (pid_val, str(full_name))
                self._lookup_keys.append(compact)
            else:
                logger.debug("compact name collision: %r already maps to %s; ignoring %s (%s)",
                             compact, self._lookup[compact][1], full_name, pid_val)

    def __len__(self) -> int:
        return len(self._lookup)

    def resolve(self, td_name: str) -> Optional[PlayerLookupHit]:
        """Resolve a tennis-data player name to a Sackmann player_id.

        Returns ``None`` if no candidate scores at or above the fuzzy threshold.
        Result is cached: each unique raw td_name is resolved only once.
        """
        if td_name in self._cache:
            return self._cache[td_name]
        result = self._resolve_impl(td_name)
        self._cache[td_name] = result
        return result

    def _resolve_impl(self, td_name: str) -> Optional[PlayerLookupHit]:
        normalized = normalize(td_name)
        if not normalized:
            return None

        # 1) Exact compact match (covers ~99% of cases per offline testing)
        if normalized in self._lookup:
            pid, full = self._lookup[normalized]
            return PlayerLookupHit(
                player_id=pid, sackmann_full=full,
                confidence=1.0, matched_via="exact",
            )

        # 2) Fuzzy fallback via rapidfuzz.process.extractOne (C-accelerated)
        if not self._lookup_keys:
            return None
        match = process.extractOne(
            normalized,
            self._lookup_keys,
            scorer=fuzz.token_set_ratio,
            score_cutoff=self._fuzzy_threshold * 100,
        )
        if match is None:
            return None
        matched_key, score, _ = match
        pid, full = self._lookup[matched_key]
        return PlayerLookupHit(
            player_id=pid, sackmann_full=full,
            confidence=score / 100.0, matched_via="fuzzy",
        )


# ---------------------------------------------------------------------------
# Match resolution via player IDs
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MatchResult:
    """Best match found for one tennis-data identity."""
    match_id: int
    confidence: float
    winner_resolution: str          # "exact" | "fuzzy"
    loser_resolution: str
    winner_sim: float               # confidence of winner_id resolution
    loser_sim: float                # confidence of loser_id resolution
    tournament_sim: float           # tournament name similarity (tiebreaker only)
    date_diff_days: int


def resolve_via_player_ids(
    td_winner: str,
    td_loser: str,
    td_tournament: str,
    td_date: date,
    matches_df: pd.DataFrame,
    player_resolver: PlayerResolver,
    *,
    date_window_days: int = 14,
    min_confidence: float = 0.70,
) -> Optional[MatchResult]:
    """Find the Sackmann match for one tennis-data identity.

    ``matches_df`` must have columns: ``match_id``, ``winner_id``, ``loser_id``,
    ``match_date``, ``tournament_name`` (the JOIN with tournaments must already
    have been performed before calling).

    Returns ``None`` if:
      - Either player can't be resolved at fuzzy_threshold
      - No Sackmann match has those two player IDs within the date window
      - Composite confidence falls below ``min_confidence``
    """
    w_hit = player_resolver.resolve(td_winner)
    l_hit = player_resolver.resolve(td_loser)

    if w_hit is None or l_hit is None:
        return None

    # Resolution confidence = average of both player confidences.
    # A high value here is what makes the player-ID pivot reliable.
    resolution_conf = (w_hit.confidence + l_hit.confidence) / 2.0

    if resolution_conf < min_confidence:
        return None

    # Find matches with exact (winner_id, loser_id). Quick boolean mask in pandas.
    mask_direct = (
        (matches_df["winner_id"] == w_hit.player_id) &
        (matches_df["loser_id"] == l_hit.player_id)
    )
    candidates = matches_df[mask_direct]

    # Defensive: try swapped roles in case staging has reversed winner/loser
    # (shouldn't happen given our loader, but cheap to check)
    if candidates.empty:
        mask_swapped = (
            (matches_df["winner_id"] == l_hit.player_id) &
            (matches_df["loser_id"] == w_hit.player_id)
        )
        candidates = matches_df[mask_swapped]
        if not candidates.empty:
            logger.debug("matched via swapped W/L for %s vs %s on %s",
                         td_winner, td_loser, td_date)

    if candidates.empty:
        return None

    # Apply date window
    window = timedelta(days=date_window_days)
    in_window = candidates[
        candidates["match_date"].apply(lambda d: abs((d - td_date).days)) <= date_window_days
    ]
    if in_window.empty:
        return None

    candidates = in_window

    # Single hit: done
    if len(candidates) == 1:
        best = candidates.iloc[0]
    else:
        # Multiple matches between same two players in window (very rare:
        # e.g. Davis Cup tie + ATP event in same week, or two best-of-3 sets
        # on consecutive days). Tiebreak:
        #   1) Highest tournament name similarity
        #   2) Smallest date_diff
        scored = candidates.copy()
        scored["t_sim"] = scored["tournament_name"].apply(
            lambda t: tournament_similarity(td_tournament, str(t))
        )
        scored["d_diff"] = scored["match_date"].apply(
            lambda d: abs((d - td_date).days)
        )
        scored = scored.sort_values(["t_sim", "d_diff"], ascending=[False, True])
        best = scored.iloc[0]

    date_diff = abs((best["match_date"] - td_date).days)
    t_sim = tournament_similarity(td_tournament, str(best["tournament_name"]))

    return MatchResult(
        match_id=int(best["match_id"]),
        confidence=round(resolution_conf, 4),
        winner_resolution=w_hit.matched_via,
        loser_resolution=l_hit.matched_via,
        winner_sim=round(w_hit.confidence, 4),
        loser_sim=round(l_hit.confidence, 4),
        tournament_sim=round(t_sim, 4),
        date_diff_days=int(date_diff),
    )


# ---------------------------------------------------------------------------
# Reporting helpers (used by CLI)
# ---------------------------------------------------------------------------

@dataclass
class MatchReport:
    """Aggregate stats from a matching run."""
    tour: str
    total_unique_identities: int = 0
    matched: int = 0
    unmatched_player_resolve: int = 0    # at least one player couldn't be resolved
    unmatched_no_match_in_window: int = 0  # players resolved but no match found
    unmatched_below_confidence: int = 0   # resolved but composite too low
    rows_updated: int = 0
    # Diagnostics
    fuzzy_winner_resolutions: int = 0
    fuzzy_loser_resolutions: int = 0
    confidence_buckets: dict[str, int] = field(default_factory=lambda: {
        "1.00":       0,
        "0.95-0.99":  0,
        "0.85-0.94":  0,
        "0.70-0.84":  0,
        "below_0.70": 0,
    })


def _bucket_label(confidence: float) -> str:
    if confidence >= 1.0:
        return "1.00"
    if confidence >= 0.95:
        return "0.95-0.99"
    if confidence >= 0.85:
        return "0.85-0.94"
    if confidence >= 0.70:
        return "0.70-0.84"
    return "below_0.70"
