"""Sackmann tennis dataset loader.

Loads ATP and WTA match data from Jeff Sackmann's GitHub repositories:
- https://github.com/JeffSackmann/tennis_atp
- https://github.com/JeffSackmann/tennis_wta

Data format reference:
- Match files: {tour}_matches_{YYYY}.csv (one per year)
- Players file: {tour}_players.csv
- Rankings file: {tour}_rankings_{decade}s.csv

Key columns in match CSVs:
- tourney_id, tourney_name, surface, draw_size, tourney_level, tourney_date
- match_num, winner_id, winner_name, ..., loser_id, loser_name, ...
- score, best_of, round, minutes
- w_ace, w_df, w_svpt, w_1stIn, w_1stWon, w_2ndWon, w_SvGms, w_bpSaved, w_bpFaced
- l_ace, l_df, l_svpt, l_1stIn, l_1stWon, l_2ndWon, l_SvGms, l_bpSaved, l_bpFaced
- winner_rank, winner_rank_points, loser_rank, loser_rank_points

Tournament levels:
- G = Grand Slam, M = Masters 1000, A = ATP 500, D = ATP 250
- F = Tour Finals, C = Challenger, S = Satellite/ITF, O = Olympics
- PM = WTA Premier Mandatory, P = WTA Premier, I = WTA International
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import pandas as pd

if TYPE_CHECKING:
    # These are only needed for type hints, not at runtime
    pass

# Lightweight stdlib logger as default; replaced by structlog at runtime
import logging

_module_logger = logging.getLogger(__name__)


def _get_logger():
    """Lazy import of structlog logger to avoid pulling deps at module load."""
    try:
        from tennis_predictor.logging_config import get_logger

        return get_logger(__name__)
    except ImportError:
        return _module_logger


logger = _get_logger()

Tour = Literal["ATP", "WTA"]


@dataclass(frozen=True)
class LoadResult:
    """Result of a data load operation."""

    source: str
    rows_processed: int
    rows_inserted: int
    rows_updated: int
    rows_skipped: int
    rows_errored: int
    duration_seconds: float

    @property
    def total_changes(self) -> int:
        return self.rows_inserted + self.rows_updated


class SackmannLoader:
    """Loads tennis data from Sackmann GitHub repositories.

    Workflow:
    1. Clone (or pull) the Sackmann repo to local cache
    2. Read CSV files for the specified year range
    3. Validate and transform data
    4. Bulk upsert into Supabase tables
    5. Log results to data_ingestion_log table
    """

    # Surface normalization map (Sackmann uses these exact strings)
    SURFACE_MAP: dict[str, str] = {
        "Hard": "Hard",
        "Clay": "Clay",
        "Grass": "Grass",
        "Carpet": "Carpet",
    }

    # Tournament level normalization
    # Some years use slightly different codes - we normalize to schema
    LEVEL_MAP: dict[str, str] = {
        "G": "G",   # Grand Slam
        "M": "M",   # Masters 1000 / WTA 1000
        "A": "A",   # ATP 500
        "D": "D",   # ATP 250
        "F": "F",   # Tour Finals
        "C": "C",   # Challenger
        "S": "S",   # Satellite / ITF
        "O": "O",   # Olympics
        "PM": "PM", # WTA Premier Mandatory (legacy)
        "P": "PM",  # WTA Premier (treated as PM)
        "I": "I",   # WTA International (legacy)
        "T1": "PM", # WTA Tier 1
        "T2": "I",  # WTA Tier 2
    }

    def __init__(self, tour: Tour, cache_dir: Path | None = None) -> None:
        """Initialize loader for ATP or WTA data.

        Args:
            tour: 'ATP' or 'WTA'
            cache_dir: Local directory to cache cloned repo. Defaults to ./tennis_{tour}/
        """
        # Lazy imports: keep module-level imports light so unit tests of
        # helper functions don't require sqlalchemy/supabase/etc.
        from tennis_predictor.config import get_settings

        self.tour: Tour = tour
        self.settings = get_settings()
        self.cache_dir = cache_dir or Path(f"tennis_{tour.lower()}")
        self.repo_url = (
            self.settings.data_load.sackmann_atp_repo
            if tour == "ATP"
            else self.settings.data_load.sackmann_wta_repo
        )
        self.player_id_prefix = tour.lower() + "_"

    # ========================================================================
    # Repo management
    # ========================================================================

    def sync_repo(self) -> None:
        """Clone the Sackmann repo, or pull latest if already cloned.

        Retries up to 3 times with exponential backoff on network errors.
        """
        # Lazy retry decorator application
        from tenacity import retry, stop_after_attempt, wait_exponential

        @retry(
            stop=stop_after_attempt(3),
            wait=wait_exponential(multiplier=2, min=2, max=30),
        )
        def _do_sync() -> None:
            if self.cache_dir.exists() and (self.cache_dir / ".git").exists():
                logger.info("pulling_sackmann_repo", path=str(self.cache_dir))
                result = subprocess.run(
                    ["git", "-C", str(self.cache_dir), "pull", "--ff-only"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=120,
                )
                if result.returncode != 0:
                    logger.warning("git_pull_failed", stderr=result.stderr)
            else:
                logger.info(
                    "cloning_sackmann_repo",
                    url=self.repo_url,
                    path=str(self.cache_dir),
                )
                self.cache_dir.parent.mkdir(parents=True, exist_ok=True)
                subprocess.run(
                    [
                        "git",
                        "clone",
                        "--depth",
                        "1",
                        self.repo_url,
                        str(self.cache_dir),
                    ],
                    check=True,
                    timeout=300,
                )

        _do_sync()

    # ========================================================================
    # Players
    # ========================================================================

    def load_players(self) -> LoadResult:
        """Load all players from {tour}_players.csv.

        Players file has columns:
            player_id, name_first, name_last, hand, dob, ioc, height, wikidata_id
        """
        start_time = datetime.now()
        players_file = self.cache_dir / f"{self.tour.lower()}_players.csv"

        if not players_file.exists():
            raise FileNotFoundError(f"Players file not found: {players_file}")

        logger.info("loading_players", file=str(players_file), tour=self.tour)

        df = pd.read_csv(
            players_file,
            dtype={
                "player_id": "string",
                "name_first": "string",
                "name_last": "string",
                "hand": "string",
                "dob": "string",  # YYYYMMDD format, parse manually
                "ioc": "string",
                "height": "Int64",
                "wikidata_id": "string",
            },
        )

        # Transform to our schema
        records = []
        for _, row in df.iterrows():
            if pd.isna(row["player_id"]) or pd.isna(row["name_last"]):
                continue

            record = {
                "player_id": f"{self.player_id_prefix}{row['player_id']}",
                "tour": self.tour,
                "name_first": _safe_str(row.get("name_first")),
                "name_last": _safe_str(row["name_last"]),
                "hand": _normalize_hand(row.get("hand")),
                "birth_date": _parse_yyyymmdd(row.get("dob")),
                "country_code": _safe_str(row.get("ioc")),
                "height_cm": _safe_int(row.get("height")),
            }
            records.append(record)

        # Bulk upsert
        rows_inserted, rows_updated = self._bulk_upsert_players(records)

        duration = (datetime.now() - start_time).total_seconds()
        result = LoadResult(
            source=f"sackmann_{self.tour.lower()}_players",
            rows_processed=len(df),
            rows_inserted=rows_inserted,
            rows_updated=rows_updated,
            rows_skipped=len(df) - len(records),
            rows_errored=0,
            duration_seconds=duration,
        )

        self._log_to_db(result, operation="full_load")
        logger.info(
            "players_loaded",
            tour=self.tour,
            processed=result.rows_processed,
            inserted=result.rows_inserted,
            duration_s=round(duration, 1),
        )
        return result

    def _bulk_upsert_players(self, records: list[dict]) -> tuple[int, int]:
        """Bulk upsert players using ON CONFLICT.

        Returns:
            (rows_inserted, rows_updated)
        """
        if not records:
            return 0, 0

        # Lazy imports (so unit tests of pure helpers don't need DB deps)
        from sqlalchemy import text

        from tennis_predictor.data.storage import get_session

        # Use raw SQL for efficient upsert - SQLAlchemy ORM would be slow for 50k+ rows
        # We use ON CONFLICT DO UPDATE to handle re-runs (players may get updated info)
        upsert_sql = text("""
            INSERT INTO players (
                player_id, tour, name_first, name_last,
                hand, birth_date, country_code, height_cm
            ) VALUES (
                :player_id, :tour, :name_first, :name_last,
                :hand, :birth_date, :country_code, :height_cm
            )
            ON CONFLICT (player_id) DO UPDATE SET
                name_first = EXCLUDED.name_first,
                name_last = EXCLUDED.name_last,
                hand = EXCLUDED.hand,
                birth_date = EXCLUDED.birth_date,
                country_code = EXCLUDED.country_code,
                height_cm = EXCLUDED.height_cm,
                updated_at = NOW()
            RETURNING (xmax = 0) AS inserted
        """)

        rows_inserted = 0
        rows_updated = 0
        batch_size = 1000

        with get_session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                for record in batch:
                    result = session.execute(upsert_sql, record)
                    row = result.fetchone()
                    if row and row.inserted:
                        rows_inserted += 1
                    else:
                        rows_updated += 1

        return rows_inserted, rows_updated

    # ========================================================================
    # Tournaments + Matches
    # ========================================================================

    def load_matches_for_year(self, year: int) -> LoadResult:
        """Load matches for a specific year.

        Loads both tournaments (deduped) and matches from {tour}_matches_{year}.csv.

        Args:
            year: Year to load (e.g. 2024)
        """
        start_time = datetime.now()
        matches_file = self.cache_dir / f"{self.tour.lower()}_matches_{year}.csv"

        if not matches_file.exists():
            logger.warning("matches_file_missing", year=year, file=str(matches_file))
            return LoadResult(
                source=f"sackmann_{self.tour.lower()}_matches_{year}",
                rows_processed=0,
                rows_inserted=0,
                rows_updated=0,
                rows_skipped=0,
                rows_errored=0,
                duration_seconds=0.0,
            )

        logger.info("loading_matches", year=year, tour=self.tour)

        # Read with explicit dtypes to avoid pandas inferring wrong types
        df = pd.read_csv(
            matches_file,
            dtype={
                "tourney_id": "string",
                "tourney_name": "string",
                "surface": "string",
                "draw_size": "Int64",
                "tourney_level": "string",
                "tourney_date": "string",
                "match_num": "Int64",
                "winner_id": "string",
                "winner_seed": "string",  # Can be "WC", "Q", etc.
                "winner_entry": "string",
                "winner_name": "string",
                "winner_hand": "string",
                "winner_ht": "Int64",
                "winner_ioc": "string",
                "winner_age": "Float64",
                "loser_id": "string",
                "loser_seed": "string",
                "loser_entry": "string",
                "loser_name": "string",
                "loser_hand": "string",
                "loser_ht": "Int64",
                "loser_ioc": "string",
                "loser_age": "Float64",
                "score": "string",
                "best_of": "Int64",
                "round": "string",
                "minutes": "Int64",
                "w_ace": "Int64",
                "w_df": "Int64",
                "w_svpt": "Int64",
                "w_1stIn": "Int64",
                "w_1stWon": "Int64",
                "w_2ndWon": "Int64",
                "w_SvGms": "Int64",
                "w_bpSaved": "Int64",
                "w_bpFaced": "Int64",
                "l_ace": "Int64",
                "l_df": "Int64",
                "l_svpt": "Int64",
                "l_1stIn": "Int64",
                "l_1stWon": "Int64",
                "l_2ndWon": "Int64",
                "l_SvGms": "Int64",
                "l_bpSaved": "Int64",
                "l_bpFaced": "Int64",
                "winner_rank": "Int64",
                "winner_rank_points": "Int64",
                "loser_rank": "Int64",
                "loser_rank_points": "Int64",
            },
            low_memory=False,
        )

        rows_processed = len(df)
        rows_errored = 0

        # Step 1: extract and upsert tournaments (deduplicated)
        tournaments = self._extract_tournaments(df, year)
        self._bulk_upsert_tournaments(tournaments)

        # Step 2: prepare match records
        match_records = []
        stats_records_template = []  # (match_source_id, winner_stats, loser_stats)

        for _, row in df.iterrows():
            try:
                match_record, winner_stats, loser_stats = self._build_match_record(row, year)
                if match_record is None:
                    rows_errored += 1
                    continue
                match_records.append(match_record)
                stats_records_template.append(
                    (match_record["source_match_id"], winner_stats, loser_stats)
                )
            except Exception as e:
                rows_errored += 1
                logger.debug("match_row_error", error=str(e), row_idx=_)
                continue

        # Step 3: bulk insert matches and capture generated match_ids
        match_id_map = self._bulk_insert_matches(match_records)

        # Step 4: bulk insert stats using the new match_ids
        stats_records = []
        for source_match_id, winner_stats, loser_stats in stats_records_template:
            match_id = match_id_map.get(source_match_id)
            if match_id is None:
                continue
            if winner_stats:
                stats_records.append({**winner_stats, "match_id": match_id})
            if loser_stats:
                stats_records.append({**loser_stats, "match_id": match_id})

        rows_inserted_stats = self._bulk_insert_match_stats(stats_records)

        duration = (datetime.now() - start_time).total_seconds()
        result = LoadResult(
            source=f"sackmann_{self.tour.lower()}_matches_{year}",
            rows_processed=rows_processed,
            rows_inserted=len(match_id_map),
            rows_updated=0,
            rows_skipped=rows_processed - len(match_id_map) - rows_errored,
            rows_errored=rows_errored,
            duration_seconds=duration,
        )

        self._log_to_db(
            result,
            operation="full_load",
            metadata={"year": year, "stats_inserted": rows_inserted_stats},
        )
        logger.info(
            "matches_loaded",
            year=year,
            tour=self.tour,
            processed=rows_processed,
            inserted=len(match_id_map),
            stats_inserted=rows_inserted_stats,
            errored=rows_errored,
            duration_s=round(duration, 1),
        )
        return result

    def _extract_tournaments(self, df: pd.DataFrame, year: int) -> list[dict]:
        """Extract unique tournaments from the matches dataframe."""
        # Deduplicate by tourney_id within this year
        tourneys = df.drop_duplicates(subset=["tourney_id"]).copy()

        records = []
        for _, row in tourneys.iterrows():
            if pd.isna(row["tourney_id"]):
                continue

            tourney_date = _parse_yyyymmdd(row.get("tourney_date"))

            record = {
                "tournament_id": f"{self.tour.lower()}_{year}_{row['tourney_id']}",
                "tour": self.tour,
                "name": _safe_str(row.get("tourney_name")) or "Unknown",
                "level": self.LEVEL_MAP.get(_safe_str(row.get("tourney_level")) or "", None),
                "surface": self.SURFACE_MAP.get(_safe_str(row.get("surface")) or "", None),
                "indoor": False,  # Sackmann doesn't directly mark this; can be enriched later
                "draw_size": _safe_int(row.get("draw_size")),
                "country_code": None,  # Not in match CSVs; would need separate enrichment
                "city": None,
                "start_date": tourney_date,
                "end_date": tourney_date,  # Approximate; can be refined
            }
            records.append(record)

        return records

    def _bulk_upsert_tournaments(self, records: list[dict]) -> None:
        """Bulk upsert tournaments."""
        if not records:
            return

        from sqlalchemy import text

        from tennis_predictor.data.storage import get_session

        upsert_sql = text("""
            INSERT INTO tournaments (
                tournament_id, tour, name, level, surface, indoor,
                draw_size, country_code, city, start_date, end_date
            ) VALUES (
                :tournament_id, :tour, :name, :level, :surface, :indoor,
                :draw_size, :country_code, :city, :start_date, :end_date
            )
            ON CONFLICT (tournament_id) DO UPDATE SET
                name = EXCLUDED.name,
                level = EXCLUDED.level,
                surface = EXCLUDED.surface,
                draw_size = EXCLUDED.draw_size,
                start_date = EXCLUDED.start_date,
                end_date = EXCLUDED.end_date
        """)

        with get_session() as session:
            for record in records:
                session.execute(upsert_sql, record)

    def _build_match_record(
        self, row: pd.Series, year: int
    ) -> tuple[dict | None, dict | None, dict | None]:
        """Transform a single Sackmann match row into our schema records.

        Returns:
            (match_record, winner_stats, loser_stats)
            Returns (None, None, None) if row is invalid (missing IDs etc.)
        """
        if pd.isna(row.get("winner_id")) or pd.isna(row.get("loser_id")):
            return None, None, None

        winner_id = f"{self.player_id_prefix}{row['winner_id']}"
        loser_id = f"{self.player_id_prefix}{row['loser_id']}"

        if winner_id == loser_id:  # Defensive check
            return None, None, None

        match_date = _parse_yyyymmdd(row.get("tourney_date"))
        if match_date is None:
            return None, None, None

        # Detect retirement / walkover from score string
        score = _safe_str(row.get("score")) or ""
        retirement = "RET" in score
        walkover = "W/O" in score or score.strip() == "W/O"

        # Build composite source_match_id for deduplication
        # Sackmann doesn't have a globally unique match_id; we construct one
        source_match_id = f"{year}_{row['tourney_id']}_{row.get('match_num', 0)}"

        match_record = {
            "tournament_id": f"{self.tour.lower()}_{year}_{row['tourney_id']}",
            "match_date": match_date,
            "tour": self.tour,
            "surface": self.SURFACE_MAP.get(_safe_str(row.get("surface")) or "", None),
            "round": _safe_str(row.get("round")),
            "best_of": _safe_int(row.get("best_of")),
            "winner_id": winner_id,
            "loser_id": loser_id,
            "winner_rank": _safe_int(row.get("winner_rank")),
            "loser_rank": _safe_int(row.get("loser_rank")),
            "winner_rank_points": _safe_int(row.get("winner_rank_points")),
            "loser_rank_points": _safe_int(row.get("loser_rank_points")),
            "winner_seed": _safe_int(row.get("winner_seed")),
            "loser_seed": _safe_int(row.get("loser_seed")),
            "winner_entry": _safe_str(row.get("winner_entry")),
            "loser_entry": _safe_str(row.get("loser_entry")),
            "score": score or None,
            "retirement": retirement,
            "walkover": walkover,
            "minutes": _safe_int(row.get("minutes")),
            "source": "sackmann",
            "source_match_id": source_match_id,
        }

        # Build stats records (only if any serve stats present)
        winner_stats = self._build_stats_record(row, winner_id, prefix="w_", is_winner=True)
        loser_stats = self._build_stats_record(row, loser_id, prefix="l_", is_winner=False)

        return match_record, winner_stats, loser_stats

    def _build_stats_record(
        self, row: pd.Series, player_id: str, prefix: str, is_winner: bool
    ) -> dict | None:
        """Build a match_stats record. Returns None if no stats present."""
        # Quick check: if all serve stats are null, skip
        if pd.isna(row.get(f"{prefix}svpt")) and pd.isna(row.get(f"{prefix}ace")):
            return None

        return {
            "player_id": player_id,
            "is_winner": is_winner,
            "aces": _safe_int(row.get(f"{prefix}ace")),
            "double_faults": _safe_int(row.get(f"{prefix}df")),
            "serve_points": _safe_int(row.get(f"{prefix}svpt")),
            "first_serves_in": _safe_int(row.get(f"{prefix}1stIn")),
            "first_serves_won": _safe_int(row.get(f"{prefix}1stWon")),
            "second_serves_won": _safe_int(row.get(f"{prefix}2ndWon")),
            "service_games": _safe_int(row.get(f"{prefix}SvGms")),
            "break_points_saved": _safe_int(row.get(f"{prefix}bpSaved")),
            "break_points_faced": _safe_int(row.get(f"{prefix}bpFaced")),
        }

    def _bulk_insert_matches(self, records: list[dict]) -> dict[str, int]:
        """Bulk insert matches and return mapping of source_match_id -> match_id.

        Uses ON CONFLICT DO NOTHING to handle duplicates safely (re-runs).
        For re-runs, we need to fetch existing match_ids separately.
        """
        if not records:
            return {}

        from sqlalchemy import text

        from tennis_predictor.data.storage import get_session

        insert_sql = text("""
            INSERT INTO matches (
                tournament_id, match_date, tour, surface, round, best_of,
                winner_id, loser_id, winner_rank, loser_rank,
                winner_rank_points, loser_rank_points,
                winner_seed, loser_seed, winner_entry, loser_entry,
                score, retirement, walkover, minutes,
                source, source_match_id
            ) VALUES (
                :tournament_id, :match_date, :tour, :surface, :round, :best_of,
                :winner_id, :loser_id, :winner_rank, :loser_rank,
                :winner_rank_points, :loser_rank_points,
                :winner_seed, :loser_seed, :winner_entry, :loser_entry,
                :score, :retirement, :walkover, :minutes,
                :source, :source_match_id
            )
            ON CONFLICT (source, source_match_id, tour) DO NOTHING
            RETURNING match_id, source_match_id
        """)

        match_id_map: dict[str, int] = {}
        batch_size = 500

        with get_session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                for record in batch:
                    result = session.execute(insert_sql, record)
                    row = result.fetchone()
                    if row:
                        match_id_map[row.source_match_id] = row.match_id

            # For records that ON CONFLICT skipped, fetch their existing IDs
            missing_source_ids = [
                r["source_match_id"]
                for r in records
                if r["source_match_id"] not in match_id_map
            ]
            if missing_source_ids:
                fetch_sql = text("""
                    SELECT match_id, source_match_id
                    FROM matches
                    WHERE source = 'sackmann'
                      AND tour = :tour
                      AND source_match_id = ANY(:ids)
                """)
                result = session.execute(
                    fetch_sql,
                    {"tour": self.tour, "ids": missing_source_ids},
                )
                for row in result:
                    match_id_map[row.source_match_id] = row.match_id

        return match_id_map

    def _bulk_insert_match_stats(self, records: list[dict]) -> int:
        """Bulk insert match_stats. ON CONFLICT DO NOTHING for re-run safety."""
        if not records:
            return 0

        from sqlalchemy import text

        from tennis_predictor.data.storage import get_session

        insert_sql = text("""
            INSERT INTO match_stats (
                match_id, player_id, is_winner,
                aces, double_faults, serve_points,
                first_serves_in, first_serves_won, second_serves_won,
                service_games, break_points_saved, break_points_faced
            ) VALUES (
                :match_id, :player_id, :is_winner,
                :aces, :double_faults, :serve_points,
                :first_serves_in, :first_serves_won, :second_serves_won,
                :service_games, :break_points_saved, :break_points_faced
            )
            ON CONFLICT (match_id, player_id) DO NOTHING
        """)

        rows_inserted = 0
        batch_size = 1000

        with get_session() as session:
            for i in range(0, len(records), batch_size):
                batch = records[i : i + batch_size]
                result = session.execute(insert_sql, batch)
                rows_inserted += result.rowcount or 0

        return rows_inserted

    # ========================================================================
    # Bulk year loading
    # ========================================================================

    def load_year_range(self, start_year: int, end_year: int) -> list[LoadResult]:
        """Load matches for a range of years.

        Args:
            start_year: Inclusive start year
            end_year: Inclusive end year

        Returns:
            List of LoadResults, one per year.
        """
        results = []
        for year in range(start_year, end_year + 1):
            try:
                result = self.load_matches_for_year(year)
                results.append(result)
            except Exception as e:
                logger.error("year_load_failed", year=year, error=str(e), exc_info=True)
                results.append(
                    LoadResult(
                        source=f"sackmann_{self.tour.lower()}_matches_{year}",
                        rows_processed=0,
                        rows_inserted=0,
                        rows_updated=0,
                        rows_skipped=0,
                        rows_errored=1,
                        duration_seconds=0.0,
                    )
                )
        return results

    # ========================================================================
    # Logging
    # ========================================================================

    def _log_to_db(
        self,
        result: LoadResult,
        operation: str,
        metadata: dict | None = None,
    ) -> None:
        """Insert audit record into data_ingestion_log table."""
        import json

        from sqlalchemy import text

        from tennis_predictor.data.storage import get_session

        insert_sql = text("""
            INSERT INTO data_ingestion_log (
                source, operation, completed_at, status,
                rows_processed, rows_inserted, rows_updated,
                rows_skipped, rows_errored, metadata
            ) VALUES (
                :source, :operation, NOW(), 'completed',
                :rows_processed, :rows_inserted, :rows_updated,
                :rows_skipped, :rows_errored, :metadata
            )
        """)

        with get_session() as session:
            session.execute(
                insert_sql,
                {
                    "source": result.source,
                    "operation": operation,
                    "rows_processed": result.rows_processed,
                    "rows_inserted": result.rows_inserted,
                    "rows_updated": result.rows_updated,
                    "rows_skipped": result.rows_skipped,
                    "rows_errored": result.rows_errored,
                    "metadata": json.dumps(metadata) if metadata else None,
                },
            )


# ============================================================================
# Helper functions
# ============================================================================

def _safe_str(value) -> str | None:
    """Convert value to str, handling NaN/None."""
    if value is None or pd.isna(value):
        return None
    s = str(value).strip()
    return s if s else None


def _safe_int(value) -> int | None:
    """Convert value to int, handling NaN/None/non-numeric seeds like 'WC'."""
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (ValueError, TypeError):
        return None  # Seeds like "WC", "Q" become None


def _normalize_hand(value) -> str | None:
    """Normalize hand to R/L/U."""
    s = _safe_str(value)
    if s is None:
        return None
    s = s.upper()
    if s in ("R", "L", "U"):
        return s
    return None


def _parse_yyyymmdd(value) -> date | None:
    """Parse YYYYMMDD string (Sackmann format) to date."""
    s = _safe_str(value)
    if s is None or len(s) != 8:
        return None
    try:
        return datetime.strptime(s, "%Y%m%d").date()
    except ValueError:
        return None
