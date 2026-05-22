"""Elo state manager: load/save player Elo ratings from/to Supabase.

The Elo Engine itself (elo.py) operates on in-memory PlayerEloState objects.
This module bridges that with the database, handling:
- Loading initial state for a date range
- Batch persisting computed ratings
- Querying current top players by Elo
"""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import TYPE_CHECKING

from sqlalchemy import text

from tennis_predictor.data.storage import get_session
from tennis_predictor.logging_config import get_logger
from tennis_predictor.models.elo import EloConfig, PlayerEloState

if TYPE_CHECKING:
    from collections.abc import Iterable

logger = get_logger(__name__)


class EloStateManager:
    """Manages in-memory Elo state for all players, with DB persistence.

    Typical usage:
        manager = EloStateManager()
        # Process matches chronologically, updating ratings
        for match in matches_iter:
            update_ratings(
                manager.get_state(match.winner_id),
                manager.get_state(match.loser_id),
                ...
            )
        # Persist final state to DB
        manager.save_to_db(algorithm_version="elo_v1_surface")
    """

    def __init__(self, config: EloConfig | None = None) -> None:
        self.config = config or EloConfig()
        self._states: dict[str, PlayerEloState] = {}

    def get_state(self, player_id: str) -> PlayerEloState:
        """Get or lazily create a PlayerEloState for a player."""
        if player_id not in self._states:
            self._states[player_id] = PlayerEloState(player_id=player_id)
        return self._states[player_id]

    def num_players(self) -> int:
        return len(self._states)

    def top_players(
        self,
        surface: str = "Overall",
        n: int = 10,
        min_matches: int = 20,
    ) -> list[tuple[str, float, int]]:
        """Get top N players by Elo on a given surface.

        Args:
            surface: 'Hard', 'Clay', 'Grass', 'Carpet', or 'Overall'
            n: Number of top players to return
            min_matches: Minimum matches on this surface to qualify

        Returns:
            List of (player_id, rating, matches_played) tuples, descending by rating
        """
        eligible = [
            (state.player_id, state.get_rating(surface, self.config), state.get_matches_played(surface))
            for state in self._states.values()
            if state.get_matches_played(surface) >= min_matches
        ]
        eligible.sort(key=lambda x: x[1], reverse=True)
        return eligible[:n]

    def save_to_db(
        self,
        rating_date: date,
        algorithm_version: str = "elo_v1_surface",
        min_matches_to_save: int = 1,
    ) -> int:
        """Persist current Elo state to elo_ratings table.

        Args:
            rating_date: Date to associate with these ratings (snapshot)
            algorithm_version: Identifier for this Elo variant
            min_matches_to_save: Skip players with fewer matches (reduce noise)

        Returns:
            Number of rating rows inserted
        """
        rows = []
        for state in self._states.values():
            for surface, rating in state.ratings.items():
                matches = state.matches_played.get(surface, 0)
                if matches < min_matches_to_save:
                    continue
                rows.append({
                    "player_id": state.player_id,
                    "rating_date": rating_date,
                    "surface": surface,
                    "elo_rating": round(rating, 2),
                    "matches_played": matches,
                    "algorithm_version": algorithm_version,
                })

        if not rows:
            logger.warning("no_ratings_to_save")
            return 0

        # Upsert using ON CONFLICT (so re-running doesn't fail)
        upsert_sql = text("""
            INSERT INTO elo_ratings (
                player_id, rating_date, surface, elo_rating,
                matches_played, algorithm_version
            ) VALUES (
                :player_id, :rating_date, :surface, :elo_rating,
                :matches_played, :algorithm_version
            )
            ON CONFLICT (player_id, rating_date, surface, algorithm_version)
            DO UPDATE SET
                elo_rating = EXCLUDED.elo_rating,
                matches_played = EXCLUDED.matches_played
        """)

        rows_inserted = 0
        batch_size = 500
        for i in range(0, len(rows), batch_size):
            batch = rows[i : i + batch_size]
            with get_session() as session:
                for row in batch:
                    session.execute(upsert_sql, row)
                    rows_inserted += 1

        logger.info(
            "elo_ratings_saved",
            count=rows_inserted,
            date=str(rating_date),
            version=algorithm_version,
        )
        return rows_inserted

    @classmethod
    def load_from_db(
        cls,
        rating_date: date,
        algorithm_version: str = "elo_v1_surface",
        config: EloConfig | None = None,
    ) -> EloStateManager:
        """Load an EloStateManager from a previously saved snapshot.

        Useful for resuming a backtest or generating fresh predictions.

        Args:
            rating_date: Date of the snapshot to load
            algorithm_version: Which Elo variant to load
            config: EloConfig to associate with the manager

        Returns:
            EloStateManager populated with the historical ratings
        """
        manager = cls(config=config)

        select_sql = text("""
            SELECT player_id, surface, elo_rating, matches_played
            FROM elo_ratings
            WHERE rating_date = :rating_date
              AND algorithm_version = :algorithm_version
        """)

        with get_session() as session:
            result = session.execute(
                select_sql,
                {"rating_date": rating_date, "algorithm_version": algorithm_version},
            )
            for row in result:
                state = manager.get_state(row.player_id)
                state.ratings[row.surface] = float(row.elo_rating)
                state.matches_played[row.surface] = row.matches_played

        logger.info(
            "elo_ratings_loaded",
            player_count=manager.num_players(),
            date=str(rating_date),
        )
        return manager
