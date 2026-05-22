"""Elo v2 state manager."""

from __future__ import annotations

from datetime import date

from tennis_predictor.models.elo_v2 import EloConfigV2, PlayerEloStateV2


class EloStateManagerV2:
    """Manages PlayerEloStateV2 instances with DB persistence."""

    def __init__(self, config: EloConfigV2 | None = None) -> None:
        self.config = config or EloConfigV2()
        self._states: dict[str, PlayerEloStateV2] = {}

    def get_state(self, player_id: str) -> PlayerEloStateV2:
        if player_id not in self._states:
            self._states[player_id] = PlayerEloStateV2(player_id=player_id)
        return self._states[player_id]

    def num_players(self) -> int:
        return len(self._states)

    def top_players(
        self,
        surface: str = "Overall",
        n: int = 10,
        min_matches: int = 20,
    ) -> list[tuple[str, float, int]]:
        eligible = [
            (s.player_id, s.get_rating(surface, self.config), s.get_matches_played(surface))
            for s in self._states.values()
            if s.get_matches_played(surface) >= min_matches
        ]
        eligible.sort(key=lambda x: x[1], reverse=True)
        return eligible[:n]

    def save_to_db(
        self,
        rating_date: date,
        algorithm_version: str = "elo_v2_surface",
        min_matches_to_save: int = 1,
    ) -> int:
        from sqlalchemy import text
        from tennis_predictor.data.storage import get_session
        from tennis_predictor.logging_config import get_logger
        logger = get_logger(__name__)

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
            return 0

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

        logger.info("elo_v2_saved", count=rows_inserted, version=algorithm_version)
        return rows_inserted

    @classmethod
    def load_from_db(
        cls,
        rating_date: date,
        algorithm_version: str = "elo_v2_surface",
        config: EloConfigV2 | None = None,
    ) -> EloStateManagerV2:
        from sqlalchemy import text
        from tennis_predictor.data.storage import get_session

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

        return manager
