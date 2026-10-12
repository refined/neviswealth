"""Minimal migration runner for plain SQL files in db/migrations/.

Each `NNNN_name.sql` file is applied once, in filename order, inside its own transaction, and
recorded in `schema_migrations`. Migrations are forward-only: to change the schema, add a new file.

    python -m app.table.migrate        # uses DATABASE_URL
"""

import logging
from pathlib import Path

import psycopg

from app.config import get_settings

log = logging.getLogger(__name__)

MIGRATIONS_DIR = Path(__file__).resolve().parents[2] / "db" / "migrations"

# Arbitrary constant: serializes concurrent runners (e.g. two app containers starting at once).
_ADVISORY_LOCK_KEY = 4_270_001


def migrate(database_url: str, directory: Path = MIGRATIONS_DIR) -> list[str]:
    applied: list[str] = []
    with psycopg.connect(database_url, autocommit=True) as conn:
        conn.execute("SELECT pg_advisory_lock(%s)", (_ADVISORY_LOCK_KEY,))
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    version    text PRIMARY KEY,
                    applied_at timestamptz NOT NULL DEFAULT now()
                )
                """
            )
            done = {row[0] for row in conn.execute("SELECT version FROM schema_migrations")}
            for path in sorted(directory.glob("*.sql")):
                if path.stem in done:
                    continue
                with conn.transaction():
                    # Without parameters psycopg sends the file as-is: many statements are fine.
                    conn.execute(path.read_text(encoding="utf-8"))  # type: ignore[arg-type]
                    conn.execute(
                        "INSERT INTO schema_migrations (version) VALUES (%s)", (path.stem,)
                    )
                log.info("Applied migration %s", path.stem)
                applied.append(path.stem)
        finally:
            conn.execute("SELECT pg_advisory_unlock(%s)", (_ADVISORY_LOCK_KEY,))
    return applied


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    versions = migrate(get_settings().database_url)
    log.info("Database is up to date (%d migration(s) applied now)", len(versions))
