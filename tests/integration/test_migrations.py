from pathlib import Path

import psycopg
import pytest

from app.table.migrate import MIGRATIONS_DIR, migrate


def test_migrations_are_recorded_and_not_reapplied(database_url: str) -> None:
    # The session fixture already migrated the database: a second run is a no-op.
    assert migrate(database_url) == []
    with psycopg.connect(database_url) as conn:
        versions = [row[0] for row in conn.execute("SELECT version FROM schema_migrations")]
    assert versions == sorted(p.stem for p in MIGRATIONS_DIR.glob("*.sql"))
    assert "0001_initial" in versions


def test_new_migration_file_is_applied_once(database_url: str, tmp_path: Path) -> None:
    (tmp_path / "0001_first.sql").write_text(
        "CREATE TABLE migration_probe (id int); INSERT INTO migration_probe VALUES (1);"
    )
    (tmp_path / "0002_second.sql").write_text("INSERT INTO migration_probe VALUES (2);")
    try:
        assert migrate(database_url, tmp_path) == ["0001_first", "0002_second"]
        assert migrate(database_url, tmp_path) == []
        with psycopg.connect(database_url) as conn:
            rows = conn.execute("SELECT id FROM migration_probe ORDER BY id").fetchall()
        assert rows == [(1,), (2,)]
    finally:
        with psycopg.connect(database_url) as conn:
            conn.execute("DROP TABLE IF EXISTS migration_probe")
            conn.execute(
                "DELETE FROM schema_migrations WHERE version IN ('0001_first', '0002_second')"
            )


def test_failing_migration_is_rolled_back_and_not_recorded(
    database_url: str, tmp_path: Path
) -> None:
    (tmp_path / "0001_broken.sql").write_text(
        "CREATE TABLE half_done (id int); SELECT no_such_function();"
    )
    with pytest.raises(psycopg.errors.UndefinedFunction):
        migrate(database_url, tmp_path)
    with psycopg.connect(database_url) as conn:
        assert conn.execute("SELECT to_regclass('half_done')").fetchone() == (None,)
        recorded = conn.execute(
            "SELECT count(*) FROM schema_migrations WHERE version = '0001_broken'"
        ).fetchone()
    assert recorded == (0,)


def test_tsquery_or_agg_combines_phrases(database_url: str) -> None:
    with psycopg.connect(database_url) as conn:
        row = conn.execute(
            """
            SELECT to_tsvector('english', 'Electricity bill for March') @@ tq,
                   to_tsvector('english', 'Passport copy') @@ tq
            FROM (SELECT tsquery_or_agg(phraseto_tsquery('english', p)) AS tq
                  FROM unnest(ARRAY['utility bill', 'electricity bill', 'lease']) AS p) x
            """
        ).fetchone()
    assert row == (True, False)
