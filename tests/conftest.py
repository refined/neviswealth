from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PG_IMAGE = "pgvector/pgvector:pg17"


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """A throwaway Postgres + pgvector container with all migrations applied."""
    from testcontainers.community.postgres import PostgresContainer

    from app.table.migrate import migrate

    with PostgresContainer(PG_IMAGE, driver=None) as pg:
        url = pg.get_connection_url()
        migrate(url)
        yield url
