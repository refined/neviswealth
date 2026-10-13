from dataclasses import dataclass

from psycopg_pool import ConnectionPool

from app.errors import DatabaseUnavailableError
from app.indexing.embedder import Embedder
from app.indexing.worker import IndexingWorker


@dataclass(frozen=True)
class HealthInfo:
    embedding_model: str
    worker_running: bool


class HealthService:
    def __init__(self, pool: ConnectionPool, embedder: Embedder, worker: IndexingWorker) -> None:
        self._pool = pool
        self._embedder = embedder
        self._worker = worker

    def check(self) -> HealthInfo:
        try:
            with self._pool.connection(timeout=5) as conn:
                conn.execute("SELECT 1")
        except Exception as exc:
            raise DatabaseUnavailableError from exc
        return HealthInfo(
            embedding_model=self._embedder.model_name,
            worker_running=self._worker.is_running,
        )
