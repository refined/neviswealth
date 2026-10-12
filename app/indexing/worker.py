"""Background indexing worker backed by a queue that lives in the `documents` table itself.

Documents are claimed with `FOR UPDATE SKIP LOCKED`, so several workers (threads or processes) can
run safely. A claim stamps `locked_at`; the result is only written if that stamp is unchanged,
which makes a concurrent "reprocess" request win over a stale in-flight run.
"""

import logging
import threading
import time

from psycopg_pool import ConnectionPool

from app.indexing.embedder import Embedder
from app.indexing.pipeline import IndexedDocument, index_document
from app.indexing.summarizer import Summarizer
from app.table import documents
from app.table.records import DocumentStatus, Job

log = logging.getLogger(__name__)


class IndexingWorker:
    def __init__(
        self,
        pool: ConnectionPool,
        embedder: Embedder,
        summarizer: Summarizer,
        *,
        chunk_max_words: int,
        max_attempts: int = 3,
        poll_interval_s: float = 2.0,
        stale_after_s: int = 300,
    ) -> None:
        self._pool = pool
        self._embedder = embedder
        self._summarizer = summarizer
        self._chunk_max_words = chunk_max_words
        self._max_attempts = max_attempts
        self._poll_interval_s = poll_interval_s
        self._stale_after_s = stale_after_s
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    # -- lifecycle -------------------------------------------------------------------------

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="indexing-worker", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread:
            self._thread.join(timeout)

    def notify(self) -> None:
        """Wake the worker immediately (called after a document is created or re-queued)."""
        self._wake.set()

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _run(self) -> None:
        log.info("Indexing worker started")
        while not self._stop.is_set():
            try:
                processed = self.process_pending(stop_event=self._stop)
            except Exception:
                log.exception("Indexing worker loop failed")
                processed = 0
            if not processed:
                self._wake.wait(self._poll_interval_s)
                self._wake.clear()
        log.info("Indexing worker stopped")

    # -- processing ------------------------------------------------------------------------

    def process_pending(
        self, limit: int | None = None, stop_event: threading.Event | None = None
    ) -> int:
        """Process queued documents until the queue is empty (or `limit` is reached)."""
        processed = 0
        while limit is None or processed < limit:
            if stop_event is not None and stop_event.is_set():
                break
            job = self._claim()
            if job is None:
                break
            self._process(job)
            processed += 1
        return processed

    def _claim(self) -> Job | None:
        with self._pool.connection() as conn:
            return documents.claim_next_document(
                conn, stale_after_s=self._stale_after_s, max_attempts=self._max_attempts
            )

    def _process(self, job: Job) -> None:
        started = time.perf_counter()
        try:
            result = index_document(
                job.title, job.content, self._embedder, self._summarizer, self._chunk_max_words
            )
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self._save(job, result, elapsed_ms)
        except Exception as exc:
            log.exception("Indexing failed for document %s (attempt %d)", job.id, job.attempts)
            self._fail(job, exc)

    def _save(self, job: Job, result: IndexedDocument, elapsed_ms: int) -> None:
        with self._pool.connection() as conn:
            saved = documents.save_index_result(
                conn, job, summary=result.summary, chunks=result.chunks, processing_ms=elapsed_ms
            )
        if saved:
            log.info(
                "Indexed document %s: %d chunks in %d ms", job.id, len(result.chunks), elapsed_ms
            )
        else:
            log.info("Document %s changed while indexing; result discarded", job.id)

    def _fail(self, job: Job, exc: Exception) -> None:
        final = job.attempts >= self._max_attempts
        status = DocumentStatus.FAILED if final else DocumentStatus.PENDING
        try:
            with self._pool.connection() as conn:
                documents.record_failure(
                    conn, job, status=status, error=f"{type(exc).__name__}: {exc}"[:2000]
                )
        except Exception:
            log.exception("Could not record indexing failure for document %s", job.id)
