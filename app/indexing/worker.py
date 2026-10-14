"""Background indexing worker backed by a queue that lives in the `documents` table itself.

Documents are claimed with `FOR UPDATE SKIP LOCKED`, so several workers (threads or processes) can
run safely. A claim stamps `locked_at`; the result is only written if that stamp is unchanged,
which makes a concurrent "reprocess" request win over a stale in-flight run.
"""

import logging
import threading
import time

from psycopg_pool import ConnectionPool

from app import table
from app.indexing.embedder import Embedder
from app.indexing.summarizer import ExtractiveSummarizer
from app.indexing.text import chunk_sentences, split_sentences
from app.table.records import DocumentStatus, Job, NewChunk

log = logging.getLogger(__name__)


class IndexingWorker:
    def __init__(
        self,
        pool: ConnectionPool,
        embedder: Embedder,
        summarizer: ExtractiveSummarizer,
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

    def process_pending(self, stop_event: threading.Event | None = None) -> int:
        processed = 0
        while stop_event is None or not stop_event.is_set():
            job = self._claim()
            if job is None:
                break
            self._process(job)
            processed += 1
        return processed

    def _claim(self) -> Job | None:
        with self._pool.connection() as conn:
            return table.documents.claim_next_document(
                conn, stale_after_s=self._stale_after_s, max_attempts=self._max_attempts
            )

    def _process(self, job: Job) -> None:
        started = time.perf_counter()
        try:
            chunks, summary = self._index(job)
            elapsed_ms = int((time.perf_counter() - started) * 1000)
            self._save(job, chunks, summary, elapsed_ms)
        except Exception as exc:
            log.exception("Indexing failed for document %s (attempt %d)", job.id, job.attempts)
            self._fail(job, exc)

    def _index(self, job: Job) -> tuple[list[NewChunk], str]:
        spans = chunk_sentences(job.content, split_sentences(job.content), self._chunk_max_words)
        # Each chunk carries the title so a chunk like "Amount due: $143" still knows it is a bill.
        vectors = self._embedder.embed_documents([f"{job.title}. {s.text}" for s in spans])
        chunks = [
            NewChunk(i, span.text, span.start, span.end, vector)
            for i, (span, vector) in enumerate(zip(spans, vectors, strict=True))
        ]
        return chunks, self._summarizer.summarize(job.content)

    def _save(self, job: Job, chunks: list[NewChunk], summary: str, elapsed_ms: int) -> None:
        with self._pool.connection() as conn:
            saved = table.documents.save_index_result(
                conn, job, summary=summary, chunks=chunks, processing_ms=elapsed_ms
            )
        if saved:
            log.info("Indexed document %s: %d chunks in %d ms", job.id, len(chunks), elapsed_ms)
        else:
            log.info("Document %s changed while indexing; result discarded", job.id)

    def _fail(self, job: Job, exc: Exception) -> None:
        final = job.attempts >= self._max_attempts
        status = DocumentStatus.FAILED if final else DocumentStatus.PENDING
        try:
            with self._pool.connection() as conn:
                table.documents.record_failure(
                    conn, job, status=status, error=f"{type(exc).__name__}: {exc}"[:2000]
                )
        except Exception:
            log.exception("Could not record indexing failure for document %s", job.id)
