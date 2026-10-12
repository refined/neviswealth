# syntax=docker/dockerfile:1
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    MODEL_CACHE_DIR=/models \
    PATH="/app/.venv/bin:$PATH"

COPY --from=ghcr.io/astral-sh/uv:0.10.10 /uv /usr/local/bin/uv

WORKDIR /app
RUN useradd --create-home --uid 1000 app && mkdir /models && chown app /models

# Dependencies first, for layer caching.
COPY pyproject.toml uv.lock .python-version ./
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev --no-install-project

# Bake the embedding model into the image: the container then runs fully offline.
# Downloaded as the runtime user, so no `chown` layer duplicates the model files.
USER app
ARG EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENV EMBEDDING_MODEL=${EMBEDDING_MODEL}
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('${EMBEDDING_MODEL}', cache_dir='/models')"
ENV HF_HUB_OFFLINE=1

COPY db/ db/
COPY scripts/seed.py scripts/seed.py
COPY app/ app/

EXPOSE 8000
CMD ["sh", "-c", "python -m app.table.migrate && exec uvicorn app.main:create_app --factory --host 0.0.0.0 --port 8000"]
