from fastapi import APIRouter, Depends, HTTPException, status

from app import schemas
from app.api.deps import Container, get_container

router = APIRouter(tags=["health"])


@router.get("/health", response_model=schemas.Health)
def health(container: Container = Depends(get_container)) -> schemas.Health:
    try:
        with container.pool.connection(timeout=5) as conn:
            conn.execute("SELECT 1")
    except Exception as exc:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, detail="Database unavailable"
        ) from exc
    return schemas.Health(
        status="ok",
        database="ok",
        embedding_model=container.embedder.model_name,
        worker_running=container.worker.is_running,
    )
