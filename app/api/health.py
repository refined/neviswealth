from fastapi import APIRouter

from app import schemas
from app.api.deps import HealthServiceDep

router = APIRouter(tags=["health"])


@router.get("/health", response_model=schemas.Health)
def health(service: HealthServiceDep) -> schemas.Health:
    info = service.check()
    return schemas.Health(
        status="ok",
        database="ok",
        embedding_model=info.embedding_model,
        worker_running=info.worker_running,
    )
