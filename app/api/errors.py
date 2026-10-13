from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse

from app.errors import DatabaseUnavailableError, DuplicateEmailError, NotFoundError

Handler = Callable[[Request, Exception], Awaitable[JSONResponse]]


def _respond_with(status_code: int) -> Handler:
    async def respond(_: Request, exc: Exception) -> JSONResponse:
        return JSONResponse(status_code=status_code, content={"detail": str(exc)})

    return respond


def register_error_handlers(app: FastAPI) -> None:
    """Map domain errors raised by the services to HTTP responses."""
    app.add_exception_handler(NotFoundError, _respond_with(status.HTTP_404_NOT_FOUND))
    app.add_exception_handler(DuplicateEmailError, _respond_with(status.HTTP_409_CONFLICT))
    app.add_exception_handler(
        DatabaseUnavailableError, _respond_with(status.HTTP_503_SERVICE_UNAVAILABLE)
    )
