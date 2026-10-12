from pathlib import Path
from typing import Any

from fastapi import APIRouter, FastAPI, Response
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles

STATIC_DIR = Path(__file__).resolve().parents[1] / "static"

router = APIRouter(include_in_schema=False)


class NoCacheStaticFiles(StaticFiles):
    # Browsers revalidate (ETag) on every load, so a redeploy shows up without a hard refresh.
    def file_response(self, *args: Any, **kwargs: Any) -> Response:
        response = super().file_response(*args, **kwargs)
        response.headers["Cache-Control"] = "no-cache"
        return response


@router.get("/")
def root() -> RedirectResponse:
    return RedirectResponse("/ui/")


def mount_ui(app: FastAPI) -> None:
    # Static files are a mount, not a route, so they cannot live on the router.
    app.mount("/ui", NoCacheStaticFiles(directory=STATIC_DIR, html=True), name="ui")
