"""Write the OpenAPI spec to docs/openapi.json (no database or model needed).

Run from the repository root: `uv run python -m scripts.export_openapi`
"""

import json
from pathlib import Path

from app.config import Settings
from app.main import create_app
from tests.fakes import HashingEmbedder

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"

if __name__ == "__main__":
    # The pool opens only in the app lifespan, so no database is needed; the fake embedder
    # avoids loading the model.
    app = create_app(Settings(run_worker=False), HashingEmbedder())
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
