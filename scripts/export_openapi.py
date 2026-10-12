"""Write the OpenAPI spec to docs/openapi.json (no database or model needed).

Run from the repository root: `uv run python -m scripts.export_openapi`
"""

import json
from pathlib import Path

from app.config import Settings
from app.indexing.embedder import HashingEmbedder
from app.main import create_app

OUTPUT = Path(__file__).resolve().parents[1] / "docs" / "openapi.json"

if __name__ == "__main__":
    # The engine connects lazily, so a placeholder URL is fine for building the schema.
    app = create_app(Settings(embedding_model="hashing", run_worker=False), HashingEmbedder())
    OUTPUT.parent.mkdir(exist_ok=True)
    OUTPUT.write_text(json.dumps(app.openapi(), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT}")
