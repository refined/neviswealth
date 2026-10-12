"""Load the demo corpus (db/seed/seed_corpus.json) through the public API.

Uses only the standard library, so it runs on the host or inside the app container:

    uv run python scripts/seed.py
    docker compose exec app python scripts/seed.py

Clients that already exist (HTTP 409) are skipped together with their documents, so re-running
the script does not create duplicates.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

CORPUS = Path(__file__).resolve().parents[1] / "db" / "seed" / "seed_corpus.json"


class Api:
    def __init__(self, base_url: str, api_key: str | None) -> None:
        self.base_url = base_url.rstrip("/")
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["X-API-Key"] = api_key

    def request(self, method: str, path: str, body: Any = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base_url + path, data=data, headers=self.headers, method=method
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return response.status, json.loads(response.read() or b"null")
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read() or b"null")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--timeout", type=float, default=120, help="seconds to wait for indexing")
    args = parser.parse_args()

    api = Api(args.base_url, args.api_key)
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))

    client_ids: dict[str, str] = {}
    for client in corpus["clients"]:
        body = {k: v for k, v in client.items() if k != "key"}
        status, payload = api.request("POST", "/clients", body)
        if status == 201:
            client_ids[client["key"]] = payload["id"]
            print(f"client   {client['email']:<32} created")
        elif status == 409:
            print(f"client   {client['email']:<32} exists, skipped with its documents")
        else:
            print(f"client   {client['email']}: HTTP {status} {payload}", file=sys.stderr)
            return 1

    document_ids = []
    for doc in corpus["documents"]:
        if doc["client"] not in client_ids:
            continue
        status, payload = api.request(
            "POST",
            f"/clients/{client_ids[doc['client']]}/documents",
            {"title": doc["title"], "content": doc["content"]},
        )
        if status != 201:
            print(f"document {doc['title']}: HTTP {status} {payload}", file=sys.stderr)
            return 1
        document_ids.append(payload["id"])
        print(f"document {doc['title']:<45} queued")

    deadline = time.monotonic() + args.timeout
    pending = set(document_ids)
    while pending and time.monotonic() < deadline:
        for doc_id in list(pending):
            _, doc = api.request("GET", f"/documents/{doc_id}")
            if doc["status"] in ("ready", "failed"):
                pending.discard(doc_id)
                if doc["status"] == "failed":
                    print(f"document {doc['title']} failed: {doc['error']}", file=sys.stderr)
        if pending:
            time.sleep(0.5)

    indexed = len(document_ids) - len(pending)
    print(f"Done: {len(client_ids)} clients, {indexed}/{len(document_ids)} documents indexed.")
    return 0 if not pending else 1


if __name__ == "__main__":
    sys.exit(main())
