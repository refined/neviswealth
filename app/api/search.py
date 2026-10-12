from typing import Literal

from fastapi import APIRouter, Depends, Query
from fastapi.exceptions import RequestValidationError
from psycopg import Connection

from app import schemas
from app.api.deps import Container, get_conn, get_container

router = APIRouter(tags=["search"])

MAX_QUERY_LENGTH = 256


@router.get("/search", response_model=list[schemas.SearchHit])
def search(
    q: str = Query(
        ...,
        max_length=MAX_QUERY_LENGTH,
        description="Free text: a name, an email or part of it, or what a document is about.",
        examples=["NevisWealth", "address proof"],
    ),
    type: Literal["client", "document"] | None = Query(
        None, description="Restrict results to one kind of entity."
    ),
    limit: int = Query(20, ge=1, le=100),
    conn: Connection = Depends(get_conn),
    container: Container = Depends(get_container),
) -> list[schemas.ClientHit | schemas.DocumentHit]:
    """Search clients and documents; one list sorted by relevance score.

    - **Clients** match on name, email (including its parts and domain), social links
      (substring and typo-tolerant) and description (full-text).
    - **Documents** match on title/content (full-text), on meaning (embedding similarity of
      chunks) and on related terms of the document type the query refers to.
    """
    query = q.strip()
    if not query:
        raise RequestValidationError(
            [
                {
                    "type": "string_too_short",
                    "loc": ("query", "q"),
                    "msg": "Query must not be blank",
                    "input": q,
                }
            ]
        )
    kinds = {type} if type else {"client", "document"}
    return container.search.search(conn, query, kinds, limit)
