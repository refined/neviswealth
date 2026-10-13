from typing import Annotated

from fastapi import APIRouter, Query
from pydantic import BaseModel, Field, StringConstraints

from app import schemas
from app.api.deps import SearchServiceDep
from app.services.search import ResultType, SearchRequest

router = APIRouter(tags=["search"])


class SearchParams(BaseModel):
    q: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)] = (
        Field(
            description="Free text: a name, an email or part of it, or what a document is about.",
            examples=["NevisWealth", "address proof"],
        )
    )
    type: ResultType | None = Field(None, description="Restrict results to one kind of entity.")
    limit: int = Field(20, ge=1, le=100)

    def to_request(self) -> SearchRequest:
        types = frozenset({self.type}) if self.type else frozenset(ResultType)
        return SearchRequest(query=self.q, types=types, limit=self.limit)


@router.get("/search", response_model=list[schemas.SearchHit])
def search(
    params: Annotated[SearchParams, Query()], service: SearchServiceDep
) -> list[schemas.ClientHit | schemas.DocumentHit]:
    """Search clients and documents; one list sorted by relevance score.

    - **Clients** match on name, email (including its parts and domain), social links
      (substring and typo-tolerant) and description (full-text).
    - **Documents** match on title/content (full-text), on meaning (embedding similarity of
      chunks) and on related terms of the document type the query refers to.
    """
    return service.search(params.to_request())
