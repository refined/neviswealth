import uuid

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from psycopg import Connection
from psycopg.errors import UniqueViolation

from app import schemas
from app.api.deps import Container, get_conn, get_container
from app.indexing.text import build_client_search_text
from app.table import clients, documents
from app.table.records import ClientRecord, DocumentRecord

router = APIRouter(tags=["clients"])

NOT_FOUND = {404: {"description": "Client not found"}}


def get_client_or_404(conn: Connection, client_id: uuid.UUID) -> ClientRecord:
    client = clients.get_client(conn, client_id)
    if client is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, detail="Client not found")
    return client


@router.post(
    "/clients",
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.Client,
    responses={409: {"description": "A client with this email already exists"}},
)
def create_client(
    body: schemas.ClientCreate, response: Response, conn: Connection = Depends(get_conn)
) -> ClientRecord:
    email = str(body.email)
    try:
        client = clients.insert_client(
            conn,
            first_name=body.first_name,
            last_name=body.last_name,
            email=email,
            description=body.description or None,
            social_links=body.social_links,
            # Pre-processing at write time: identity terms for substring / fuzzy search.
            search_text=build_client_search_text(
                body.first_name, body.last_name, email, body.social_links
            ),
        )
    except UniqueViolation as exc:
        raise HTTPException(
            status.HTTP_409_CONFLICT, detail="A client with this email already exists"
        ) from exc
    conn.commit()
    response.headers["Location"] = f"/clients/{client.id}"
    return client


@router.get("/clients", response_model=list[schemas.Client])
def list_clients(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    conn: Connection = Depends(get_conn),
) -> list[ClientRecord]:
    return clients.list_clients(conn, limit=limit, offset=offset)


@router.get("/clients/{client_id}", response_model=schemas.Client, responses=NOT_FOUND)
def get_client(client_id: uuid.UUID, conn: Connection = Depends(get_conn)) -> ClientRecord:
    return get_client_or_404(conn, client_id)


@router.post(
    "/clients/{client_id}/documents",
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.Document,
    responses=NOT_FOUND,
    tags=["documents"],
)
def create_document(
    client_id: uuid.UUID,
    body: schemas.DocumentCreate,
    response: Response,
    conn: Connection = Depends(get_conn),
    container: Container = Depends(get_container),
) -> DocumentRecord:
    get_client_or_404(conn, client_id)
    # The full-text index is maintained by Postgres synchronously; chunking, embeddings and the
    # summary are produced asynchronously by the indexing worker.
    document = documents.insert_document(
        conn, client_id=client_id, title=body.title, content=body.content
    )
    conn.commit()
    container.worker.notify()
    response.headers["Location"] = f"/documents/{document.id}"
    return document


@router.get(
    "/clients/{client_id}/documents",
    response_model=list[schemas.Document],
    responses=NOT_FOUND,
    tags=["documents"],
)
def list_client_documents(
    client_id: uuid.UUID, conn: Connection = Depends(get_conn)
) -> list[DocumentRecord]:
    get_client_or_404(conn, client_id)
    return documents.list_client_documents(conn, client_id)
