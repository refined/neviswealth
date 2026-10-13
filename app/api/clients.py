import uuid

from fastapi import APIRouter, Query, Response, status

from app import schemas
from app.api.deps import ClientServiceDep, DocumentServiceDep
from app.table.records import ClientRecord, DocumentRecord

router = APIRouter(tags=["clients"])

NOT_FOUND = {404: {"description": "Client not found"}}


@router.post(
    "/clients",
    status_code=status.HTTP_201_CREATED,
    response_model=schemas.Client,
    responses={409: {"description": "A client with this email already exists"}},
)
def create_client(
    body: schemas.ClientCreate, response: Response, clients: ClientServiceDep
) -> ClientRecord:
    client = clients.create(
        first_name=body.first_name,
        last_name=body.last_name,
        email=str(body.email),
        description=body.description or None,
        social_links=body.social_links,
    )
    response.headers["Location"] = f"/clients/{client.id}"
    return client


@router.get("/clients", response_model=list[schemas.Client])
def list_clients(
    clients: ClientServiceDep,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> list[ClientRecord]:
    return clients.list(limit=limit, offset=offset)


@router.get("/clients/{client_id}", response_model=schemas.Client, responses=NOT_FOUND)
def get_client(client_id: uuid.UUID, clients: ClientServiceDep) -> ClientRecord:
    return clients.get(client_id)


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
    documents: DocumentServiceDep,
) -> DocumentRecord:
    document = documents.create(client_id, title=body.title, content=body.content)
    response.headers["Location"] = f"/documents/{document.id}"
    return document


@router.get(
    "/clients/{client_id}/documents",
    response_model=list[schemas.Document],
    responses=NOT_FOUND,
    tags=["documents"],
)
def list_client_documents(
    client_id: uuid.UUID, documents: DocumentServiceDep
) -> list[DocumentRecord]:
    return documents.list_for_client(client_id)
