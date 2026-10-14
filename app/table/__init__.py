# Submodules are imported here so callers can write `from app import table` and then
# `table.documents.headlines(...)`: the module name says which table the SQL is about.
from app.table import clients, documents

__all__ = ["clients", "documents"]
