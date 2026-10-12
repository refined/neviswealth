import numpy as np
from psycopg_pool import ConnectionPool


def create_pool(database_url: str, *, min_size: int = 1, max_size: int = 5) -> ConnectionPool:
    """A small connection pool (the target box has one vCPU). Opened by the app lifespan."""
    return ConnectionPool(database_url, min_size=min_size, max_size=max_size, open=False)


def vector_literal(vector: np.ndarray) -> str:
    """pgvector's text format, `[0.1,0.2,...]`; cast it in SQL with `%s::vector`."""
    return "[" + ",".join(f"{x:.7g}" for x in vector.tolist()) + "]"
