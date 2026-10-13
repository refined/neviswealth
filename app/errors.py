class NotFoundError(Exception):
    def __init__(self, entity: str) -> None:
        super().__init__(f"{entity} not found")
        self.entity = entity


class DuplicateEmailError(Exception):
    def __init__(self) -> None:
        super().__init__("A client with this email already exists")


class DatabaseUnavailableError(Exception):
    def __init__(self) -> None:
        super().__init__("Database unavailable")
