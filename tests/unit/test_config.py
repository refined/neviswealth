import numpy as np
import pytest

from app.config import Settings
from app.table.connection import vector_literal


def test_defaults_without_environment() -> None:
    settings = Settings.from_env({})
    assert settings == Settings()
    assert settings.run_worker is True
    assert settings.api_key is None


def test_values_are_parsed_by_field_type() -> None:
    settings = Settings.from_env(
        {
            "DATABASE_URL": "postgresql://u:p@db:5432/x",
            "RUN_WORKER": "false",
            "WORKER_MAX_ATTEMPTS": "5",
            "SEMANTIC_MIN_SIMILARITY": "0.3",
            "API_KEY": "s3cret",
            "UNRELATED_VARIABLE": "ignored",
        }
    )
    assert settings.database_url == "postgresql://u:p@db:5432/x"
    assert settings.run_worker is False
    assert settings.worker_max_attempts == 5
    assert settings.semantic_min_similarity == 0.3
    assert settings.api_key == "s3cret"


@pytest.mark.parametrize(("raw", "expected"), [("1", True), ("Yes", True), ("off", False)])
def test_boolean_spellings(raw: str, expected: bool) -> None:
    assert Settings.from_env({"RUN_WORKER": raw}).run_worker is expected


def test_empty_optional_value_means_unset() -> None:
    # docker-compose passes `API_KEY: ${API_KEY:-}`, i.e. an empty string when not configured.
    assert Settings.from_env({"API_KEY": ""}).api_key is None


def test_invalid_values_fail_loudly() -> None:
    with pytest.raises(ValueError, match="RUN_WORKER"):
        Settings.from_env({"RUN_WORKER": "maybe"})
    with pytest.raises(ValueError):
        Settings.from_env({"WORKER_MAX_ATTEMPTS": "three"})


def test_vector_literal_uses_pgvector_text_format() -> None:
    assert vector_literal(np.array([0.5, -1.0, 0.123456789], dtype=np.float32)) == (
        "[0.5,-1,0.1234568]"
    )
