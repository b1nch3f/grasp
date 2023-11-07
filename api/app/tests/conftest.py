from typing import Generator

import pytest
from fastapi.testclient import TestClient
from embedding_model.config.core import config

from app.main import app


@pytest.fixture(scope="module")
def test_data() -> str:
    return 'a good day to die hard'


@pytest.fixture()
def client() -> Generator:
    with TestClient(app) as _client:
        yield _client
        app.dependency_overrides = {}
