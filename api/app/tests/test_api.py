import math

import numpy as np
from fastapi.testclient import TestClient

from embedding_model.config.core import config


def test_make_embeddings(client: TestClient, test_data: str) -> None:
    # Given
    payload = {
        # ensure pydantic plays well with np.nan
        "input": test_data
    }

    # When
    response = client.post(
        "http://localhost:8001/api/v1/vectorize",
        json=payload,
    )

    # Then
    assert response.status_code == 200
    prediction_data = response.json()
    assert prediction_data["embeddings"]
    assert len(prediction_data["embeddings"]) == config.embedding_model_config.embedding_size
    assert prediction_data["errors"] is None
