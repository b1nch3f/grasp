import json
from typing import Any

from fastapi import APIRouter, HTTPException
from loguru import logger

from embedding_model.embedding import make_embeddings

from app import __version__, schemas
from app.config import settings

api_router = APIRouter()


@api_router.post("/vectorize", response_model=schemas.Vector, status_code=200)
async def vectorize(input_data: schemas.DataInput) -> Any:
    """
    Generate embeddings for input string
    """

    results = await make_embeddings(input_data=input_data.input)

    if results["errors"] is not None:
        logger.warning(f"Embedding validation error: {results.get('errors')}")
        raise HTTPException(status_code=400, detail=json.loads(results["errors"]))
    
    return results
