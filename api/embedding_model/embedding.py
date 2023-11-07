import typing as t

from sentence_transformers import SentenceTransformer

from embedding_model import __version__ as _version
from embedding_model.config.core import config
from embedding_model.preprocessing.validation import validate_input

model = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-tas-b')


async def make_embeddings(input_data: str) -> list:
    """
    Generate embeddings for input string
    """
    validated_data, errors = validate_input(input_data=input_data)
    results = {"embeddings": None, "errors": errors}

    if not errors:
        results = {
            "embeddings": model.encode(validated_data)[:config.embedding_model_config.embedding_size].tolist(),
            "version": _version,
            "errors": errors,
        }

    
    
    return results