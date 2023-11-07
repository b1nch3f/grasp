from typing import Any, List, Optional

from pydantic import BaseModel


class DataInput(BaseModel):
    input: str


class Vector(BaseModel):
    errors: Optional[Any]
    version: str
    embeddings: Optional[List[float]]
