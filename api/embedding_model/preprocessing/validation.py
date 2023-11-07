from typing import List, Optional, Tuple

from pydantic import BaseModel, ValidationError


def validate_input(input_data: str) -> Tuple[str, Optional[str]]:
    """Check model inputs for unprocessable values."""

    validated_data = input_data
    errors = None

    try:
        DataInput(
            input=validated_data
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class DataInput(BaseModel):
    input: str
