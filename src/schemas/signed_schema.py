import time
from dataclasses import dataclass
from typing import TypeVar, Generic

from src.schemas.schema import Schema

T = TypeVar("T", bound=Schema)

@dataclass(frozen=True)
class SignedSchema(Schema, Generic[T]):
    """
    A schema with the issuer ID attached.

    Attributes:
        signature (str): the signature
        timestamp (float): a timestamp for when the schema was created
        schema (T): the schema
    """
    signature: str
    timestamp: float
    schema: T