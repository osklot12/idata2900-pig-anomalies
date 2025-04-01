from dataclasses import dataclass
from typing import TypeVar, Generic

from src.schemas.schemas.schema import Schema

T = TypeVar("T", bound=Schema)

@dataclass(frozen=True)
class SignedSchema(Schema, Generic[T]):
    """
    A schema with the issuer ID attached.

    Attributes:
        signature (str): the signature
        schema (T): the schema
    """
    signature: str
    schema: T