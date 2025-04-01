from dataclasses import dataclass
from typing import TypeVar, Generic

from src.schemas.schema import Schema

T = TypeVar("T", bound=Schema)

@dataclass(frozen=True)
class SignedSchema(Schema, Generic[T]):
    """
    A schema with the issuer ID attached.

    Attributes:
        issuer_id (str): The issuer ID
        schema (T): the schema
    """
    issuer_id: str
    schema: T

    @staticmethod
    def sign(issuer_id: str, schema: T) -> "SignedSchema[T]":
        return SignedSchema(issuer_id=issuer_id, schema=schema)