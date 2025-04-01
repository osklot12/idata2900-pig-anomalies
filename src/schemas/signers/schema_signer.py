from abc import ABC, abstractmethod
from typing import TypeVar

from src.schemas.schema import Schema
from src.schemas.signed_schema import SignedSchema

T = TypeVar("T", bound=Schema)

class SchemaSigner(ABC):
    """Interface for schema signers."""

    @abstractmethod
    def sign(self, schema: T) -> SignedSchema[T]:
        """
        Signs the given schema.

        Args:
            schema (T): the schema to sign

        Returns:
            SignedSchema[T]: the signed schema
        """
        raise NotImplementedError