from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.schemas.schemas.schema import Schema
from src.schemas.schemas.signed_schema import SignedSchema

T = TypeVar("T", bound=Schema)

class SchemaSigner(Generic[T], ABC):
    """Interface for schema signing."""

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