from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.schemas.schema import Schema

T = TypeVar("T", bound=Schema)
C = TypeVar("C", bound=Schema)

class SchemaConverter(Generic[T, C], ABC):
    """Interface for schema converters."""

    @abstractmethod
    def convert(self, schema: T) -> C:
        """
        Converts a schema.

        Args:
            schema: the schema to convert

        Returns:
            C: the converted schema
        """
        raise NotImplementedError