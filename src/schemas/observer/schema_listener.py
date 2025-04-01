from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.schemas.schema import Schema

T = TypeVar('T', bound=Schema)


class SchemaListener(ABC, Generic[T]):
    """Interface for schema listeners."""

    @abstractmethod
    def new_schema(self, schema: T) -> None:
        """
        Notifies about the new schema.

        Args:
            schema (T): the schema
        """
        raise NotImplementedError
