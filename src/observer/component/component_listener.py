from abc import ABC, abstractmethod
from typing import TypeVar, Generic

from src.observer.component.schema.schema import Schema

T = TypeVar('T', bound=Schema)


class ComponentListener(ABC, Generic[T]):
    """Interface for component listeners."""

    @abstractmethod
    def new_schema(self, component_id: str, schema: T) -> None:
        """
        Notifies about the new schema for a component.

        Args:
            component_id (str): the id of the component
            schema (T): the schema
        """
        raise NotImplementedError
