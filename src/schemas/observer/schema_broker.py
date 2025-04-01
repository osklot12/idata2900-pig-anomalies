from typing import TypeVar, Generic, List

from src.schemas.observer.broker import Broker
from src.schemas.observer.schema_listener import SchemaListener
from src.schemas.schema import Schema

T = TypeVar('T', bound=Schema)


class SchemaBroker(Broker, Generic[T]):
    """Publisher of schema objects in the observer pattern."""

    def __init__(self):
        """Initializes a SchemaBroker instance."""
        self._subscribers: List[SchemaListener[T]] = []

    def subscribe(self, subscriber: SchemaListener[T]) -> None:
        """
        Subscribes a listener to receive updates about component.

        Args:
            subscriber (SchemaListener[T]): the listener to subscribe
        """
        self._subscribers.append(subscriber)

    def unsubscribe(self, subscriber: SchemaListener[T]) -> None:
        """
        Unsubscribes a listener to stop receiving updates.

        Args:
            subscriber (SchemaListener[T]): the listener to unsubscribe
        """
        if subscriber in self._subscribers:
            self._subscribers.remove(subscriber)

    def notify(self, schema: T) -> None:
        for subscriber in self._subscribers:
            subscriber.new_schema(schema)